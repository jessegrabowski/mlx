// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "mlx/compile_impl.h"
#include "mlx/loop.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mlx::core {

namespace {

// Infer the loop length from the scanned inputs' shared leading axis
int loop_length(const std::vector<array>& xs) {
  if (xs.empty()) {
    throw std::invalid_argument(
        "[scan] At least one scanned input is required.");
  }
  for (auto& x : xs) {
    if (x.ndim() == 0) {
      throw std::invalid_argument(
          "[scan] Scanned inputs must have at least one dimension.");
    }
  }
  int length = xs[0].shape(0);
  for (auto& x : xs) {
    if (x.shape(0) != length) {
      std::ostringstream msg;
      msg << "[scan] Scanned inputs have mismatched lengths along axis 0: "
          << length << " and " << x.shape(0) << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  return length;
}

// Order the outputs of a tangent body as [carry..., carry_tangent...,
// y..., y_tangent...]
std::vector<array> interleave_tangents(
    const std::vector<array>& outs,
    const std::vector<array>& tangents,
    int num_carry) {
  std::vector<array> result(outs.begin(), outs.begin() + num_carry);
  result.insert(result.end(), tangents.begin(), tangents.begin() + num_carry);
  result.insert(result.end(), outs.begin() + num_carry, outs.end());
  result.insert(result.end(), tangents.begin() + num_carry, tangents.end());
  return result;
}

// The shape of `length` stacked rows of shape `row`
Shape stacked_shape(Shape row, int length) {
  row.insert(row.begin(), length);
  return row;
}

// Reverse an array along its leading (step) axis
array reverse_steps(const array& a, StreamOrDevice s) {
  Shape start(a.ndim(), 0);
  start[0] = a.shape(0) - 1;
  Shape stop = a.shape();
  stop[0] = -a.shape(0) - 1;
  Shape strides(a.ndim(), 1);
  strides[0] = -1;
  return slice(a, std::move(start), std::move(stop), std::move(strides), s);
}

bool is_stop_gradient(const array& a) {
  if (!a.has_primitive()) {
    return false;
  }
  auto& p = a.primitive();
  return typeid(p) == typeid(StopGradient);
}

} // namespace

Scan::Scan(
    Stream stream,
    std::vector<array> inputs,
    std::vector<array> outputs,
    std::vector<array> tape,
    int num_consts,
    int num_carry,
    int length,
    bool fuse)
    : Primitive(stream),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      tape_(std::move(tape)),
      num_consts_(num_consts),
      num_carry_(num_carry),
      length_(length),
      fuse_(fuse) {}

std::vector<array> Scan::replay(const std::vector<array>& args) const {
  return detail::compile_replace_loop_body(inputs_, tape_, outputs_, args);
}

std::vector<array> Scan::replay_at(
    std::vector<array> args,
    const std::vector<int>& positions,
    const std::vector<array>& values) const {
  for (size_t k = 0; k < values.size(); ++k) {
    args[positions[k]] = values[k];
  }
  return replay(args);
}

std::vector<bool> Scan::stopped_outputs(
    const std::vector<array>& primals,
    const std::vector<int>& const_positions) const {
  std::vector<bool> stopped;
  for (auto& out : outputs_) {
    auto in = std::find_if(inputs_.begin(), inputs_.end(), [&](auto& a) {
      return a.id() == out.id();
    });
    int i = in - inputs_.begin();
    if (in == inputs_.end()) {
      stopped.push_back(is_stop_gradient(out));
    } else if (i < num_consts_) {
      bool selected =
          std::find(const_positions.begin(), const_positions.end(), i) !=
          const_positions.end();
      stopped.push_back(!selected && is_stop_gradient(primals[i]));
    } else {
      stopped.push_back(false);
    }
  }
  return stopped;
}

std::vector<array> Scan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  // The new scan carries [carry..., carry_tangent...] and scans
  // [x..., x_tangent...]. It also computes the primals again, because jvp
  // returns only tangents. Every carry gets a tangent, because the tangent of
  // a const or an x reaches the carries.
  int first_x = num_consts_ + num_carry_;
  int num_xs = primals.size() - first_x;
  std::vector<array> init(
      primals.begin() + num_consts_, primals.begin() + first_x);
  for (int i = 0; i < num_carry_; ++i) {
    init.push_back(zeros_like(init[i], stream()));
  }
  std::vector<array> xs(primals.begin() + first_x, primals.end());
  std::vector<std::optional<array>> const_tangents(num_consts_);
  std::vector<int> tangent_xs;
  for (size_t k = 0; k < argnums.size(); ++k) {
    int i = argnums[k];
    if (i < num_consts_) {
      const_tangents[i] = tangents[k];
    } else if (i < first_x) {
      // A carry tangent must keep the dtype of its carry
      init[num_carry_ + i - num_consts_] =
          astype(tangents[k], primals[i].dtype(), stream());
    } else {
      tangent_xs.push_back(i - first_x);
      xs.push_back(tangents[k]);
    }
  }

  auto body = [&](const std::vector<array>& args) {
    // args are [carry..., carry_tangent..., x..., x_tangent...]
    auto inputs = primals;
    std::vector<int> positions;
    std::vector<array> body_primals;
    std::vector<array> body_tangents;
    auto differentiate = [&](int i, const array& primal, const array& tangent) {
      positions.push_back(i);
      body_primals.push_back(primal);
      body_tangents.push_back(tangent);
    };
    for (int i = 0; i < num_consts_; ++i) {
      if (const_tangents[i]) {
        differentiate(i, primals[i], *const_tangents[i]);
      }
    }
    for (int i = 0; i < num_carry_; ++i) {
      differentiate(num_consts_ + i, args[i], args[num_carry_ + i]);
    }
    for (int j = 0; j < num_xs; ++j) {
      inputs[first_x + j] = args[2 * num_carry_ + j];
    }
    for (size_t k = 0; k < tangent_xs.size(); ++k) {
      int j = tangent_xs[k];
      differentiate(
          first_x + j,
          args[2 * num_carry_ + j],
          args[2 * num_carry_ + num_xs + k]);
    }

    auto f = [&](const std::vector<array>& values) {
      return replay_at(inputs, positions, values);
    };
    auto [outs, outs_tangents] = mlx::core::jvp(f, body_primals, body_tangents);
    return interleave_tangents(outs, outs_tangents, num_carry_);
  };
  auto [carry, ys] = scan(body, init, xs, stream());

  int num_ys = outputs_.size() - num_carry_;
  std::vector<array> out_tangents(carry.begin() + num_carry_, carry.end());
  out_tangents.insert(out_tangents.end(), ys.begin() + num_ys, ys.end());
  return out_tangents;
}

std::vector<array> Scan::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  // A forward scan stacks the carry that enters each step. A backward scan
  // runs over the reversed steps. It carries the carry cotangents and one sum
  // per selected const, and its ys are the x cotangents.
  int first_x = num_consts_ + num_carry_;
  int num_xs = primals.size() - first_x;
  std::vector<array> init(
      primals.begin() + num_consts_, primals.begin() + first_x);
  std::vector<array> xs(primals.begin() + first_x, primals.end());

  std::vector<array> step_carries;
  if (num_carry_ > 0) {
    auto forward = [&](const std::vector<array>& args) {
      auto inputs = primals;
      std::copy(args.begin(), args.end(), inputs.begin() + num_consts_);
      auto outs = replay(inputs);
      outs.erase(outs.begin() + num_carry_, outs.end());
      outs.insert(outs.end(), args.begin(), args.begin() + num_carry_);
      return outs;
    };
    step_carries = scan(forward, init, xs, stream()).second;
  }

  std::vector<int> const_positions;
  std::vector<int> x_positions;
  for (int i : argnums) {
    if (i < num_consts_) {
      const_positions.push_back(i);
    } else if (i >= first_x) {
      x_positions.push_back(i);
    }
  }

  std::vector<array> back_init(
      cotangents.begin(), cotangents.begin() + num_carry_);
  for (int i : const_positions) {
    back_init.push_back(zeros_like(primals[i], stream()));
  }
  std::vector<array> back_xs;
  for (auto& a : step_carries) {
    back_xs.push_back(reverse_steps(a, stream()));
  }
  for (auto& x : xs) {
    back_xs.push_back(reverse_steps(x, stream()));
  }
  for (size_t i = num_carry_; i < cotangents.size(); ++i) {
    back_xs.push_back(reverse_steps(cotangents[i], stream()));
  }

  auto stopped = stopped_outputs(primals, const_positions);
  std::vector<int> positions = const_positions;
  for (int i = num_consts_; i < first_x; ++i) {
    positions.push_back(i);
  }
  positions.insert(positions.end(), x_positions.begin(), x_positions.end());

  // The backward args are [carry_cotangent..., sum..., carry..., x...,
  // y_cotangent...]
  int num_sums = const_positions.size();
  int carry_at = num_carry_ + num_sums;
  int y_cotangent_at = carry_at + num_carry_ + num_xs;
  auto backward = [&](const std::vector<array>& args) {
    auto inputs = primals;
    std::copy(
        args.begin() + carry_at,
        args.begin() + y_cotangent_at,
        inputs.begin() + num_consts_);
    std::vector<array> body_primals;
    for (int i : positions) {
      body_primals.push_back(inputs[i]);
    }
    std::vector<array> body_cotangents;
    for (int c = 0; c < num_carry_; ++c) {
      if (!stopped[c]) {
        body_cotangents.push_back(args[c]);
      }
    }
    for (size_t k = num_carry_; k < outputs_.size(); ++k) {
      if (!stopped[k]) {
        body_cotangents.push_back(args[y_cotangent_at + k - num_carry_]);
      }
    }

    auto f = [&](const std::vector<array>& values) {
      return replay_at(inputs, positions, values);
    };
    auto grads = mlx::core::vjp(f, body_primals, body_cotangents).second;

    // Carries must keep their dtype, so cast each cotangent to its primal
    std::vector<array> outs;
    for (int c = 0; c < num_carry_; ++c) {
      outs.push_back(astype(
          grads[num_sums + c], primals[num_consts_ + c].dtype(), stream()));
    }
    for (int k = 0; k < num_sums; ++k) {
      outs.push_back(
          add(args[num_carry_ + k],
              astype(grads[k], primals[const_positions[k]].dtype(), stream()),
              stream()));
    }
    outs.insert(outs.end(), grads.begin() + num_sums + num_carry_, grads.end());
    return outs;
  };
  auto [back_carry, back_ys] = scan(backward, back_init, back_xs, stream());

  std::vector<array> vjps;
  int next_sum = 0;
  int next_x = 0;
  for (int i : argnums) {
    if (i < num_consts_) {
      vjps.push_back(back_carry[num_carry_ + next_sum++]);
    } else if (i < first_x) {
      vjps.push_back(back_carry[i - num_consts_]);
    } else {
      vjps.push_back(reverse_steps(back_ys[next_x++], stream()));
    }
  }
  return vjps;
}

std::pair<std::vector<array>, std::vector<int>> Scan::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  // The new scan batches every carry at axis 0, and every x at axis 1 so that
  // a step slice is batched at axis 0. An unbatched init is broadcast, because
  // a carry keeps its shape and a batched input can reach any carry.
  int batch_size = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (axes[i] >= 0) {
      batch_size = inputs[i].shape(axes[i]);
    }
  }
  int first_x = num_consts_ + num_carry_;
  std::vector<array> init;
  for (int i = num_consts_; i < first_x; ++i) {
    if (axes[i] >= 0) {
      init.push_back(moveaxis(inputs[i], axes[i], 0, stream()));
    } else {
      Shape shape = inputs[i].shape();
      shape.insert(shape.begin(), batch_size);
      init.push_back(broadcast_to(inputs[i], shape, stream()));
    }
  }
  std::vector<array> xs;
  for (size_t i = first_x; i < inputs.size(); ++i) {
    xs.push_back(
        axes[i] >= 0 ? moveaxis(inputs[i], axes[i], 1, stream()) : inputs[i]);
  }

  auto body = [&](const std::vector<array>& args) {
    // args are [carry..., x...]. Vmap does not batch captured arrays, so the
    // consts are explicit inputs.
    std::vector<array> body_inputs(
        inputs.begin(), inputs.begin() + num_consts_);
    std::vector<int> in_axes(axes.begin(), axes.begin() + num_consts_);
    for (int i = 0; i < num_carry_; ++i) {
      body_inputs.push_back(args[i]);
      in_axes.push_back(0);
    }
    for (size_t j = 0; j < xs.size(); ++j) {
      body_inputs.push_back(args[num_carry_ + j]);
      in_axes.push_back(axes[first_x + j] >= 0 ? 0 : -1);
    }
    auto f = [this](const std::vector<array>& a) { return replay(a); };
    return mlx::core::vmap(
        f, in_axes, std::vector<int>(outputs_.size(), 0))(body_inputs);
  };
  auto [carry, ys] = scan(body, init, xs, stream());

  std::vector<int> out_axes(carry.size(), 0);
  out_axes.resize(carry.size() + ys.size(), 1);
  carry.insert(carry.end(), ys.begin(), ys.end());
  return {std::move(carry), std::move(out_axes)};
}

std::vector<Shape> Scan::output_shapes(const std::vector<array>& inputs) {
  // The body is traced at fixed shapes, so a shapeless compile can reuse a
  // scan only at the traced shapes
  size_t first_x = num_consts_ + num_carry_;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto traced = i < first_x ? inputs_[i].shape()
                              : stacked_shape(inputs_[i].shape(), length_);
    if (inputs[i].shape() != traced) {
      throw std::invalid_argument(
          "[scan] Shapeless compile cannot change the shape of scan inputs.");
    }
  }
  std::vector<Shape> shapes;
  for (int i = 0; i < num_carry_; ++i) {
    shapes.push_back(outputs_[i].shape());
  }
  for (size_t i = num_carry_; i < outputs_.size(); ++i) {
    shapes.push_back(stacked_shape(outputs_[i].shape(), length_));
  }
  return shapes;
}

const Scan::FusedBody& Scan::fused_body() const {
  std::call_once(fuse_flag_, [this]() {
    fused_body_ =
        detail::compile_fuse_loop_body(inputs_, tape_, outputs_, fuse_);
  });
  return fused_body_;
}

std::pair<std::vector<array>, std::vector<array>> scan(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& init,
    const std::vector<array>& xs,
    StreamOrDevice s /* = {} */) {
  int length = loop_length(xs);
  if (length == 0) {
    throw std::invalid_argument("[scan] Zero-length scan is not supported.");
  }

  auto input_specs = init;
  for (auto& x : xs) {
    Shape slice_shape(x.shape().begin() + 1, x.shape().end());
    input_specs.push_back(
        array(std::move(slice_shape), x.dtype(), nullptr, {}));
  }

  // Trace on the scan stream, so the replay runs the body there
  auto stream = to_stream(s);
  detail::LoopBodyTrace trace;
  {
    StreamContext body_stream(stream);
    trace = detail::compile_loop_body(fun, input_specs);
  }
  int num_carry = init.size();
  detail::compile_validate_loop_carry(trace, num_carry);
  if (trace.outputs.empty()) {
    throw std::invalid_argument(
        "[scan] The loop body must return at least one array.");
  }

  // The replay runs all body ops on the scan stream, so reject other streams
  std::unordered_set<std::uintptr_t> leaf_ids;
  for (auto& in : trace.inputs) {
    leaf_ids.insert(in.id());
  }
  for (auto& c : trace.consts) {
    leaf_ids.insert(c.id());
  }
  for (auto& node : trace.tape) {
    if (leaf_ids.find(node.id()) == leaf_ids.end() &&
        node.primitive().stream() != stream) {
      throw std::invalid_argument(
          "[scan] The loop body must run entirely on the scan's stream.");
    }
  }

  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  for (auto& c : init) {
    shapes.push_back(c.shape());
    dtypes.push_back(c.dtype());
  }
  for (size_t i = num_carry; i < trace.outputs.size(); ++i) {
    auto& y = trace.outputs[i];
    shapes.push_back(stacked_shape(y.shape(), length));
    dtypes.push_back(y.dtype());
  }

  // Trace-side and node inputs pair up positionally, consts first
  int num_consts = trace.consts.size();
  auto trace_inputs = trace.consts;
  trace_inputs.insert(
      trace_inputs.end(), trace.inputs.begin(), trace.inputs.end());
  auto node_inputs = std::move(trace.consts);
  node_inputs.insert(node_inputs.end(), init.begin(), init.end());
  node_inputs.insert(node_inputs.end(), xs.begin(), xs.end());

  auto node_outputs = array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<Scan>(
          stream,
          std::move(trace_inputs),
          std::move(trace.outputs),
          std::move(trace.tape),
          num_consts,
          num_carry,
          length,
          trace.fuse),
      std::move(node_inputs));

  std::vector<array> final_carry(
      node_outputs.begin(), node_outputs.begin() + num_carry);
  std::vector<array> stacked_ys(
      node_outputs.begin() + num_carry, node_outputs.end());
  return {std::move(final_carry), std::move(stacked_ys)};
}

} // namespace mlx::core
