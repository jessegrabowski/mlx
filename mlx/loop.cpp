// Copyright © 2026 Apple Inc.

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
      auto full = inputs;
      for (size_t k = 0; k < values.size(); ++k) {
        full[positions[k]] = values[k];
      }
      return replay(full);
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
    Shape stacked = y.shape();
    stacked.insert(stacked.begin(), length);
    shapes.push_back(std::move(stacked));
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
