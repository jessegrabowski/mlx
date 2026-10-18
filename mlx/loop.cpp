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

// A placeholder shaped like one row of the stacked array `a`
array row_placeholder(const array& a) {
  return array(
      Shape(a.shape().begin() + 1, a.shape().end()), a.dtype(), nullptr, {});
}

// Whether the backward part of `split` reads an array in [begin, end)
bool backward_reads_any(
    const detail::LoopBodySplit& split,
    const std::vector<array>& backward_outputs,
    std::vector<array>::const_iterator begin,
    std::vector<array>::const_iterator end) {
  std::unordered_set<std::uintptr_t> reads;
  for (auto& a : split.backward_tape) {
    for (auto& in : a.inputs()) {
      reads.insert(in.id());
    }
  }
  for (auto& out : backward_outputs) {
    reads.insert(out.id());
  }
  return std::any_of(
      begin, end, [&reads](const array& a) { return reads.count(a.id()); });
}

// The terms of `a` that mlx vjp adds up, for example when a const is read
// more than once
std::vector<array> additive_terms(const array& a) {
  if (!a.has_primitive()) {
    return {a};
  }
  auto& p = a.primitive();
  if (typeid(p) != typeid(Add) || a.inputs()[0].shape() != a.shape() ||
      a.inputs()[1].shape() != a.shape()) {
    return {a};
  }
  auto terms = additive_terms(a.inputs()[0]);
  auto right = additive_terms(a.inputs()[1]);
  terms.insert(terms.end(), right.begin(), right.end());
  return terms;
}

// The two factors of a matrix product of two matrices
std::optional<std::pair<array, array>> matmul_factors(const array& a) {
  if (!a.has_primitive()) {
    return std::nullopt;
  }
  auto& p = a.primitive();
  if (typeid(p) != typeid(Matmul) || a.inputs()[0].ndim() != 2 ||
      a.inputs()[1].ndim() != 2) {
    return std::nullopt;
  }
  return std::make_pair(a.inputs()[0], a.inputs()[1]);
}

// One step of a const cotangent: the carried sum, plus each product whose
// factors are smaller than the product
struct ConstCotangentStep {
  array sum;
  std::vector<std::pair<array, array>> stacked_products;
  bool has_carried_terms = false;
};

ConstCotangentStep add_const_cotangent(
    array sum,
    const array& cotangent,
    Dtype dtype,
    StreamOrDevice s) {
  ConstCotangentStep step{std::move(sum)};
  for (auto& term : additive_terms(cotangent)) {
    auto product = matmul_factors(term);
    if (product) {
      auto& [a, b] = *product;
      int64_t m = a.shape(0);
      int64_t inner = a.shape(1);
      int64_t n = b.shape(1);
      if (inner * (m + n) < m * n) {
        step.stacked_products.push_back(*product);
        continue;
      }
      if (term.dtype() == dtype) {
        step.sum = addmm(step.sum, a, b, 1.0f, 1.0f, s);
        step.has_carried_terms = true;
        continue;
      }
    }
    step.sum = add(step.sum, astype(term, dtype, s), s);
    step.has_carried_terms = true;
  }
  return step;
}

// The sum over steps of A_t B_t: A stacked (L, m, inner) and B stacked
// (L, inner, n) give one (m, L inner) by (L inner, n) matmul
array sum_stacked_products(const array& a, const array& b, StreamOrDevice s) {
  auto a_rows =
      reshape(moveaxis(a, 0, 1, s), {a.shape(1), a.shape(0) * a.shape(2)}, s);
  auto b_rows = reshape(b, {b.shape(0) * b.shape(1), b.shape(2)}, s);
  return matmul(a_rows, b_rows, s);
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
    bool reverse,
    bool fuse)
    : Primitive(stream),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      tape_(std::move(tape)),
      num_consts_(num_consts),
      num_carry_(num_carry),
      length_(length),
      reverse_(reverse),
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
  auto [carry, ys] = scan(body, init, xs, reverse_, stream());

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
  // One vjp step is traced once and split. A forward scan runs the forward
  // part and stacks the residuals. A backward scan runs the backward part in
  // the opposite order. It carries the carry cotangents and one sum per
  // selected const, and its ys are the x cotangents.
  int first_x = num_consts_ + num_carry_;
  int num_xs = primals.size() - first_x;
  std::vector<array> init(
      primals.begin() + num_consts_, primals.begin() + first_x);
  std::vector<array> xs(primals.begin() + first_x, primals.end());

  std::vector<int> const_positions;
  std::vector<int> x_positions;
  for (int i : argnums) {
    if (i < num_consts_) {
      const_positions.push_back(i);
    } else if (i >= first_x) {
      x_positions.push_back(i);
    }
  }

  auto stopped = stopped_outputs(primals, const_positions);
  std::vector<int> positions = const_positions;
  for (int i = num_consts_; i < first_x; ++i) {
    positions.push_back(i);
  }
  positions.insert(positions.end(), x_positions.begin(), x_positions.end());

  // The step args are [carry_cotangent..., sum..., carry..., x...,
  // y_cotangent...]
  int num_sums = const_positions.size();
  int carry_at = num_carry_ + num_sums;
  int x_at = carry_at + num_carry_;
  int y_cotangent_at = x_at + num_xs;
  std::vector<array> step_specs;
  auto add_spec = [&step_specs](const array& a) {
    step_specs.push_back(array(a.shape(), a.dtype(), nullptr, {}));
  };
  // A carry cotangent has the shape and dtype of its carry
  for (auto& c : init) {
    add_spec(c);
  }
  for (int i : const_positions) {
    add_spec(primals[i]);
  }
  for (auto& c : init) {
    add_spec(c);
  }
  for (auto& x : xs) {
    step_specs.push_back(row_placeholder(x));
  }
  for (size_t k = num_carry_; k < cotangents.size(); ++k) {
    step_specs.push_back(row_placeholder(cotangents[k]));
  }

  // A const cotangent term that is a product A_t B_t is not added to the
  // carried sum when the factors are smaller than the product. Its factors
  // are stacked, and one matmul after the scan sums the terms.
  std::vector<std::vector<int>> stacked_terms(num_sums);
  std::vector<bool> has_carried_terms(num_sums, false);

  // The step returns [next_carry..., carry_cotangent..., sum...,
  // x_cotangent..., stacked_factor...]
  auto step = [&](const std::vector<array>& args) {
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
    auto [body_outs, grads] = mlx::core::vjp(f, body_primals, body_cotangents);

    // Carries must keep their dtype, so cast each cotangent to its primal
    std::vector<array> outs(body_outs.begin(), body_outs.begin() + num_carry_);
    for (int c = 0; c < num_carry_; ++c) {
      outs.push_back(astype(
          grads[num_sums + c], primals[num_consts_ + c].dtype(), stream()));
    }
    std::vector<array> factors;
    int num_x_cotangents = grads.size() - num_sums - num_carry_;
    for (int k = 0; k < num_sums; ++k) {
      auto const_step = add_const_cotangent(
          args[num_carry_ + k],
          grads[k],
          primals[const_positions[k]].dtype(),
          stream());
      outs.push_back(const_step.sum);
      has_carried_terms[k] = const_step.has_carried_terms;
      for (auto& [a, b] : const_step.stacked_products) {
        stacked_terms[k].push_back(num_x_cotangents + factors.size());
        factors.push_back(a);
        factors.push_back(b);
      }
    }
    outs.insert(outs.end(), grads.begin() + num_sums + num_carry_, grads.end());
    outs.insert(outs.end(), factors.begin(), factors.end());
    return outs;
  };
  detail::LoopBodyTrace trace;
  {
    StreamContext step_stream(stream());
    trace = detail::compile_loop_body(step, step_specs);
  }
  std::vector<array> next_carry(
      trace.outputs.begin(), trace.outputs.begin() + num_carry_);
  std::vector<array> back_outputs(
      trace.outputs.begin() + num_carry_, trace.outputs.end());
  auto split = detail::compile_split_loop_body(trace, next_carry, back_outputs);

  // The backward part reads a carry or an x only if a vjp rule needs it
  auto step_input = [&trace](int i) { return trace.inputs.cbegin() + i; };
  bool back_reads_carry = backward_reads_any(
      split, back_outputs, step_input(carry_at), step_input(x_at));
  // The xs also set the backward length when nothing else is scanned
  int num_residuals = split.residuals.size();
  bool back_scans_xs =
      backward_reads_any(
          split, back_outputs, step_input(x_at), step_input(y_cotangent_at)) ||
      (num_residuals == 0 && !back_reads_carry &&
       cotangents.size() == num_carry_);

  // Leaves are [const..., step arg...]
  auto leaves = trace.consts;
  leaves.insert(leaves.end(), trace.inputs.begin(), trace.inputs.end());
  auto bind = [&](const std::vector<array>& step_args) {
    auto bound = trace.consts;
    bound.insert(bound.end(), step_args.begin(), step_args.end());
    return bound;
  };

  std::vector<array> stacked_residuals;
  std::vector<array> stacked_carries;
  if (num_residuals > 0 || back_reads_carry) {
    // The cotangent args are not read by the forward part
    auto forward = [&](const std::vector<array>& args) {
      auto step_args = trace.inputs;
      std::copy(args.begin(), args.end(), step_args.begin() + carry_at);
      auto outs = next_carry;
      outs.insert(outs.end(), split.residuals.begin(), split.residuals.end());
      outs = detail::compile_replace_loop_body(
          leaves, split.forward_tape, outs, bind(step_args));
      if (back_reads_carry) {
        outs.insert(outs.end(), args.begin(), args.begin() + num_carry_);
      }
      return outs;
    };
    auto ys = scan(forward, init, xs, reverse_, stream()).second;
    stacked_residuals.assign(ys.begin(), ys.begin() + num_residuals);
    stacked_carries.assign(ys.begin() + num_residuals, ys.end());
  }

  // A carry cotangent must keep the dtype of its carry
  std::vector<array> back_init;
  for (int c = 0; c < num_carry_; ++c) {
    back_init.push_back(
        astype(cotangents[c], primals[num_consts_ + c].dtype(), stream()));
  }
  for (int i : const_positions) {
    back_init.push_back(zeros_like(primals[i], stream()));
  }
  auto back_xs = stacked_residuals;
  back_xs.insert(back_xs.end(), stacked_carries.begin(), stacked_carries.end());
  if (back_scans_xs) {
    back_xs.insert(back_xs.end(), xs.begin(), xs.end());
  }
  back_xs.insert(
      back_xs.end(), cotangents.begin() + num_carry_, cotangents.end());

  // The backward args are [carry_cotangent..., sum..., residual...,
  // carry..., x..., y_cotangent...], where the carries and xs are present
  // only when they are scanned
  auto back_leaves = leaves;
  back_leaves.insert(
      back_leaves.end(), split.residuals.begin(), split.residuals.end());
  auto backward = [&](const std::vector<array>& args) {
    auto step_args = trace.inputs;
    auto next = args.begin() + carry_at;
    std::copy(args.begin(), next, step_args.begin());
    std::vector<array> residuals(next, next + num_residuals);
    next += num_residuals;
    if (back_reads_carry) {
      std::copy(next, next + num_carry_, step_args.begin() + carry_at);
      next += num_carry_;
    }
    if (back_scans_xs) {
      std::copy(next, next + num_xs, step_args.begin() + x_at);
      next += num_xs;
    }
    std::copy(next, args.end(), step_args.begin() + y_cotangent_at);
    auto bound = bind(step_args);
    bound.insert(bound.end(), residuals.begin(), residuals.end());
    return detail::compile_replace_loop_body(
        back_leaves, split.backward_tape, back_outputs, bound);
  };
  auto [back_carry, back_ys] =
      scan(backward, back_init, back_xs, !reverse_, stream());

  // The cotangent of selected const k is its carried sum plus its stacked
  // products, which ys at and at + 1 hold
  auto const_cotangent = [&](int k) {
    auto dtype = primals[const_positions[k]].dtype();
    std::optional<array> total;
    if (has_carried_terms[k] || stacked_terms[k].empty()) {
      total = back_carry[num_carry_ + k];
    }
    for (int at : stacked_terms[k]) {
      auto term = astype(
          sum_stacked_products(back_ys[at], back_ys[at + 1], stream()),
          dtype,
          stream());
      total = total ? add(*total, term, stream()) : term;
    }
    return *total;
  };

  std::vector<array> vjps;
  int next_sum = 0;
  int next_x = 0;
  for (int i : argnums) {
    if (i < num_consts_) {
      vjps.push_back(const_cotangent(next_sum++));
    } else if (i < first_x) {
      vjps.push_back(back_carry[i - num_consts_]);
    } else {
      vjps.push_back(back_ys[next_x++]);
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
  auto [carry, ys] = scan(body, init, xs, reverse_, stream());

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
    bool reverse /* = false */,
    StreamOrDevice s /* = {} */) {
  int length = loop_length(xs);
  if (length == 0) {
    throw std::invalid_argument("[scan] Zero-length scan is not supported.");
  }

  auto input_specs = init;
  for (auto& x : xs) {
    input_specs.push_back(row_placeholder(x));
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
          reverse,
          trace.fuse),
      std::move(node_inputs));

  std::vector<array> final_carry(
      node_outputs.begin(), node_outputs.begin() + num_carry);
  std::vector<array> stacked_ys(
      node_outputs.begin() + num_carry, node_outputs.end());
  return {std::move(final_carry), std::move(stacked_ys)};
}

} // namespace mlx::core
