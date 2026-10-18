// Copyright © 2026 Apple Inc.

#include <cstring>
#include <optional>

#include "mlx/allocator.h"
#include "mlx/backend/common/scan_loop.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// Shape/strides for a single row so the copy kernels never see a rank-0 view
std::pair<Shape, Strides> row_layout(const array& row_like) {
  if (row_like.ndim() == 0) {
    return {Shape{1}, Strides{1}};
  }
  return {row_like.shape(), make_contiguous_strides(row_like.shape())};
}

struct LoopSizes {
  int num_consts;
  int num_carry;
  int length;
  bool reverse;
};

// Run all steps of a scan inside a task of stream `s`
void run_loop(
    const ScanPlan& plan,
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const LoopSizes& sizes,
    Stream s) {
  auto [num_consts, num_carry, length, reverse] = sizes;

  // Use an encoder of this thread, so no other thread sees the inline mode
  auto& encoder =
      cpu::get_command_encoders().try_emplace(s.index, s).first->second;
  struct InlineDispatch {
    cpu::CommandEncoder& encoder;
    bool was_inline = encoder.set_inline(true);
    ~InlineDispatch() {
      encoder.set_inline(was_inline);
    }
  } inline_dispatch{encoder};

  std::vector<std::optional<array>> env(plan.num_slots);
  for (int i = 0; i < num_consts + num_carry; ++i) {
    env[plan.input_slots[i]] = inputs[i];
  }

  int num_ys = outputs.size() - num_carry;
  std::vector<array> node_inputs;
  std::vector<array> new_carries;
  for (int t = 0; t < length; ++t) {
    int row_index = reverse ? length - 1 - t : t;
    for (size_t j = num_consts + num_carry; j < inputs.size(); ++j) {
      auto& x = inputs[j];
      Shape row_shape(x.shape().begin() + 1, x.shape().end());
      array row(row_shape, x.dtype(), nullptr, {});

      if (x.flags().row_contiguous) {
        row.copy_shared_buffer(
            x,
            Strides(x.strides().begin() + 1, x.strides().end()),
            x.flags(),
            row.size(),
            row_index * x.strides(0));
      } else {
        row.set_data(allocator::malloc(row.nbytes()));
        auto [data_shape, row_strides] = row_layout(row);
        Strides x_row_strides(x.strides().begin() + 1, x.strides().end());
        if (x_row_strides.empty()) {
          x_row_strides = Strides{1};
        }
        copy_cpu_inplace(
            /* src = */ x,
            /* dst = */ row,
            /* data_shape = */ data_shape,
            /* i_strides = */ x_row_strides,
            /* o_strides = */ row_strides,
            /* i_offset = */ row_index * x.strides(0),
            /* o_offset = */ 0,
            CopyType::GeneralGeneral,
            s);
      }
      env[plan.input_slots[j]] = std::move(row);
    }

    for (auto& step : plan.steps) {
      node_inputs.clear();
      for (int slot : step.input_slots) {
        node_inputs.push_back(env[slot].value());
      }
      // Release bindings at their last use, so the node can donate them
      for (int i : step.released_inputs) {
        env[step.input_slots[i]].reset();
      }

      auto node_outputs = array::make_arrays(
          step.shapes, step.dtypes, step.node->primitive_ptr(), {});
      step.node->primitive().eval_cpu(node_inputs, node_outputs);
      // The kernels already ran, so their temporaries can go
      encoder.temporaries().clear();

      for (size_t k = 0; k < node_outputs.size(); ++k) {
        env[step.output_slots[k]] = std::move(node_outputs[k]);
      }
    }

    for (int j = 0; j < num_ys; ++j) {
      auto& y = env[plan.output_slots[num_carry + j]].value();
      auto& out = outputs[num_carry + j];
      if (y.flags().row_contiguous && y.dtype() == out.dtype()) {
        std::memcpy(
            out.data<char>() + row_index * y.nbytes(),
            y.data<char>(),
            y.nbytes());
        continue;
      }
      auto [data_shape, out_strides] = row_layout(y);
      auto y_strides = y.ndim() == 0 ? Strides{1} : y.strides();
      copy_cpu_inplace(
          /* src = */ y,
          /* dst = */ out,
          /* data_shape = */ data_shape,
          /* i_strides = */ y_strides,
          /* o_strides = */ out_strides,
          /* i_offset = */ 0,
          /* o_offset = */ row_index * static_cast<int64_t>(y.size()),
          CopyType::GeneralGeneral,
          s);
    }

    // Read all new carries before rebinding, because a body can permute them
    new_carries.clear();
    for (int i = 0; i < num_carry; ++i) {
      new_carries.push_back(env[plan.output_slots[i]].value());
    }
    for (int i = 0; i < num_carry; ++i) {
      env[plan.input_slots[num_consts + i]] = std::move(new_carries[i]);
    }

    // Release stale outputs, so the next iteration can donate the carries
    for (int slot : plan.node_output_slots) {
      env[slot].reset();
    }
  }

  for (int i = 0; i < num_carry; ++i) {
    auto& carry = env[plan.input_slots[num_consts + i]].value();
    copy_cpu_inplace(carry, outputs[i], CopyType::GeneralGeneral, s);
  }
}

} // namespace

void Scan::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto s = stream();
  auto& encoder = cpu::get_command_encoder(s);

  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  auto& body = fused_body();
  auto plan = make_scan_plan(body.inputs, body.tape, body.outputs, num_consts_);
  std::vector<array> weak_outputs;
  for (auto& out : outputs) {
    weak_outputs.push_back(array::unsafe_weak_copy(out));
  }

  // Run the whole loop in one task, with the body kernels run at once. The
  // primitive owns the body that the plan points into.
  encoder.dispatch(
      [primitive = outputs[0].primitive_ptr(),
       plan = std::move(plan),
       inputs = inputs,
       outputs = std::move(weak_outputs),
       sizes = LoopSizes{num_consts_, num_carry_, length_, reverse_},
       s]() mutable { run_loop(plan, inputs, outputs, sizes, s); });
}

} // namespace mlx::core
