// Copyright © 2026 Apple Inc.

#include <algorithm>
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
  std::vector<std::optional<array>> env(plan.num_slots);
  for (int i = 0; i < num_consts_ + num_carry_; ++i) {
    env[plan.input_slots[i]] = inputs[i];
  }

  // Keep replaced and released buffers until the queued kernels run
  std::vector<array> retired;
  auto bind = [&](int slot, array value) {
    if (env[slot]) {
      retired.push_back(std::move(*env[slot]));
    }
    env[slot] = std::move(value);
  };

  int num_ys = outputs.size() - num_carry_;
  std::vector<array> node_inputs;
  try {
    for (int t = 0; t < length_; ++t) {
      int row_index = step_row(t);
      for (size_t j = num_consts_ + num_carry_; j < inputs_.size(); ++j) {
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
        bind(plan.input_slots[j], std::move(row));
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
        try {
          step.node->primitive().eval_cpu(node_inputs, node_outputs);
        } catch (...) {
          // Keep the released buffers alive for the queued kernels
          encoder.dispatch([keep = std::move(node_inputs)]() {});
          throw;
        }

        // Keep released buffers alive until the iteration's kernels finish,
        // except buffers donated into the outputs
        for (int i : step.released_inputs) {
          auto& in = node_inputs[i];
          auto donated = [&in](const array& out) {
            return out.data_shared_ptr() == in.data_shared_ptr();
          };
          if (std::none_of(node_outputs.begin(), node_outputs.end(), donated)) {
            retired.push_back(in);
          }
        }

        for (size_t k = 0; k < node_outputs.size(); ++k) {
          bind(step.output_slots[k], std::move(node_outputs[k]));
        }
      }

      for (int j = 0; j < num_ys; ++j) {
        auto& y = env[plan.output_slots[num_carry_ + j]].value();
        auto [data_shape, out_strides] = row_layout(y);
        auto y_strides = y.ndim() == 0 ? Strides{1} : y.strides();
        copy_cpu_inplace(
            /* src = */ y,
            /* dst = */ outputs[num_carry_ + j],
            /* data_shape = */ data_shape,
            /* i_strides = */ y_strides,
            /* o_strides = */ out_strides,
            /* i_offset = */ 0,
            /* o_offset = */ row_index * static_cast<int64_t>(y.size()),
            CopyType::GeneralGeneral,
            s);
      }

      // Read all new carries before rebinding, because a body can permute them
      std::vector<array> new_carries;
      new_carries.reserve(num_carry_);
      for (int i = 0; i < num_carry_; ++i) {
        new_carries.push_back(env[plan.output_slots[i]].value());
      }
      for (int i = 0; i < num_carry_; ++i) {
        bind(plan.input_slots[num_consts_ + i], std::move(new_carries[i]));
      }

      // Release stale outputs, so the next iteration can donate the carry
      // buffers. Release carry aliases without a retire.
      std::vector<std::uintptr_t> carried_ids;
      for (int i = 0; i < num_carry_; ++i) {
        carried_ids.push_back(
            env[plan.input_slots[num_consts_ + i]].value().id());
      }
      for (int slot : plan.node_output_slots) {
        auto& value = env[slot];
        if (value &&
            std::find(carried_ids.begin(), carried_ids.end(), value->id()) ==
                carried_ids.end()) {
          retired.push_back(std::move(*value));
        }
        value.reset();
      }

      // Drain the encoder temporaries of body ops, or the scan keeps all of
      // them until it ends
      auto& temps = encoder.temporaries();
      retired.insert(
          retired.end(),
          std::make_move_iterator(temps.begin()),
          std::make_move_iterator(temps.end()));
      temps.clear();

      if (!retired.empty()) {
        encoder.dispatch([batch = std::move(retired)]() {});
        retired.clear();
      }
    }

    for (int i = 0; i < num_carry_; ++i) {
      auto& carry = env[plan.input_slots[num_consts_ + i]].value();
      copy_cpu_inplace(carry, outputs[i], CopyType::GeneralGeneral, s);
    }
  } catch (...) {
    // Keep the live bindings for the queued kernels, then rethrow
    std::vector<array> keep;
    for (auto& value : env) {
      if (value) {
        keep.push_back(std::move(*value));
      }
    }
    keep.insert(
        keep.end(),
        std::make_move_iterator(retired.begin()),
        std::make_move_iterator(retired.end()));
    encoder.dispatch([keep = std::move(keep)]() {});
    throw;
  }

  // Keep the live bindings until all queued kernels run
  std::vector<array> remaining;
  for (auto& value : env) {
    if (value) {
      remaining.push_back(std::move(*value));
    }
  }
  encoder.dispatch([remaining = std::move(remaining)]() {});
}

} // namespace mlx::core
