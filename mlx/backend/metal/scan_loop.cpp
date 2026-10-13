// Copyright © 2026 Apple Inc.

#include <unordered_map>
#include <unordered_set>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

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

void Scan::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto s = stream();
  auto& encoder = metal::get_command_encoder(s);

  // Metal does not retain buffers, so keep them until the command buffer
  // completes
  auto retain = [&](std::vector<array> keep) {
    encoder.get_command_buffer()->addCompletedHandler(
        [keep = std::move(keep)](MTL::CommandBuffer*) {});
  };

  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  std::unordered_map<std::uintptr_t, array> env;
  std::unordered_set<std::uintptr_t> leaf_ids;
  for (auto& in : inputs_) {
    leaf_ids.insert(in.id());
  }
  for (int i = 0; i < num_consts_ + num_carry_; ++i) {
    env.insert_or_assign(inputs_[i].id(), inputs[i]);
  }

  // Keep these bindings for the whole iteration. Release all others at
  // their last tape use, so body ops can donate them.
  std::unordered_set<std::uintptr_t> kept_ids;
  for (auto& o : outputs_) {
    kept_ids.insert(o.id());
  }
  for (int i = 0; i < num_consts_; ++i) {
    kept_ids.insert(inputs_[i].id());
  }
  std::unordered_map<std::uintptr_t, size_t> last_use;
  std::vector<std::uintptr_t> replay_output_ids;
  for (size_t idx = 0; idx < tape_.size(); ++idx) {
    if (leaf_ids.find(tape_[idx].id()) != leaf_ids.end()) {
      continue;
    }
    for (auto& in : tape_[idx].inputs()) {
      last_use[in.id()] = idx;
    }
    for (auto& o : tape_[idx].outputs()) {
      replay_output_ids.push_back(o.id());
    }
  }

  // Release replaced bindings when the command buffer completes
  std::vector<array> retired;
  auto bind = [&](std::uintptr_t id, array value) {
    if (auto it = env.find(id); it != env.end()) {
      retired.push_back(std::move(it->second));
      it->second = std::move(value);
    } else {
      env.emplace(id, std::move(value));
    }
  };

  int num_ys = outputs.size() - num_carry_;
  try {
    for (int t = 0; t < length_; ++t) {
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
              t * x.strides(0));
        } else {
          row.set_data(allocator::malloc(row.nbytes()));
          auto [data_shape, row_strides] = row_layout(row);
          Strides x_row_strides(x.strides().begin() + 1, x.strides().end());
          if (x_row_strides.empty()) {
            x_row_strides = Strides{1};
          }
          // The general copy kernel has no rank-0 variant
          auto ctype = x_row_strides == row_strides ? CopyType::Vector
                                                    : CopyType::GeneralGeneral;
          copy_gpu_inplace(
              /* in = */ x,
              /* out = */ row,
              /* data_shape = */ data_shape,
              /* i_strides = */ x_row_strides,
              /* o_strides = */ row_strides,
              /* i_offset = */ t * x.strides(0),
              /* o_offset = */ 0,
              ctype,
              s);
        }
        bind(inputs_[j].id(), std::move(row));
      }

      // env.at cannot fail: compile_loop_body closes the tape over its inputs
      for (size_t idx = 0; idx < tape_.size(); ++idx) {
        auto& node = tape_[idx];
        if (leaf_ids.find(node.id()) != leaf_ids.end()) {
          continue;
        }
        std::vector<array> node_inputs;
        node_inputs.reserve(node.inputs().size());
        for (auto& in : node.inputs()) {
          node_inputs.push_back(env.at(in.id()));
        }
        // Erase bindings at their last use, so the op can donate them
        std::unordered_set<std::uintptr_t> dropped_ids;
        for (auto& in : node.inputs()) {
          auto id = in.id();
          if (kept_ids.find(id) == kept_ids.end()) {
            if (auto it = last_use.find(id);
                it != last_use.end() && it->second == idx) {
              if (env.erase(id) > 0) {
                dropped_ids.insert(id);
              }
            }
          }
        }

        auto trace_outs = node.outputs();
        std::vector<Shape> shapes;
        std::vector<Dtype> dtypes;
        for (auto& o : trace_outs) {
          shapes.push_back(o.shape());
          dtypes.push_back(o.dtype());
        }
        auto node_outputs = array::make_arrays(
            std::move(shapes), dtypes, node.primitive_ptr(), {});

        try {
          node.primitive().eval_gpu(node_inputs, node_outputs);
        } catch (...) {
          // Keep the dropped buffers alive for the recorded kernels
          retain(std::move(node_inputs));
          throw;
        }

        // Keep dropped buffers alive until the command buffer completes,
        // except buffers donated into the outputs
        std::vector<array> dropped;
        for (size_t i = 0; i < node_inputs.size(); ++i) {
          if (dropped_ids.find(node.inputs()[i].id()) != dropped_ids.end()) {
            dropped.push_back(node_inputs[i]);
          }
        }
        for (auto& out : node_outputs) {
          auto same_data = [&out](const array& a) {
            return a.data_shared_ptr() == out.data_shared_ptr();
          };
          dropped.erase(
              std::remove_if(dropped.begin(), dropped.end(), same_data),
              dropped.end());
        }
        if (!dropped.empty()) {
          retain(std::move(dropped));
        }

        for (size_t k = 0; k < node_outputs.size(); ++k) {
          bind(trace_outs[k].id(), std::move(node_outputs[k]));
        }
      }

      // Copy into a row view, because the vector kernel sizes its dispatch
      // from the destination
      for (int j = 0; j < num_ys; ++j) {
        auto& y = env.at(outputs_[num_carry_ + j].id());
        auto& stacked = outputs[num_carry_ + j];
        auto [data_shape, row_strides] = row_layout(y);
        array out_row(data_shape, y.dtype(), nullptr, {});
        out_row.copy_shared_buffer(
            stacked,
            row_strides,
            stacked.flags(),
            y.size(),
            t * static_cast<int64_t>(y.size()));
        if (y.flags().row_contiguous) {
          copy_gpu_inplace(y, out_row, CopyType::Vector, s);
        } else {
          auto y_strides = y.strides();
          copy_gpu_inplace(
              /* in = */ y,
              /* out = */ out_row,
              /* data_shape = */ data_shape,
              /* i_strides = */ y_strides,
              /* o_strides = */ row_strides,
              /* i_offset = */ 0,
              /* o_offset = */ 0,
              CopyType::GeneralGeneral,
              s);
        }
      }

      // Read all new carries before rebinding, because a body can permute them
      std::vector<array> new_carries;
      new_carries.reserve(num_carry_);
      for (int i = 0; i < num_carry_; ++i) {
        new_carries.push_back(env.at(outputs_[i].id()));
      }
      for (int i = 0; i < num_carry_; ++i) {
        bind(inputs_[num_consts_ + i].id(), std::move(new_carries[i]));
      }

      // Erase stale outputs, so the next iteration can donate the carry
      // buffers. Erase carry aliases without a retire.
      std::unordered_set<std::uintptr_t> carried_value_ids;
      for (int i = 0; i < num_carry_; ++i) {
        carried_value_ids.insert(env.at(inputs_[num_consts_ + i].id()).id());
      }
      for (auto o_id : replay_output_ids) {
        if (auto it = env.find(o_id); it != env.end()) {
          if (carried_value_ids.find(it->second.id()) ==
              carried_value_ids.end()) {
            retired.push_back(std::move(it->second));
          }
          env.erase(it);
        }
      }

      if (!retired.empty()) {
        retain(std::move(retired));
        retired.clear();
      }

      // Commit during the scan, so the GPU starts before encoding ends
      if (encoder.needs_commit()) {
        encoder.end_encoding();
        scheduler::notify_new_task(s);
        encoder.commit([s]() { scheduler::notify_task_completion(s); });
      }
    }

    for (int i = 0; i < num_carry_; ++i) {
      auto& carry = env.at(inputs_[num_consts_ + i].id());
      auto ctype = carry.flags().row_contiguous ? CopyType::Vector
                                                : CopyType::GeneralGeneral;
      copy_gpu_inplace(carry, outputs[i], ctype, s);
    }
  } catch (...) {
    // Keep the live bindings for the recorded kernels, then rethrow
    std::vector<array> keep;
    keep.reserve(env.size() + retired.size());
    for (auto& [id, value] : env) {
      keep.push_back(std::move(value));
    }
    keep.insert(
        keep.end(),
        std::make_move_iterator(retired.begin()),
        std::make_move_iterator(retired.end()));
    retain(std::move(keep));
    throw;
  }

  // Keep the live bindings until the command buffer completes
  std::vector<array> remaining;
  remaining.reserve(env.size());
  for (auto& [id, value] : env) {
    remaining.push_back(std::move(value));
  }
  retain(std::move(remaining));
}

} // namespace mlx::core
