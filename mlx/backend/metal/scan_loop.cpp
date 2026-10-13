// Copyright © 2026 Apple Inc.

#include <fmt/format.h>

#include <atomic>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "mlx/allocator.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/compile_impl.h"
#include "mlx/graph_utils.h"
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

// Run a chain of elementwise nodes as one kernel, with the time loop in the
// kernel and the carries in registers
struct MegakernelPlan {
  std::vector<const array*> nodes;
  // Trace-side input indices bound as kernel buffers, in signature order
  std::vector<int> buffer_inputs;
  size_t elems;
};

std::atomic<size_t> single_kernel_dispatches{0};

bool is_elementwise_node(const array& a) {
  auto& p = a.primitive();
  return a.siblings().empty() && (is_unary(p) || is_binary(p) || is_ternary(p));
}

std::optional<MegakernelPlan> plan_scan_megakernel(
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& tape,
    int num_consts,
    int num_carry) {
  std::unordered_map<std::uintptr_t, int> leaf_index;
  for (int i = 0; i < static_cast<int>(trace_inputs.size()); ++i) {
    leaf_index.emplace(trace_inputs[i].id(), i);
  }

  MegakernelPlan plan;
  for (auto& entry : tape) {
    if (leaf_index.find(entry.id()) == leaf_index.end()) {
      plan.nodes.push_back(&entry);
    }
  }
  if (plan.nodes.empty()) {
    return std::nullopt;
  }

  auto elem_shape = plan.nodes.front()->shape();
  std::vector<bool> used(trace_inputs.size(), false);
  std::unordered_set<std::uintptr_t> produced;
  // A node reads scalar or element-shaped leaves, or earlier node outputs
  auto readable = [&](const array& in) {
    if (auto it = leaf_index.find(in.id()); it != leaf_index.end()) {
      used[it->second] = true;
      return is_scalar(in) || in.shape() == elem_shape;
    }
    return produced.find(in.id()) != produced.end();
  };

  for (auto* node : plan.nodes) {
    auto& primitive = node->primitive();
    if (typeid(primitive) == typeid(Compiled)) {
      auto& body = static_cast<const Compiled&>(primitive);
      if (!body.constant_ids().empty()) {
        return std::nullopt;
      }
      for (auto& in : body.fused_inputs()) {
        if (!readable(in)) {
          return std::nullopt;
        }
      }
    } else if (is_elementwise_node(*node)) {
      for (auto& in : node->inputs()) {
        if (!readable(in)) {
          return std::nullopt;
        }
      }
    } else {
      return std::nullopt;
    }
    for (auto& o : node->outputs()) {
      if (o.shape() != elem_shape) {
        return std::nullopt;
      }
      produced.insert(o.id());
    }
  }

  // Each output must be a node output or a leaf
  for (int k = 0; k < num_carry; ++k) {
    if (trace_inputs[num_consts + k].shape() != elem_shape) {
      return std::nullopt;
    }
    used[num_consts + k] = true;
  }
  for (auto& o : trace_outputs) {
    if (o.shape() != elem_shape) {
      return std::nullopt;
    }
    if (auto it = leaf_index.find(o.id()); it != leaf_index.end()) {
      used[it->second] = true;
    } else if (produced.find(o.id()) == produced.end()) {
      return std::nullopt;
    }
  }

  plan.elems = 1;
  for (auto dim : elem_shape) {
    plan.elems *= dim;
  }
  if (plan.elems == 0 || plan.elems > UINT32_MAX) {
    return std::nullopt;
  }
  for (int i = 0; i < static_cast<int>(trace_inputs.size()); ++i) {
    if (used[i]) {
      plan.buffer_inputs.push_back(i);
    }
  }
  return plan;
}

std::string build_scan_megakernel(
    const std::string& kernel_name,
    const MegakernelPlan& plan,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    int num_consts,
    int num_carry) {
  NodeNamer namer;
  std::string os;
  int cnt = 0;
  int num_loop_invariant = num_consts + num_carry;

  os += fmt::format(
      "[[host_name(\"{0}\")]]\n[[kernel]] void {0}(\n", kernel_name);
  for (int i : plan.buffer_inputs) {
    auto& leaf = trace_inputs[i];
    os += fmt::format(
        "    device const {0}* {1} [[buffer({2})]],\n",
        get_type_string(leaf.dtype()),
        namer.get_name(leaf),
        cnt++);
  }
  for (size_t k = 0; k < trace_outputs.size(); ++k) {
    os += fmt::format(
        "    device {0}* sout_{1} [[buffer({2})]],\n",
        get_type_string(trace_outputs[k].dtype()),
        k,
        cnt++);
  }
  os += fmt::format("    constant const uint& length [[buffer({})]],\n", cnt++);
  os += fmt::format("    constant const uint& n [[buffer({})]],\n", cnt++);
  os += "    uint index [[thread_position_in_grid]]) {\n";

  // Read the consts and the initial carries once
  for (int i : plan.buffer_inputs) {
    if (i >= num_loop_invariant) {
      continue;
    }
    auto& leaf = trace_inputs[i];
    os += fmt::format(
        "  {0} tmp_{1} = {1}[{2}];\n",
        get_type_string(leaf.dtype()),
        namer.get_name(leaf),
        is_scalar(leaf) ? "0" : "index");
  }

  os += "  for (uint t = 0; t < length; ++t) {\n";
  for (int i : plan.buffer_inputs) {
    if (i < num_loop_invariant) {
      continue;
    }
    auto& leaf = trace_inputs[i];
    os += fmt::format(
        "    {0} tmp_{1} = {1}[{2}];\n",
        get_type_string(leaf.dtype()),
        namer.get_name(leaf),
        is_scalar(leaf) ? "t" : "ulong(t) * n + index");
  }

  for (auto* node : plan.nodes) {
    auto& primitive = node->primitive();
    if (typeid(primitive) != typeid(Compiled)) {
      emit_fused_ops(os, {*node}, namer, "    ");
      continue;
    }
    // Alias the node outputs to the inner outputs of its fused tape
    auto& body = static_cast<const Compiled&>(primitive);
    emit_fused_ops(os, body.fused_tape(), namer, "    ");
    auto outer = node->outputs();
    for (size_t q = 0; q < outer.size(); ++q) {
      os += fmt::format(
          "    {0} tmp_{1} = tmp_{2};\n",
          get_type_string(outer[q].dtype()),
          namer.get_name(outer[q]),
          namer.get_name(body.fused_outputs()[q]));
    }
  }

  for (size_t k = num_carry; k < trace_outputs.size(); ++k) {
    os += fmt::format(
        "    sout_{0}[ulong(t) * n + index] = tmp_{1};\n",
        k,
        namer.get_name(trace_outputs[k]));
  }
  // Read all new carries before rebinding, because a body can permute them
  for (int k = 0; k < num_carry; ++k) {
    os += fmt::format(
        "    {0} next_{1} = tmp_{2};\n",
        get_type_string(trace_outputs[k].dtype()),
        k,
        namer.get_name(trace_outputs[k]));
  }
  for (int k = 0; k < num_carry; ++k) {
    os += fmt::format(
        "    tmp_{0} = next_{1};\n",
        namer.get_name(trace_inputs[num_consts + k]),
        k);
  }
  os += "  }\n";

  for (int k = 0; k < num_carry; ++k) {
    os += fmt::format(
        "  sout_{0}[index] = tmp_{1};\n",
        k,
        namer.get_name(trace_inputs[num_consts + k]));
  }
  os += "}\n";
  return os;
}

} // namespace

size_t scan_single_kernel_dispatches() {
  return single_kernel_dispatches.load();
}

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

  if (auto plan = plan_scan_megakernel(
          inputs_, outputs_, tape_, num_consts_, num_carry_)) {
    const std::string token = "SCAN_SINGLE_KERNEL";
    auto source = build_scan_megakernel(
        token, *plan, inputs_, outputs_, num_consts_, num_carry_);
    auto kernel_name =
        "scan_loop_" + std::to_string(std::hash<std::string>{}(source));
    for (auto pos = source.find(token); pos != std::string::npos;
         pos = source.find(token, pos)) {
      source.replace(pos, token.size(), kernel_name);
    }

    // Copy before the pipeline state is set, because each copy sets its own
    std::vector<array> bound;
    std::vector<array> copies;
    for (int i : plan->buffer_inputs) {
      auto& runtime = inputs[i];
      if (runtime.ndim() > 0 && !runtime.flags().row_contiguous) {
        copies.push_back(contiguous_copy_gpu(runtime, s));
        bound.push_back(copies.back());
      } else {
        bound.push_back(runtime);
      }
    }

    auto& d = metal::device(s.device);
    auto lib = d.get_library(kernel_name, [&]() {
      std::string kernel = metal::utils();
      concatenate(
          kernel,
          metal::unary_ops(),
          metal::binary_ops(),
          metal::ternary_ops());
      kernel += source;
      return kernel;
    });
    auto kernel = d.get_kernel(kernel_name, lib);
    encoder.set_compute_pipeline_state(kernel);

    int cnt = 0;
    for (auto& in : bound) {
      encoder.set_input_array(in, cnt++);
    }
    for (auto& out : outputs) {
      encoder.set_output_array(out, cnt++);
    }
    uint32_t length = length_;
    uint32_t elems = plan->elems;
    encoder.set_bytes(length, cnt++);
    encoder.set_bytes(elems, cnt++);
    size_t group =
        std::min<size_t>(plan->elems, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(
        MTL::Size(plan->elems, 1, 1), MTL::Size(group, 1, 1));
    encoder.add_temporaries(std::move(copies));
    single_kernel_dispatches++;
    return;
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
