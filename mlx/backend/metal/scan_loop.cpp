// Copyright © 2026 Apple Inc.

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "mlx/allocator.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/scan_loop.h"
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

// Run a chain of elementwise nodes as one kernel, with the time loop in the
// kernel and the carries in registers
struct MegakernelPlan {
  std::vector<const array*> nodes;
  // Trace-side input indices bound as kernel buffers, in signature order
  std::vector<int> buffer_inputs;
  size_t elems;
};

std::atomic<size_t> single_kernel_dispatches{0};

// A fused section past the buffer limit cannot build its kernels
bool fits_kernel_buffers(const std::vector<array>& tape) {
  for (auto& node : tape) {
    if (!node.has_primitive()) {
      continue;
    }
    auto& primitive = node.primitive();
    if (typeid(primitive) == typeid(Compiled) &&
        compiled_kernel_buffers(static_cast<const Compiled&>(primitive)) >
            max_kernel_buffers) {
      return false;
    }
  }
  return true;
}

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
  std::unordered_set<std::uintptr_t> emitted;
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
      // Fusing an earlier section rewires the outer inputs only, so an inner
      // input can be that section's old output, which its tape emits
      auto& inner = body.fused_inputs();
      for (size_t j = 0; j < inner.size(); ++j) {
        auto& outer = node->inputs()[j];
        if (!readable(outer) ||
            (inner[j].id() != outer.id() &&
             emitted.find(inner[j].id()) == emitted.end())) {
          return std::nullopt;
        }
      }
      for (auto& o : body.fused_outputs()) {
        emitted.insert(o.id());
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
  // The length and size take two buffer slots
  if (plan.buffer_inputs.size() + trace_outputs.size() + 2 >
      max_kernel_buffers) {
    return std::nullopt;
  }
  return plan;
}

std::string build_scan_megakernel(
    const std::string& kernel_name,
    const MegakernelPlan& plan,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    int num_consts,
    int num_carry,
    bool reverse) {
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
  os += reverse ? "    uint row = length - 1 - t;\n" : "    uint row = t;\n";
  for (int i : plan.buffer_inputs) {
    if (i < num_loop_invariant) {
      continue;
    }
    auto& leaf = trace_inputs[i];
    os += fmt::format(
        "    {0} tmp_{1} = {1}[{2}];\n",
        get_type_string(leaf.dtype()),
        namer.get_name(leaf),
        is_scalar(leaf) ? "row" : "ulong(row) * n + index");
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
        "    sout_{0}[ulong(row) * n + index] = tmp_{1};\n",
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
  auto& s = stream();
  auto& compute_encoder = metal::get_command_encoder(s);

  // Metal does not retain buffers, so keep them until the command buffer
  // completes
  auto retain = [&](std::vector<array> keep) {
    compute_encoder.get_command_buffer()->addCompletedHandler(
        [keep = std::move(keep)](MTL::CommandBuffer*) {});
  };

  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  // Replay the unfused body when the fused one does not fit the kernels
  auto& fused = fused_body();
  std::call_once(
      gpu_fits_flag_, [&]() { gpu_fits_ = fits_kernel_buffers(fused.tape); });
  auto& body_inputs = gpu_fits_ ? fused.inputs : inputs_;
  auto& body_tape = gpu_fits_ ? fused.tape : tape_;
  auto& body_outputs = gpu_fits_ ? fused.outputs : outputs_;
  if (auto plan = plan_scan_megakernel(
          body_inputs, body_outputs, body_tape, num_consts_, num_carry_)) {
    const std::string token = "SCAN_SINGLE_KERNEL";
    auto source = build_scan_megakernel(
        token,
        *plan,
        body_inputs,
        body_outputs,
        num_consts_,
        num_carry_,
        reverse_);
    auto kernel_name =
        fmt::format("scan_loop_{:x}", std::hash<std::string>{}(source));
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
    compute_encoder.set_compute_pipeline_state(kernel);

    int cnt = 0;
    for (auto& in : bound) {
      compute_encoder.set_input_array(in, cnt++);
    }
    for (auto& out : outputs) {
      compute_encoder.set_output_array(out, cnt++);
    }
    uint32_t length = length_;
    uint32_t elems = plan->elems;
    compute_encoder.set_bytes(length, cnt++);
    compute_encoder.set_bytes(elems, cnt++);
    size_t group =
        std::min<size_t>(plan->elems, kernel->maxTotalThreadsPerThreadgroup());
    compute_encoder.dispatch_threads(
        MTL::Size(plan->elems, 1, 1), MTL::Size(group, 1, 1));
    compute_encoder.add_temporaries(std::move(copies));
    single_kernel_dispatches++;
    return;
  }

  auto plan = make_scan_plan(body_inputs, body_tape, body_outputs, num_consts_);
  std::vector<std::optional<array>> env(plan.num_slots);
  for (int i = 0; i < num_consts_ + num_carry_; ++i) {
    env[plan.input_slots[i]] = inputs[i];
  }

  // Keep replaced and released buffers until the command buffer completes
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
          // The general copy kernel has no rank-0 variant
          auto ctype = x_row_strides == row_strides ? CopyType::Vector
                                                    : CopyType::GeneralGeneral;
          copy_gpu_inplace(
              /* in = */ x,
              /* out = */ row,
              /* data_shape = */ data_shape,
              /* i_strides = */ x_row_strides,
              /* o_strides = */ row_strides,
              /* i_offset = */ row_index * x.strides(0),
              /* o_offset = */ 0,
              ctype,
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
          step.node->primitive().eval_gpu(node_inputs, node_outputs);
        } catch (...) {
          // Keep the released buffers alive for the recorded kernels
          retain(std::move(node_inputs));
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

      // Copy into a row view, because the vector kernel sizes its dispatch
      // from the destination
      for (int j = 0; j < num_ys; ++j) {
        auto& y = env[plan.output_slots[num_carry_ + j]].value();
        auto& stacked = outputs[num_carry_ + j];
        auto [data_shape, row_strides] = row_layout(y);
        array out_row(data_shape, y.dtype(), nullptr, {});
        out_row.copy_shared_buffer(
            stacked,
            row_strides,
            stacked.flags(),
            y.size(),
            row_index * static_cast<int64_t>(y.size()));
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

      if (!retired.empty()) {
        retain(std::move(retired));
        retired.clear();
      }

      // Commit during the scan, so the GPU starts before encoding ends
      if (compute_encoder.needs_commit()) {
        compute_encoder.end_encoding();
        scheduler::notify_new_task(s);
        compute_encoder.commit([s]() { scheduler::notify_task_completion(s); });
      }
    }

    for (int i = 0; i < num_carry_; ++i) {
      auto& carry = env[plan.input_slots[num_consts_ + i]].value();
      auto ctype = carry.flags().row_contiguous ? CopyType::Vector
                                                : CopyType::GeneralGeneral;
      copy_gpu_inplace(carry, outputs[i], ctype, s);
    }
  } catch (...) {
    // Keep the live bindings for the recorded kernels, then rethrow
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
    retain(std::move(keep));
    throw;
  }

  // Keep the live bindings until the command buffer completes
  std::vector<array> remaining;
  for (auto& value : env) {
    if (value) {
      remaining.push_back(std::move(*value));
    }
  }
  retain(std::move(remaining));
}

} // namespace mlx::core
