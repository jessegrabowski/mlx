// Copyright © 2026 Apple Inc.

#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/common/scan_loop.h"

namespace mlx::core {

ScanPlan make_scan_plan(
    const std::vector<array>& trace_inputs,
    const std::vector<array>& tape,
    const std::vector<array>& trace_outputs,
    int num_consts) {
  ScanPlan plan;
  std::unordered_map<std::uintptr_t, int> slots;
  auto slot_of = [&slots](const array& a) {
    return slots.emplace(a.id(), static_cast<int>(slots.size())).first->second;
  };

  for (auto& in : trace_inputs) {
    plan.input_slots.push_back(slot_of(in));
  }

  // Never release the consts and the outputs during an iteration
  std::unordered_set<std::uintptr_t> kept_ids;
  for (int i = 0; i < num_consts; ++i) {
    kept_ids.insert(trace_inputs[i].id());
  }
  for (auto& o : trace_outputs) {
    kept_ids.insert(o.id());
  }

  // The tape lists the trace inputs too, so skip them
  std::vector<const array*> nodes;
  for (auto& entry : tape) {
    if (slots.find(entry.id()) == slots.end()) {
      nodes.push_back(&entry);
    }
  }
  std::unordered_map<std::uintptr_t, size_t> last_use;
  for (size_t idx = 0; idx < nodes.size(); ++idx) {
    for (auto& in : nodes[idx]->inputs()) {
      last_use[in.id()] = idx;
    }
  }

  for (size_t idx = 0; idx < nodes.size(); ++idx) {
    auto& node = *nodes[idx];
    ScanStep step{&node};
    auto& inputs = node.inputs();
    for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
      auto id = inputs[i].id();
      step.input_slots.push_back(slot_of(inputs[i]));
      if (kept_ids.find(id) == kept_ids.end() && last_use.at(id) == idx) {
        step.released_inputs.push_back(i);
      }
    }
    for (auto& o : node.outputs()) {
      step.output_slots.push_back(slot_of(o));
      step.shapes.push_back(o.shape());
      step.dtypes.push_back(o.dtype());
      plan.node_output_slots.push_back(step.output_slots.back());
    }
    plan.steps.push_back(std::move(step));
  }

  for (auto& o : trace_outputs) {
    plan.output_slots.push_back(slot_of(o));
  }
  plan.num_slots = slots.size();
  return plan;
}

} // namespace mlx::core
