// Copyright © 2026 Apple Inc.

#pragma once

#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

// A body node of a scan, with its bindings resolved to slots
struct ScanStep {
  const array* node;
  std::vector<int> input_slots;
  std::vector<int> output_slots;
  // Positions of the inputs released before the node runs, so that the node
  // can donate their buffers
  std::vector<int> released_inputs;
  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
};

// The bindings of a scan body replay, resolved once before the time loop
struct ScanPlan {
  int num_slots;
  // Slots of the trace inputs: consts, then carries, then xs
  std::vector<int> input_slots;
  std::vector<int> output_slots;
  std::vector<ScanStep> steps;
  // Released at the end of each iteration
  std::vector<int> node_output_slots;
};

ScanPlan make_scan_plan(
    const std::vector<array>& trace_inputs,
    const std::vector<array>& tape,
    const std::vector<array>& trace_outputs,
    int num_consts);

// Shape/strides for a single row so the copy kernels never see a rank-0 view
inline std::pair<Shape, Strides> row_layout(const array& row_like) {
  if (row_like.ndim() == 0) {
    return {Shape{1}, Strides{1}};
  }
  return {row_like.shape(), make_contiguous_strides(row_like.shape())};
}

} // namespace mlx::core
