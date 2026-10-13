// Copyright © 2026 Apple Inc.

#pragma once

#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/graph_utils.h"

namespace mlx::core {

// Buffer argument slots available to a Metal kernel
constexpr int max_kernel_buffers = 31;

// Emit the statements computing each node of a fused tape into tmp_ locals
void emit_fused_ops(
    std::string& os,
    const std::vector<array>& tape,
    NodeNamer& namer,
    const std::string& indent);

// Number of scans dispatched as a single looped kernel since startup
size_t scan_single_kernel_dispatches();

} // namespace mlx::core
