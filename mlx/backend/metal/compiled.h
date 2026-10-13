// Copyright © 2026 Apple Inc.

#pragma once

#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/graph_utils.h"

namespace mlx::core {

// Emit the statements computing each node of a fused tape into tmp_ locals
void emit_fused_ops(
    std::string& os,
    const std::vector<array>& tape,
    NodeNamer& namer,
    const std::string& indent);

} // namespace mlx::core
