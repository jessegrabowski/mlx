// Copyright © 2026 Apple Inc.

#pragma once

#include <functional>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx::core {

/**
 * Run `fun` once for each slice of `xs` along axis 0. Return the final carry
 * and the stacked outputs. The body maps (carry..., x...) to (carry..., y...)
 * and must not change the shape or dtype of a carry.
 */
MLX_API std::pair<std::vector<array>, std::vector<array>> scan(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& init,
    const std::vector<array>& xs,
    StreamOrDevice s = {});

} // namespace mlx::core
