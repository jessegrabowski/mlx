// Copyright © 2026 Apple Inc.

#include "mlx/primitives.h"

namespace mlx::core {

void Scan::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[Scan::eval_cpu] NYI.");
}

} // namespace mlx::core
