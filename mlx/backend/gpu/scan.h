#pragma once

#include "mlx/array.h"
#include "mlx/primitives.h"

namespace mlx::core {

void scan_gpu_inplace(
    const array& in,
    array& out,
    CumulativeScan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    const Stream& s);

} // namespace mlx::core
