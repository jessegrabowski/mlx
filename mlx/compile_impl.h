// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/api.h"
#include "mlx/array.h"

namespace mlx::core {

// Elementwise primitive classes the compiler fuses
bool is_unary(const Primitive& p);
bool is_binary(const Primitive& p);
bool is_ternary(const Primitive& p);

} // namespace mlx::core

namespace mlx::core::detail {

using ArraysAndExtra = std::pair<std::vector<array>, std::shared_ptr<void>>;
using ArrayFnWithExtra =
    std::function<ArraysAndExtra(const std::vector<array>&)>;

// This is not part of the general C++ API as calling with a bad id is a bad
// idea.
MLX_API std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::uintptr_t fun_id,
    bool shapeless = false,
    std::vector<uint64_t> constants = {});

MLX_API ArrayFnWithExtra compile(
    ArrayFnWithExtra fun,
    std::uintptr_t fun_id,
    bool shapeless,
    std::vector<uint64_t> constants);

// Get the compiler cache of current thread.
class CompileCache;
using CompileCacheWeakPtr = std::weak_ptr<CompileCache>;
MLX_API CompileCacheWeakPtr compile_cache();

// Erase cached compile function.
MLX_API void compile_erase(
    const CompileCacheWeakPtr& cache,
    std::uintptr_t fun_id);

// Clear the compiler cache causing a recompilation of all compiled functions
// when called again.
MLX_API void compile_clear_cache(const CompileCacheWeakPtr& cache);

bool compile_available_for_device(const Device& device);

std::tuple<std::vector<array>, std::vector<array>, std::shared_ptr<void>>
compile_trace(
    const ArrayFnWithExtra& fun,
    const std::vector<array>& inputs,
    bool shapeless);

using ParentsMap =
    std::unordered_map<std::uintptr_t, std::vector<std::pair<array, int>>>;

// Traverses the graph to build a tape and a map of array ids to their parents
std::pair<std::vector<array>, ParentsMap> compile_dfs(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const std::vector<array>& original_inputs);

// Simplify the tape.
void compile_simplify(
    std::vector<array>& tape,
    ParentsMap& parents_map,
    std::vector<array>& outputs,
    int passes);

std::vector<array> compile_replace(
    const std::vector<array>& tape,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& inputs,
    bool shapeless);

struct LoopBodyTrace {
  std::vector<array> inputs;
  std::vector<array> consts;
  std::vector<array> tape;
  std::vector<array> outputs;
};

// Trace a loop body once on placeholder inputs shaped like `input_specs`,
// hoisting captured loop-invariant arrays into explicit `consts`
LoopBodyTrace compile_loop_body(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& input_specs);

// Check that the traced body returns one output per carry with an unchanged
// shape and dtype
void compile_validate_loop_carry(const LoopBodyTrace& trace, int num_carry);

void compile_validate_shapeless(const std::vector<array>& tape);

} // namespace mlx::core::detail
