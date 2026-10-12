// Copyright © 2026 Apple Inc.

#include <sstream>
#include <unordered_set>

#include "mlx/compile_impl.h"
#include "mlx/loop.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// Infer the loop length from the scanned inputs' shared leading axis
int loop_length(const std::vector<array>& xs) {
  if (xs.empty()) {
    throw std::invalid_argument(
        "[scan] At least one scanned input is required.");
  }
  for (auto& x : xs) {
    if (x.ndim() == 0) {
      throw std::invalid_argument(
          "[scan] Scanned inputs must have at least one dimension.");
    }
  }
  int length = xs[0].shape(0);
  for (auto& x : xs) {
    if (x.shape(0) != length) {
      std::ostringstream msg;
      msg << "[scan] Scanned inputs have mismatched lengths along axis 0: "
          << length << " and " << x.shape(0) << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  return length;
}

} // namespace

Scan::Scan(
    Stream stream,
    std::vector<array> inputs,
    std::vector<array> outputs,
    std::vector<array> tape,
    int num_consts,
    int num_carry,
    int length)
    : Primitive(stream),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      tape_(std::move(tape)),
      num_consts_(num_consts),
      num_carry_(num_carry),
      length_(length) {}

std::pair<std::vector<array>, std::vector<array>> scan(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& init,
    const std::vector<array>& xs,
    StreamOrDevice s /* = {} */) {
  int length = loop_length(xs);
  if (length == 0) {
    throw std::invalid_argument("[scan] Zero-length scan is not supported.");
  }

  auto input_specs = init;
  for (auto& x : xs) {
    Shape slice_shape(x.shape().begin() + 1, x.shape().end());
    input_specs.push_back(
        array(std::move(slice_shape), x.dtype(), nullptr, {}));
  }

  // Trace on the scan stream, so the replay runs the body there
  auto stream = to_stream(s);
  detail::LoopBodyTrace trace;
  {
    StreamContext body_stream(stream);
    trace = detail::compile_loop_body(fun, input_specs);
  }
  int num_carry = init.size();
  detail::compile_validate_loop_carry(trace, num_carry);
  if (trace.outputs.empty()) {
    throw std::invalid_argument(
        "[scan] The loop body must return at least one array.");
  }

  // The replay runs all body ops on the scan stream, so reject other streams
  std::unordered_set<std::uintptr_t> leaf_ids;
  for (auto& in : trace.inputs) {
    leaf_ids.insert(in.id());
  }
  for (auto& c : trace.consts) {
    leaf_ids.insert(c.id());
  }
  for (auto& node : trace.tape) {
    if (leaf_ids.find(node.id()) == leaf_ids.end() &&
        node.primitive().stream() != stream) {
      throw std::invalid_argument(
          "[scan] The loop body must run entirely on the scan's stream.");
    }
  }

  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  for (auto& c : init) {
    shapes.push_back(c.shape());
    dtypes.push_back(c.dtype());
  }
  for (size_t i = num_carry; i < trace.outputs.size(); ++i) {
    auto& y = trace.outputs[i];
    Shape stacked = y.shape();
    stacked.insert(stacked.begin(), length);
    shapes.push_back(std::move(stacked));
    dtypes.push_back(y.dtype());
  }

  // Trace-side and node inputs pair up positionally, consts first
  int num_consts = trace.consts.size();
  auto trace_inputs = trace.consts;
  trace_inputs.insert(
      trace_inputs.end(), trace.inputs.begin(), trace.inputs.end());
  auto node_inputs = std::move(trace.consts);
  node_inputs.insert(node_inputs.end(), init.begin(), init.end());
  node_inputs.insert(node_inputs.end(), xs.begin(), xs.end());

  auto node_outputs = array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<Scan>(
          stream,
          std::move(trace_inputs),
          std::move(trace.outputs),
          std::move(trace.tape),
          num_consts,
          num_carry,
          length),
      std::move(node_inputs));

  std::vector<array> final_carry(
      node_outputs.begin(), node_outputs.begin() + num_carry);
  std::vector<array> stacked_ys(
      node_outputs.begin() + num_carry, node_outputs.end());
  return {std::move(final_carry), std::move(stacked_ys)};
}

} // namespace mlx::core
