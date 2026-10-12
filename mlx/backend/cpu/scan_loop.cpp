// Copyright © 2026 Apple Inc.

#include <unordered_map>
#include <unordered_set>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// Shape/strides for a single row so the copy kernels never see a rank-0 view
std::pair<Shape, Strides> row_layout(const array& row_like) {
  if (row_like.ndim() == 0) {
    return {Shape{1}, Strides{1}};
  }
  return {row_like.shape(), make_contiguous_strides(row_like.shape())};
}

} // namespace

void Scan::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto s = stream();
  auto& encoder = cpu::get_command_encoder(s);

  for (auto& out : outputs) {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  // Bind consts and the initial carries; x slices rebind every iteration.
  // The env holds a reference to everything a queued kernel may read, which
  // also keeps body ops from donating a buffer that is still needed.
  std::unordered_map<std::uintptr_t, array> env;
  std::unordered_set<std::uintptr_t> leaf_ids;
  for (auto& in : inputs_) {
    leaf_ids.insert(in.id());
  }
  for (int i = 0; i < num_consts_ + num_carry_; ++i) {
    env.insert_or_assign(inputs_[i].id(), inputs[i]);
  }

  int num_ys = outputs.size() - num_carry_;
  for (int t = 0; t < length_; ++t) {
    // Bind this iteration's row of each scanned input: a zero-copy view
    // into row-contiguous inputs, a strided copy otherwise
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
            t * x.strides(0));
      } else {
        row.set_data(allocator::malloc(row.nbytes()));
        auto [data_shape, row_strides] = row_layout(row);
        Strides x_row_strides(x.strides().begin() + 1, x.strides().end());
        if (x_row_strides.empty()) {
          x_row_strides = Strides{1};
        }
        copy_cpu_inplace(
            /* src = */ x,
            /* dst = */ row,
            /* data_shape = */ data_shape,
            /* i_strides = */ x_row_strides,
            /* o_strides = */ row_strides,
            /* i_offset = */ t * x.strides(0),
            /* o_offset = */ 0,
            CopyType::GeneralGeneral,
            s);
      }
      encoder.add_temporary(row);
      env.insert_or_assign(inputs_[j].id(), std::move(row));
    }

    // Replay the tape; leaves are already bound in the env
    for (auto& node : tape_) {
      if (leaf_ids.find(node.id()) != leaf_ids.end()) {
        continue;
      }
      std::vector<array> node_inputs;
      node_inputs.reserve(node.inputs().size());
      for (auto& in : node.inputs()) {
        node_inputs.push_back(env.at(in.id()));
      }

      auto trace_outs = node.outputs();
      std::vector<Shape> shapes;
      std::vector<Dtype> dtypes;
      for (auto& o : trace_outs) {
        shapes.push_back(o.shape());
        dtypes.push_back(o.dtype());
      }
      auto node_outputs = array::make_arrays(
          std::move(shapes), dtypes, node.primitive_ptr(), node_inputs);

      node.primitive().eval_cpu(node_inputs, node_outputs);
      for (size_t k = 0; k < node_outputs.size(); ++k) {
        encoder.add_temporary(node_outputs[k]);
        env.insert_or_assign(trace_outs[k].id(), std::move(node_outputs[k]));
      }
    }

    // Write this iteration's ys into the stacked outputs
    for (int j = 0; j < num_ys; ++j) {
      auto& y = env.at(outputs_[num_carry_ + j].id());
      auto [data_shape, out_strides] = row_layout(y);
      auto y_strides = y.ndim() == 0 ? Strides{1} : y.strides();
      copy_cpu_inplace(
          /* src = */ y,
          /* dst = */ outputs[num_carry_ + j],
          /* data_shape = */ data_shape,
          /* i_strides = */ y_strides,
          /* o_strides = */ out_strides,
          /* i_offset = */ 0,
          /* o_offset = */ t * static_cast<int64_t>(y.size()),
          CopyType::GeneralGeneral,
          s);
    }

    // Read every new carry before rebinding, so a body that permutes its
    // carries does not read an already-overwritten binding
    std::vector<array> new_carries;
    new_carries.reserve(num_carry_);
    for (int i = 0; i < num_carry_; ++i) {
      new_carries.push_back(env.at(outputs_[i].id()));
    }
    for (int i = 0; i < num_carry_; ++i) {
      env.insert_or_assign(inputs_[num_consts_ + i].id(), new_carries[i]);
    }
  }

  // The final carries are the loop's first outputs
  for (int i = 0; i < num_carry_; ++i) {
    auto& carry = env.at(inputs_[num_consts_ + i].id());
    copy_cpu_inplace(carry, outputs[i], CopyType::GeneralGeneral, s);
  }
}

} // namespace mlx::core
