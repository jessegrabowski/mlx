// Copyright © 2026 Apple Inc.

#include "mlx/mlx.h"
#include "time_utils.h"

namespace mx = mlx::core;

void time_elementwise_scan() {
  int length = 2000;
  int width = 256;

  auto w = mx::multiply(mx::ones({width}), mx::array(0.9f));
  auto xs = mx::multiply(mx::ones({length, width}), mx::array(0.01f));
  auto init = mx::zeros({width});
  mx::eval(w, xs, init);

  auto body =
      [&w](const std::vector<mx::array>& args) -> std::vector<mx::array> {
    return {mx::tanh(mx::add(mx::multiply(args[0], w), args[1]))};
  };

  auto scan_gpu = [&]() {
    auto [carry, ys] = mx::scan(body, {init}, {xs});
    return carry[0];
  };
  TIMEM("elementwise gpu", scan_gpu);

  auto scan_cpu = [&]() {
    auto [carry, ys] = mx::scan(body, {init}, {xs}, mx::Device::cpu);
    return carry[0];
  };
  TIMEM("elementwise cpu", scan_cpu);

  auto scan_construct = [&]() {
    auto [carry, ys] = mx::scan(body, {init}, {xs});
    return mx::zeros({1});
  };
  TIMEM("elementwise construct only", scan_construct);

  auto unrolled = [&]() {
    auto c = init;
    for (int t = 0; t < length; ++t) {
      auto x = mx::reshape(mx::slice(xs, {t, 0}, {t + 1, width}), {width});
      c = mx::tanh(mx::add(mx::multiply(c, w), x));
    }
    return c;
  };
  TIMEM("elementwise unrolled raw", unrolled);

  auto unrolled_fn =
      [&](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
    auto c = ins[0];
    for (int t = 0; t < length; ++t) {
      auto x = mx::reshape(mx::slice(ins[1], {t, 0}, {t + 1, width}), {width});
      c = mx::tanh(mx::add(mx::multiply(c, w), x));
    }
    return {c};
  };
  auto compiled = mx::compile(unrolled_fn);
  auto compiled_run = [&]() { return compiled({init, xs})[0]; };
  TIMEM("elementwise unrolled compiled", compiled_run);
}

void time_matmul_scan() {
  int length = 2000;
  int width = 128;

  auto W = mx::multiply(mx::eye(width), mx::array(0.9f));
  auto xs = mx::multiply(mx::ones({length, width}), mx::array(0.01f));
  auto init = mx::zeros({width});
  mx::eval(W, xs, init);

  auto body =
      [&W](const std::vector<mx::array>& args) -> std::vector<mx::array> {
    return {mx::tanh(mx::add(mx::matmul(W, args[0]), args[1]))};
  };

  auto scan_gpu = [&]() {
    auto [carry, ys] = mx::scan(body, {init}, {xs});
    return carry[0];
  };
  TIMEM("matmul gpu", scan_gpu);

  auto scan_cpu = [&]() {
    auto [carry, ys] = mx::scan(body, {init}, {xs}, mx::Device::cpu);
    return carry[0];
  };
  TIMEM("matmul cpu", scan_cpu);

  auto unrolled_fn =
      [&](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
    auto c = ins[0];
    for (int t = 0; t < length; ++t) {
      auto x = mx::reshape(mx::slice(ins[1], {t, 0}, {t + 1, width}), {width});
      c = mx::tanh(mx::add(mx::matmul(W, c), x));
    }
    return {c};
  };
  auto compiled = mx::compile(unrolled_fn);
  auto compiled_run = [&]() { return compiled({init, xs})[0]; };
  TIMEM("matmul unrolled compiled", compiled_run);
}

int main() {
  time_elementwise_scan();
  time_matmul_scan();
}
