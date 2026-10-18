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
    auto [carry, ys] = mx::scan(body, {init}, {xs}, false, mx::Device::cpu);
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
    auto [carry, ys] = mx::scan(body, {init}, {xs}, false, mx::Device::cpu);
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

using Body = std::function<mx::array(const mx::array&, const mx::array&)>;

// The body as an unrolled loop, which mx::compile traces step by step
mx::array unrolled_carry(
    const Body& step,
    mx::array c,
    const mx::array& xs,
    bool reverse) {
  int length = xs.shape(0);
  int width = xs.shape(1);
  for (int i = 0; i < length; ++i) {
    int t = reverse ? length - 1 - i : i;
    auto x = mx::reshape(mx::slice(xs, {t, 0}, {t + 1, width}), {width});
    c = step(c, x);
  }
  return c;
}

mx::array scan_carry(
    const Body& step,
    const mx::array& init,
    const mx::array& xs,
    bool reverse,
    mx::StreamOrDevice s = {}) {
  auto body = [&step](const std::vector<mx::array>& a) {
    return std::vector<mx::array>{step(a[0], a[1])};
  };
  return mx::scan(body, {init}, {xs}, reverse, s).first[0];
}

// Time the forward scan, its gradient in both directions, and the gradient
// of the compiled unrolled loop. The weight is a differentiated input.
void time_grad(const std::string& name, const Body& step_of_weight_input) {
  int length = 2000;
  int width = 128;
  auto w = mx::multiply(mx::ones({width}), mx::array(0.9f));
  auto xs = mx::multiply(mx::ones({length, width}), mx::array(0.01f));
  auto init = mx::zeros({width});
  mx::eval(w, xs, init);
  std::vector<int> argnums = {0, 1, 2};

  for (auto device : {mx::Device::gpu, mx::Device::cpu}) {
    std::string where = device == mx::Device::gpu ? " gpu" : " cpu";
    mx::StreamContext on_device(device);
    for (bool reverse : {false, true}) {
      std::string label = name + where + (reverse ? " reverse" : "");
      auto loss = [&, reverse](const std::vector<mx::array>& in) {
        Body step = [&in, &step_of_weight_input](
                        const mx::array& c, const mx::array& x) {
          return step_of_weight_input(mx::multiply(c, in[2]), x);
        };
        return mx::sum(scan_carry(step, in[0], in[1], reverse));
      };
      auto forward = [&]() { return loss({init, xs, w}); };
      TIMEM(label + " forward", forward);

      auto grad_fn = mx::value_and_grad(loss, argnums);
      auto grad = [&]() { return grad_fn({init, xs, w}).second; };
      TIMEM(label + " grad", grad);
    }

    std::string label = name + where;
    auto unrolled_loss = [&](const std::vector<mx::array>& in) {
      Body step = [&in, &step_of_weight_input](
                      const mx::array& c, const mx::array& x) {
        return step_of_weight_input(mx::multiply(c, in[2]), x);
      };
      return std::vector<mx::array>{
          mx::sum(unrolled_carry(step, in[0], in[1], false))};
    };
    auto unrolled_grad = mx::compile([&](const std::vector<mx::array>& in) {
      auto loss = [&](const std::vector<mx::array>& a) {
        return unrolled_loss(a)[0];
      };
      return mx::value_and_grad(loss, argnums)(in).second;
    });
    auto compiled_grad = [&]() { return unrolled_grad({init, xs, w}); };
    TIMEM(label + " unrolled compiled grad", compiled_grad);
  }
}

// Time a batch of independent scans as one vmapped scan, against a vmap of
// the compiled unrolled loop
void time_vmap(const std::string& name, const Body& step) {
  int batch = 16;
  int length = 2000;
  int width = 128;
  auto xs = mx::multiply(mx::ones({batch, length, width}), mx::array(0.01f));
  auto init = mx::zeros({batch, width});
  mx::eval(xs, init);

  for (auto device : {mx::Device::gpu, mx::Device::cpu}) {
    std::string label = name + (device == mx::Device::gpu ? " gpu" : " cpu");
    mx::StreamContext on_device(device);
    auto scanned = mx::vmap(
        [&](const std::vector<mx::array>& in) {
          return std::vector<mx::array>{scan_carry(step, in[0], in[1], false)};
        },
        {0, 0},
        {0});
    auto run_scan = [&]() { return scanned({init, xs}); };
    TIMEM(label + " vmap scan", run_scan);

    auto unrolled = mx::compile(
        mx::vmap(
            [&](const std::vector<mx::array>& in) {
              return std::vector<mx::array>{
                  unrolled_carry(step, in[0], in[1], false)};
            },
            {0, 0},
            {0}));
    auto run_unrolled = [&]() { return unrolled({init, xs}); };
    TIMEM(label + " vmap unrolled compiled", run_unrolled);
  }
}

int main() {
  time_elementwise_scan();
  time_matmul_scan();

  auto W = mx::multiply(mx::eye(128), mx::array(0.9f));
  mx::eval(W);
  Body elementwise = [](const mx::array& c, const mx::array& x) {
    return mx::tanh(mx::add(c, x));
  };
  Body matmul = [&W](const mx::array& c, const mx::array& x) {
    return mx::tanh(mx::add(mx::matmul(W, c), x));
  };
  time_grad("elementwise", elementwise);
  time_grad("matmul", matmul);
  time_vmap("elementwise", elementwise);
  time_vmap("matmul", matmul);
}
