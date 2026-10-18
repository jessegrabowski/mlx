# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx_tests
from mlx.utils import tree_flatten, tree_map


def unrolled(fun, init, xs, reverse=False):
    length = tree_flatten(xs)[0][1].shape[0]
    steps = range(length - 1, -1, -1) if reverse else range(length)
    carry, ys = init, [None] * length
    for t in steps:
        carry, ys[t] = fun(carry, tree_map(lambda a: a[t], xs))
    if ys[0] is None:
        return carry, None
    return carry, tree_map(lambda *rows: mx.stack(rows), *ys)


def leaves(tree):
    return [v for _, v in tree_flatten(tree)]


class TestScan(mlx_tests.MLXTestCase):
    def assert_jvp_matches_unrolled(self, model, primals):
        # All primals at once, then each alone so the others get no tangent
        tangents = [
            mx.random.normal(p.shape, key=mx.random.key(i)).astype(p.dtype)
            for i, p in enumerate(primals)
        ]
        cases = [(list(range(len(primals))), tangents)]
        cases += [([i], [tangents[i]]) for i in range(len(primals))]
        for positions, case_tangents in cases:

            def fn(scan_fn, positions=positions):
                def f(*values):
                    full = list(primals)
                    for i, v in zip(positions, values):
                        full[i] = v
                    return leaves(model(scan_fn, *full))

                return f

            case_primals = [primals[i] for i in positions]
            _, expected = mx.jvp(fn(unrolled), case_primals, case_tangents)
            _, actual = mx.jvp(fn(mx.scan), case_primals, case_tangents)
            for a, e in zip(actual, expected):
                self.assertEqual(a.dtype, e.dtype)
                self.assertTrue(
                    mx.allclose(a, e, atol=1e-5), f"tangents of {positions}"
                )

    def assert_vjp_matches_unrolled(self, model, primals):
        # All primals at once, then each alone so the others get no cotangent
        def fn(scan_fn, positions):
            def f(*values):
                full = list(primals)
                for i, v in zip(positions, values):
                    full[i] = v
                return leaves(model(scan_fn, *full))

            return f

        outputs = leaves(model(unrolled, *primals))
        cotangents = [
            mx.random.normal(o.shape, key=mx.random.key(20 + i)).astype(o.dtype)
            for i, o in enumerate(outputs)
        ]
        cases = [list(range(len(primals)))] + [[i] for i in range(len(primals))]
        for positions in cases:
            case_primals = [primals[i] for i in positions]
            _, expected = mx.vjp(fn(unrolled, positions), case_primals, cotangents)
            _, actual = mx.vjp(fn(mx.scan, positions), case_primals, cotangents)
            for a, e in zip(actual, expected):
                self.assertEqual((a.shape, a.dtype), (e.shape, e.dtype))
                self.assertTrue(
                    mx.allclose(a, e, atol=1e-5), f"cotangents of {positions}"
                )

    def test_docstring_example(self):
        def step(total, x):
            total = total + x
            return total, total

        total, sums = mx.scan(step, mx.array(0.0), mx.arange(4.0))
        self.assertEqual(total.item(), 6.0)
        self.assertEqual(sums.tolist(), [0.0, 1.0, 3.0, 6.0])

    def test_reverse(self):
        W = 0.5 * mx.random.normal((3, 3), key=mx.random.key(23))
        xs = mx.random.normal((5, 3), key=mx.random.key(24))
        bodies = {
            "elementwise": lambda c, x: (0.9 * c + mx.sin(x), c * x),
            "matmul": lambda c, x: (mx.tanh(c @ W + x), {"c": c, "x": x}),
        }
        for name, body in bodies.items():
            expected = leaves(unrolled(body, mx.ones(3), xs, reverse=True))
            actual = leaves(mx.scan(body, mx.ones(3), xs, reverse=True))
            self.assertEqual(len(actual), len(expected), name)
            for a, e in zip(actual, expected):
                self.assertEqual(a.shape, e.shape, name)
                self.assertTrue(mx.allclose(a, e, atol=1e-6), name)

        carry, ys = mx.scan(
            lambda c, x: (c * 0.5 + x, None), mx.ones(3), xs, reverse=True
        )
        expected, _ = unrolled(
            lambda c, x: (c * 0.5 + x, None), mx.ones(3), xs, reverse=True
        )
        self.assertIsNone(ys)
        self.assertTrue(mx.allclose(carry, expected, atol=1e-6))

        # The last step reads the first row, and each y keeps the row of its x
        total, ys = mx.scan(
            lambda c, x: (c + x, c), mx.array(0.0), mx.arange(4.0), reverse=True
        )
        self.assertEqual(total.item(), 6.0)
        self.assertEqual(ys.tolist(), [6.0, 5.0, 3.0, 0.0])

    def test_rnn_cell_with_captured_weights(self):
        W = 0.3 * mx.random.normal((8, 8), key=mx.random.key(0))
        U = 0.3 * mx.random.normal((4, 8), key=mx.random.key(1))
        xs = mx.random.normal((12, 2, 4), key=mx.random.key(2))

        def cell(h, x):
            h = mx.tanh(h @ W + x @ U)
            return h, h

        h0 = mx.zeros((2, 8))
        carry, ys = mx.scan(cell, h0, xs)

        h = h0
        expected = []
        for t in range(12):
            h, _ = cell(h, xs[t])
            expected.append(h)
        self.assertEqual(ys.shape, (12, 2, 8))
        self.assertTrue(mx.allclose(ys, mx.stack(expected), atol=1e-6))
        self.assertTrue(mx.allclose(carry, h, atol=1e-6))

    def test_shapeless_compile(self):
        def fun(init, xs, W):
            return mx.scan(lambda c, x: (mx.tanh(c @ W + x), 2 * c), init, xs)

        compiled = mx.compile(fun, shapeless=True)
        keys = mx.random.split(mx.random.key(22), 3)
        W = 0.3 * mx.random.normal((3, 3), key=keys[0])
        for key in keys[1:]:
            args = (mx.ones(3), mx.random.normal((4, 3), key=key), W)
            for a, e in zip(leaves(compiled(*args)), leaves(fun(*args))):
                self.assertEqual(a.shape, e.shape)
                self.assertTrue(mx.allclose(a, e, atol=1e-6))

        # A new ndim retraces. A new length, carry, row, or const shape reuses
        # the trace, which has fixed shapes, so it must fail loudly
        changed = [
            (mx.ones(3), mx.ones((7, 3)), W),
            (mx.ones(5), mx.ones((4, 3)), W),
            (mx.ones(5), mx.ones((4, 5)), mx.ones((5, 5))),
            (mx.ones(3), mx.ones((4, 3)), mx.ones((3, 1))),
        ]
        for args in changed:
            with self.assertRaisesRegex(ValueError, "Shapeless compile"):
                compiled(*args)

    def test_shapeless_compile_of_vmap(self):
        # A shapeless compile cannot trace a lazy captured random array
        W = 0.3 * mx.random.normal((3, 3), key=mx.random.key(28))
        mx.eval(W)

        def fun(init, xs):
            return mx.scan(lambda c, x: (mx.tanh(c @ W + x), c), init, xs)

        batched = mx.vmap(fun)
        compiled = mx.compile(batched, shapeless=True)
        args = (mx.ones((2, 3)), mx.random.normal((2, 4, 3), key=mx.random.key(29)))
        for a, e in zip(leaves(compiled(*args)), leaves(batched(*args))):
            self.assertEqual(a.shape, e.shape)
            self.assertTrue(mx.allclose(a, e, atol=1e-6))

        # A new batch size changes the shape of the scan inputs
        with self.assertRaisesRegex(ValueError, "Shapeless compile"):
            compiled(mx.ones((3, 3)), mx.ones((3, 4, 3)))

    def test_shapeless_compile_of_grad(self):
        W = 0.3 * mx.random.normal((3, 3), key=mx.random.key(25))
        xs = mx.random.normal((4, 3), key=mx.random.key(26))
        for reverse in (False, True):

            def loss(init, xs, W):
                carry, ys = mx.scan(
                    lambda c, x: (mx.tanh(c @ W + x), c * x),
                    init,
                    xs,
                    reverse=reverse,
                )
                return carry.sum() + ys.sum()

            for transform in (mx.grad, mx.value_and_grad):
                f = transform(loss, argnums=(0, 1, 2))
                expected = leaves(f(mx.ones(3), xs, W))
                actual = leaves(mx.compile(f, shapeless=True)(mx.ones(3), xs, W))
                for a, e in zip(actual, expected):
                    self.assertEqual(a.shape, e.shape)
                    self.assertTrue(mx.allclose(a, e, atol=1e-5))

    def test_streams(self):
        devices = [mx.cpu]
        if mx.is_available(mx.gpu):
            devices.append(mx.gpu)
        xs = mx.arange(1.0, 7.0)
        for device in devices:
            carry, ys = mx.scan(
                lambda c, x: (c * x, c * x), mx.array(1.0), xs, stream=device
            )
            self.assertEqual(carry.item(), 720.0)
            self.assertTrue(mx.array_equal(ys, mx.cumprod(xs)))

    def test_stop_gradient(self):
        scale = mx.stop_gradient(2 * mx.ones(2))
        carry, _ = mx.scan(
            lambda c, x: (c * scale + x, None), mx.zeros(2), mx.ones((3, 2))
        )
        self.assertEqual(carry.tolist(), [7.0, 7.0])

        carry, _ = mx.scan(
            lambda c, x: (mx.stop_gradient(c) * 2 + x, None),
            mx.zeros(2),
            mx.ones((3, 2)),
        )
        self.assertEqual(carry.tolist(), [7.0, 7.0])

    def test_captured_array_as_output(self):
        W = 1.5 * mx.full((3,), 2.0)
        carry, ys = mx.scan(lambda c, x: (W, (c + x, W)), mx.ones(3), mx.ones((4, 3)))
        self.assertEqual(carry.tolist(), [3.0, 3.0, 3.0])
        self.assertEqual(ys[0][:, 0].tolist(), [2.0, 4.0, 4.0, 4.0])
        self.assertEqual(ys[1].tolist(), [[3.0] * 3] * 4)
        self.assertEqual(W.tolist(), [3.0, 3.0, 3.0])

    def test_carry_tuple_and_multiple_xs(self):
        def fib(c, x):
            a, b = c
            return (b, a + b), a * x

        (a, b), ys = mx.scan(fib, (mx.array(0), mx.array(1)), mx.ones(8, mx.int32))
        self.assertEqual((a.item(), b.item()), (21, 34))
        self.assertEqual(ys.tolist(), [0, 1, 1, 2, 3, 5, 8, 13])

        xs = (mx.arange(5.0), 2 * mx.arange(5.0))
        carry, ys = mx.scan(lambda c, x: (c + x[0] * x[1], x[1]), mx.array(0.0), xs)
        self.assertEqual(carry.item(), 60.0)
        self.assertTrue(mx.array_equal(ys, xs[1]))

    def test_no_ys(self):
        carry, ys = mx.scan(lambda c, x: (c + x, None), mx.array(0.0), mx.ones(4))
        self.assertEqual(carry.item(), 4.0)
        self.assertIsNone(ys)

    def test_tree_ys(self):
        def body(c, x):
            c = c + x
            return c, {"total": c, "pair": (x, -x)}

        xs = mx.arange(4.0)
        _, ys = mx.scan(body, mx.array(0.0), xs)
        self.assertTrue(mx.array_equal(ys["total"], mx.cumsum(xs)))
        self.assertTrue(mx.array_equal(ys["pair"][0], xs))
        self.assertTrue(mx.array_equal(ys["pair"][1], -xs))

    def test_errors(self):
        xs = mx.ones(3)
        with self.assertRaisesRegex(ValueError, "must return a tuple"):
            mx.scan(lambda c, x: c + x, mx.array(0.0), xs)
        with self.assertRaisesRegex(ValueError, "carry changed from"):
            mx.scan(lambda c, x: (mx.stack([c, c]), x), mx.array(0.0), xs)
        with self.assertRaisesRegex(ValueError, "mismatched lengths"):
            mx.scan(lambda c, x: (c, x), mx.array(0.0), (xs, mx.ones(4)))

    def test_nested_dict_carry(self):
        def body(c, x):
            # Return the keys in a different order than init
            h = c["rnn"]["h"] + x
            return {"count": c["count"] + 1, "rnn": {"h": h}}, h

        init = {"rnn": {"h": mx.zeros(2)}, "count": mx.array(0)}
        carry, ys = mx.scan(body, init, mx.ones((3, 2)))
        self.assertEqual(carry["count"].item(), 3)
        self.assertEqual(carry["rnn"]["h"].tolist(), [3.0, 3.0])
        self.assertEqual(ys.tolist(), [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    def test_structure_errors(self):
        xs = mx.ones((3, 2))
        init = (mx.zeros(2), mx.zeros(2))
        with self.assertRaisesRegex(ValueError, r"carry must be a tuple of length 2"):
            mx.scan(lambda c, x: ((c[0] + x,), c[1]), init, xs)
        with self.assertRaisesRegex(ValueError, r"as in init, but got list"):
            mx.scan(lambda c, x: ([c[0], c[1]], None), init, xs)
        with self.assertRaisesRegex(ValueError, r"carry\[1\] must be an array"):
            mx.scan(lambda c, x: ((c[0], 1.0), None), init, xs)
        with self.assertRaisesRegex(ValueError, r"must be a dict with keys \['h'\]"):
            mx.scan(lambda c, x: ({"g": c["h"]}, None), {"h": mx.zeros(2)}, xs)
        with self.assertRaisesRegex(ValueError, r"must be a dict with keys \['h'\]"):
            mx.scan(lambda c, x: ({**c, "g": c["h"]}, None), {"h": mx.zeros(2)}, xs)
        with self.assertRaisesRegex(
            ValueError, r"carry\['h'\] changed from float32 with shape \(2\)"
        ):
            mx.scan(lambda c, x: ({"h": mx.ones(3)}, None), {"h": mx.zeros(2)}, xs)
        with self.assertRaisesRegex(ValueError, r"carry changed from int32"):
            mx.scan(lambda c, x: (c + x, None), mx.zeros(2, mx.int32), xs)
        with self.assertRaisesRegex(ValueError, r"init\[1\] must be an array"):
            mx.scan(lambda c, x: (c, None), (mx.zeros(2), 0.0), xs)
        with self.assertRaisesRegex(ValueError, r"y must be an array"):
            mx.scan(lambda c, x: (c, 0.0), mx.zeros(2), xs)

    def test_jvp(self):
        h0 = mx.random.normal((3,), key=mx.random.key(0))
        xs = mx.random.normal((6, 3), key=mx.random.key(1))
        W = 0.5 * mx.random.normal((3, 3), key=mx.random.key(2))

        def scanned(h0, xs, W):
            def cell(h, x):
                h = mx.tanh(h @ W + x)
                return h, 2 * h

            return mx.scan(cell, h0, xs)

        def unrolled(h0, xs, W):
            h, ys = h0, []
            for x in xs:
                h = mx.tanh(h @ W + x)
                ys.append(2 * h)
            return h, mx.stack(ys)

        primals = [h0, xs, W]
        tangents = [mx.random.normal(p.shape, key=mx.random.key(3)) for p in primals]
        _, expected = mx.jvp(unrolled, primals, tangents)
        _, actual = mx.jvp(scanned, primals, tangents)
        for a, e in zip(actual, expected):
            self.assertTrue(mx.allclose(a, e, atol=1e-5))

        # The tangent of a captured const reaches the carry
        def final(W):
            return scanned(h0, xs, W)[0]

        _, (tangent,) = mx.jvp(final, [W], [mx.ones_like(W)])
        eps = 1e-2
        fd = (final(W + eps) - final(W - eps)) / (2 * eps)
        self.assertTrue(mx.allclose(tangent, fd, atol=1e-3))

    def test_jvp_stop_gradient(self):
        W = mx.array([0.5, -1.0, 2.0])

        def model(scan, init, xs, W):
            def body(c, x):
                c = mx.tanh(mx.stop_gradient(c) * W + x)
                return c, mx.stop_gradient(W) * c

            return scan(body, init, xs)

        primals = [mx.ones(3), mx.random.normal((5, 3), key=mx.random.key(9)), W]
        self.assert_jvp_matches_unrolled(model, primals)

    def test_jvp_partial_xs_and_unused_inputs(self):
        def model(scan, keep, acc, a, b, unused):
            def body(c, x):
                k, s = c
                return (k, s * 0.9 + x[0] * x[1]), mx.sin(x[0])

            return scan(body, (keep, acc), (a, b, unused))

        key = mx.random.key(10)
        primals = [mx.ones(2), mx.zeros(2)] + [
            mx.random.normal((4, 2), key=k) for k in mx.random.split(key, 3)
        ]
        self.assert_jvp_matches_unrolled(model, primals)

    def test_jvp_outputs_that_are_inputs(self):
        def model(scan, init, xs, W):
            return scan(lambda c, x: (x, (c, W, x * c)), init, xs)

        primals = [
            mx.ones(3),
            mx.random.normal((4, 3), key=mx.random.key(11)),
            2 * mx.ones(3),
        ]
        self.assert_jvp_matches_unrolled(model, primals)

    def test_jvp_tangent_dtype_differs_from_carry(self):
        xs = mx.random.normal((4, 3), key=mx.random.key(12))

        def scanned(h):
            return [mx.scan(lambda c, x: (mx.sin(c) + x, None), h, xs)[0]]

        def reference(h):
            return [unrolled(lambda c, x: (mx.sin(c) + x, None), h, xs)[0]]

        tangent = [mx.ones(3, dtype=mx.float16)]
        _, (expected,) = mx.jvp(reference, [mx.ones(3)], tangent)
        _, (actual,) = mx.jvp(scanned, [mx.ones(3)], tangent)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(mx.allclose(actual, expected, atol=1e-3))

    def test_vjp(self):
        def model(scan, h0, xs, W, b):
            def cell(c, x):
                n, h = c
                h = mx.tanh(h @ W + x + b)
                return (n + 1, h), h

            return scan(cell, (mx.array(0), h0), xs)

        keys = mx.random.split(mx.random.key(14), 4)
        primals = [
            mx.random.normal((3,), key=keys[0]),
            mx.random.normal((6, 3), key=keys[1]),
            0.5 * mx.random.normal((3, 3), key=keys[2]),
            mx.random.normal((3,), key=keys[3]),
        ]
        self.assert_vjp_matches_unrolled(model, primals)

        # The gradient of a captured const sums over the steps
        def loss(W):
            return model(mx.scan, primals[0], primals[1], W, primals[3])[0][1].sum()

        grad = mx.grad(loss)(primals[2])
        eps = 1e-3
        direction = mx.random.normal((3, 3), key=mx.random.key(15))
        fd = (
            loss(primals[2] + eps * direction) - loss(primals[2] - eps * direction)
        ) / (2 * eps)
        self.assertTrue(mx.allclose((grad * direction).sum(), fd, atol=1e-3))

    def test_vjp_stop_gradient(self):
        W = mx.array([0.5, -1.0, 2.0])

        def model(scan, init, xs, W):
            def body(c, x):
                c = mx.tanh(mx.stop_gradient(c) * W + x)
                return c, mx.stop_gradient(W) * c

            return scan(body, init, xs)

        primals = [mx.ones(3), mx.random.normal((5, 3), key=mx.random.key(16)), W]
        self.assert_vjp_matches_unrolled(model, primals)

    def test_stop_gradient_outputs(self):
        W = mx.array([0.5, -1.0, 2.0])
        stopped_W = mx.stop_gradient(2 * W)

        def model(scan, init, xs, W):
            def body(c, x):
                y = (mx.stop_gradient(c * x), stopped_W, mx.sum(c), c * x * W)
                return mx.stop_gradient(mx.sin(c) * W + x), y

            return scan(body, init, xs)

        def loss(scan_fn, *primals):
            return sum(mx.sum(mx.sin(a)) for a in leaves(model(scan_fn, *primals)))

        primals = [mx.ones(3), mx.random.normal((5, 3), key=mx.random.key(22)), W]
        argnums = tuple(range(len(primals)))
        expected = mx.grad(lambda *p: loss(unrolled, *p), argnums=argnums)(*primals)
        actual = mx.grad(lambda *p: loss(mx.scan, *p), argnums=argnums)(*primals)
        for a, e in zip(actual, expected):
            self.assertTrue(mx.allclose(a, e, atol=1e-5))
        self.assert_jvp_matches_unrolled(model, primals)

    def test_vjp_partial_xs_and_unused_inputs(self):
        def model(scan, keep, acc, a, b, unused):
            def body(c, x):
                k, s = c
                return (k, s * 0.9 + x[0] * x[1]), mx.sin(x[0])

            return scan(body, (keep, acc), (a, b, unused))

        key = mx.random.key(17)
        primals = [mx.ones(2), mx.zeros(2)] + [
            mx.random.normal((4, 2), key=k) for k in mx.random.split(key, 3)
        ]
        self.assert_vjp_matches_unrolled(model, primals)

    def test_vjp_outputs_that_are_inputs(self):
        def model(scan, init, xs, W):
            return scan(lambda c, x: (x, (c, W, x * c)), init, xs)

        primals = [
            mx.ones(3),
            mx.random.normal((4, 3), key=mx.random.key(18)),
            2 * mx.ones(3),
        ]
        self.assert_vjp_matches_unrolled(model, primals)

    def test_vjp_cotangent_dtype_differs_from_carry(self):
        @mx.custom_function
        def wide_sin(x):
            return mx.sin(x)

        @wide_sin.vjp
        def wide_sin_vjp(primals, cotangent, output):
            return cotangent.astype(mx.float32) * mx.cos(primals)

        xs = mx.random.normal((4, 3), key=mx.random.key(21)).astype(mx.float16)

        def loss(scan_fn, h):
            carry, _ = scan_fn(lambda c, x: (wide_sin(c) + x, None), h, xs)
            return carry.astype(mx.float32).sum()

        def loss_of_final_carry(scan_fn, h):
            carry, _ = scan_fn(lambda c, x: (c * x + x, None), h, xs)
            return wide_sin(carry).astype(mx.float32).sum()

        h = mx.ones(3, dtype=mx.float16)
        for f in (loss, loss_of_final_carry):
            expected = mx.grad(lambda h: f(unrolled, h))(h)
            actual = mx.grad(lambda h: f(mx.scan, h))(h)
            self.assertEqual(actual.dtype, mx.float16)
            self.assertTrue(mx.allclose(actual, expected, atol=1e-2))

    def test_grad_of_grad(self):
        xs = mx.random.normal((5, 3), key=mx.random.key(19))

        def loss(scan_fn, w):
            carry, ys = scan_fn(
                lambda c, x: (mx.tanh(c * w + x), c * c), mx.ones(3), xs
            )
            return carry.sum() + ys.sum()

        def second(scan_fn):
            return mx.grad(lambda w: mx.grad(lambda v: loss(scan_fn, v))(w).sum())

        w = mx.array([0.3, -0.2, 0.8])
        self.assertTrue(mx.allclose(second(mx.scan)(w), second(unrolled)(w), atol=1e-5))

    def test_transforms_of_reverse(self):
        def model(scan, init, xs, W):
            def body(c, x):
                c = mx.tanh(c @ W + x)
                return c, (c * x, mx.sum(c))

            return scan(body, init, xs, reverse=True)

        keys = mx.random.split(mx.random.key(27), 3)
        primals = [
            mx.random.normal((3,), key=keys[0]),
            mx.random.normal((5, 3), key=keys[1]),
            0.5 * mx.random.normal((3, 3), key=keys[2]),
        ]
        self.assert_jvp_matches_unrolled(model, primals)
        self.assert_vjp_matches_unrolled(model, primals)

        batched_xs = mx.stack([primals[1], 2 * primals[1]])
        actual = leaves(
            mx.vmap(lambda xs: model(mx.scan, primals[0], xs, primals[2]))(batched_xs)
        )
        rows = [
            leaves(model(unrolled, primals[0], xs, primals[2])) for xs in batched_xs
        ]
        for a, e in zip(actual, (mx.stack(r) for r in zip(*rows))):
            self.assertEqual(a.shape, e.shape)
            self.assertTrue(mx.allclose(a, e, atol=1e-5))

    def test_wide_body(self):
        # Sections of this body need more buffers than a Metal kernel has
        def body(c, x):
            a, b, e = x
            new = [c[(i + 1) % 14] * 0.9 + mx.sin(c[i]) * a + b * e for i in range(14)]
            return new, (sum(new), new[3] * new[7])

        init = [mx.full((2,), 0.1 * (i + 1)) for i in range(14)]
        keys = mx.random.split(mx.random.key(13), 3)
        xs = tuple(mx.random.normal((5, 2), key=k) for k in keys)
        carry, ys = mx.scan(body, init, xs)
        expected_carry, expected_ys = unrolled(body, init, xs)
        for a, e in zip(leaves((carry, ys)), leaves((expected_carry, expected_ys))):
            self.assertTrue(mx.allclose(a, e, atol=1e-5))

    def test_vmap(self):
        def model(init, xs, W, bias):
            def body(c, x):
                n, h = c
                h = mx.tanh(h @ W + x + bias)
                return (n + 1, h), h * n.astype(h.dtype)

            return leaves(mx.scan(body, (mx.array(0), init), xs))

        keys = mx.random.split(mx.random.key(15), 4)
        init = mx.random.normal((5, 3), key=keys[0])
        xs = mx.random.normal((6, 5, 3), key=keys[1])
        W = 0.5 * mx.random.normal((5, 3, 3), key=keys[2])
        bias = mx.random.normal((5, 3), key=keys[3])
        # Batch each input alone, then all at an axis the scan must move
        configs = [
            ((init, xs[:, 0], W[0], bias[0]), (0, None, None, None)),
            ((init[0], mx.moveaxis(xs, 1, 0), W[0], bias[0]), (None, 0, None, None)),
            ((init[0], xs[:, 0], W, bias[0]), (None, None, 0, None)),
            ((init[0], xs[:, 0], W[0], bias), (None, None, None, 0)),
            ((init.T, mx.moveaxis(xs, 1, 2), W, bias), (1, 2, 0, 0)),
        ]
        for args, in_axes in configs:
            got = mx.vmap(model, in_axes=in_axes)(*args)
            rows = []
            for b in range(5):
                element = [
                    a if axis is None else mx.take(a, b, axis=axis)
                    for a, axis in zip(args, in_axes)
                ]
                rows.append(model(*element))
            expected = [mx.stack(r) for r in zip(*rows)]
            self.assertEqual(len(got), len(expected))
            for g, e in zip(got, expected):
                self.assertEqual((g.shape, g.dtype), (e.shape, e.dtype), f"{in_axes}")
                self.assertTrue(mx.allclose(g, e, atol=1e-5), f"{in_axes}")

    def test_body_exception_propagates(self):
        def body(c, x):
            raise RuntimeError("body failed")

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            mx.scan(body, mx.array(0.0), mx.ones(3))

        carry, _ = mx.scan(lambda c, x: (c + x, None), mx.array(0.0), mx.ones(3))
        self.assertEqual(carry.item(), 3.0)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
