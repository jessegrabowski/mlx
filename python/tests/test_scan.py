# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx_tests


class TestScan(mlx_tests.MLXTestCase):
    def test_cumsum(self):
        xs = mx.arange(10.0)
        carry, ys = mx.scan(lambda c, x: (c + x, c + x), mx.array(0.0), xs)
        self.assertTrue(mx.array_equal(ys, mx.cumsum(xs)))
        self.assertEqual(carry.item(), 45.0)

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

    def test_body_exception_propagates(self):
        def body(c, x):
            raise RuntimeError("body failed")

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            mx.scan(body, mx.array(0.0), mx.ones(3))

        carry, _ = mx.scan(lambda c, x: (c + x, None), mx.array(0.0), mx.ones(3))
        self.assertEqual(carry.item(), 3.0)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
