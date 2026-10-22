import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from parkai.agents.nn import MLP, Adam, clip_grad_norm, orthogonal


def test_gradients():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 6, 3], rng)
    for b in net.b:
        b += rng.standard_normal(b.shape) * 0.1
    x = rng.standard_normal((4, 5))
    target = rng.standard_normal((4, 3))
    loss = lambda: 0.5 * ((net.forward(x) - target) ** 2).sum()
    out = net.forward(x, keep=True)
    grads = net.backward(out - target)
    worst = 0.0
    for p, g in zip(net.params(), grads):
        for idx in np.ndindex(*p.shape):
            old = p[idx]
            p[idx] = old + 1e-6
            lp = loss()
            p[idx] = old - 1e-6
            lm = loss()
            p[idx] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx])))
    assert worst < 1e-5, worst
    return worst


def test_orthogonal():
    rng = np.random.default_rng(1)
    q = orthogonal(rng, 8, 8)
    assert np.allclose(q @ q.T, np.eye(8), atol=1e-10)
    w = orthogonal(rng, 4, 9, gain=2.0)
    assert np.allclose(w @ w.T, 4 * np.eye(4), atol=1e-10)
    w = orthogonal(rng, 9, 4)
    assert np.allclose(w.T @ w, np.eye(4), atol=1e-10)


def test_clip():
    g = [np.full(4, 3.0), np.full(4, 4.0)]
    n = clip_grad_norm(g, 1.0)
    assert abs(n - np.sqrt(100.0)) < 1e-9
    assert abs(np.sqrt(sum((x * x).sum() for x in g)) - 1.0) < 1e-5


def test_adam_converges():
    rng = np.random.default_rng(2)
    net = MLP([2, 16, 1], rng)
    opt = Adam(net.params(), lr=1e-2)
    x = rng.standard_normal((64, 2))
    y = np.sin(x[:, :1]) + x[:, 1:] ** 2
    first = None
    for _ in range(600):
        out = net.forward(x, keep=True)
        l = ((out - y) ** 2).mean()
        first = first if first is not None else l
        opt.step(net.backward(2 * (out - y) / len(x)))
    assert l < 0.2 * first, (first, l)


if __name__ == "__main__":
    print("max relative grad error", test_gradients())
    test_orthogonal()
    test_clip()
    test_adam_converges()
    print("all nn tests passed")
