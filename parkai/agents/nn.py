import numpy as np


def orthogonal(rng, rows, cols, gain=1.0):
    a = rng.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return gain * q[:rows, :cols]


class MLP:
    def __init__(self, sizes, rng, out_gain=1.0, hidden_gain=np.sqrt(2.0)):
        self.sizes = list(sizes)
        self.W, self.b = [], []
        n = len(sizes) - 1
        for i in range(n):
            gain = out_gain if i == n - 1 else hidden_gain
            self.W.append(orthogonal(rng, sizes[i], sizes[i + 1], gain))
            self.b.append(np.zeros(sizes[i + 1]))
        self._cache = None

    def params(self):
        return [p for pair in zip(self.W, self.b) for p in pair]

    def forward(self, x, keep=False):
        acts = [x]
        for i in range(len(self.W)):
            x = x @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                x = np.tanh(x)
            acts.append(x)
        if keep:
            self._cache = acts
        return x

    def activations(self, x):
        self.forward(x, keep=True)
        return self._cache

    def backward(self, dout):
        acts = self._cache
        grads = [None] * (2 * len(self.W))
        d = dout
        for i in reversed(range(len(self.W))):
            grads[2 * i] = acts[i].T @ d
            grads[2 * i + 1] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1.0 - acts[i] ** 2)
        return grads


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-5):
        self.params = params
        self.lr, self.betas, self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

    def state_dict(self):
        return dict(m=[x.copy() for x in self.m], v=[x.copy() for x in self.v], t=self.t)

    def load_state_dict(self, s):
        for dst, src in zip(self.m, s["m"]):
            dst[...] = src
        for dst, src in zip(self.v, s["v"]):
            dst[...] = src
        self.t = s["t"]


def clip_grad_norm(grads, max_norm):
    total = float(np.sqrt(sum(float((g * g).sum()) for g in grads)))
    if max_norm and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g *= scale
    return total
