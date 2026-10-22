import numpy as np

from .base import Agent
from .nn import MLP, Adam, clip_grad_norm


class RunningNorm:
    def __init__(self, dim, clip=10.0):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        n = len(x)
        bm, bv = x.mean(0), x.var(0)
        delta = bm - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * n / tot
        m2 = self.var * self.count + bv * n + delta ** 2 * self.count * n / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)

    def state_dict(self):
        return dict(mean=self.mean.copy(), var=self.var.copy(), count=self.count)

    def load_state_dict(self, s):
        self.mean, self.var, self.count = s["mean"].copy(), s["var"].copy(), s["count"]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, obs_dim, n_actions, hidden=(64, 64), lr=3e-4, clip=0.2, epochs=4, minibatch=256,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, seed=0, normalize_obs=True, target_kl=None,
                 **config):
        super().__init__(obs_dim, n_actions, hidden=tuple(hidden), lr=lr, clip=clip, epochs=epochs,
                         minibatch=minibatch, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                         seed=seed, normalize_obs=normalize_obs, target_kl=target_kl, **config)
        self.hidden = tuple(hidden)
        self.lr, self.clip, self.epochs, self.minibatch = lr, clip, epochs, minibatch
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.target_kl = target_kl
        self.rng = np.random.default_rng(seed)
        self.pi = MLP([obs_dim, *self.hidden, n_actions], self.rng, out_gain=0.01)
        self.vf = MLP([obs_dim, *self.hidden, 1], self.rng, out_gain=1.0)
        self.norm = RunningNorm(obs_dim) if normalize_obs else None
        self.opt = Adam(self.pi.params() + self.vf.params(), lr=lr)

    def _prep(self, obs):
        obs = np.asarray(obs, np.float64)
        return self.norm(obs) if self.norm is not None else obs

    def act(self, obs, deterministic=False):
        x = self._prep(obs)
        probs = softmax(self.pi.forward(x))
        value = self.vf.forward(x)[:, 0]
        if deterministic:
            actions = probs.argmax(-1)
        else:
            u = self.rng.random((len(x), 1))
            actions = np.minimum((probs.cumsum(-1) < u).sum(-1), self.n_actions - 1)
        logp = np.log(probs[np.arange(len(x)), actions] + 1e-12)
        return actions, dict(probs=probs, value=value, logp=logp)

    def value(self, obs):
        return self.vf.forward(self._prep(obs))[:, 0]

    def _gae(self, batch):
        rewards, values = batch["rewards"], batch["values"]
        notdone = 1.0 - batch["dones"].astype(np.float64)
        t_len = len(rewards)
        adv = np.zeros_like(rewards, dtype=np.float64)
        last = np.zeros(rewards.shape[1])
        next_v = np.asarray(batch["last_values"], np.float64)
        for t in reversed(range(t_len)):
            delta = rewards[t] + batch["gamma"] * next_v * notdone[t] - values[t]
            last = delta + batch["gamma"] * batch["lam"] * notdone[t] * last
            adv[t] = last
            next_v = values[t]
        return adv, adv + values

    def update(self, batch):
        adv, ret = self._gae(batch)
        obs = self._prep(batch["obs"].reshape(-1, self.obs_dim))
        acts = batch["actions"].reshape(-1)
        old_logp = batch["logp"].reshape(-1)
        old_v = batch["values"].reshape(-1)
        adv, ret = adv.reshape(-1), ret.reshape(-1)
        n = len(obs)
        var_ret = ret.var()
        explained = float(1.0 - np.var(ret - old_v) / var_ret) if var_ret > 1e-12 else 0.0
        stats = dict(pl=[], vl=[], ent=[], kl=[], cf=[])
        stop = False
        for _ in range(self.epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.minibatch):
                idx = perm[s:s + self.minibatch]
                m = len(idx)
                x, a = obs[idx], acts[idx]
                A = adv[idx]
                A = (A - A.mean()) / (A.std() + 1e-8)
                logits = self.pi.forward(x, keep=True)
                probs = softmax(logits)
                logp_all = np.log(probs + 1e-12)
                logp = logp_all[np.arange(m), a]
                ratio = np.exp(logp - old_logp[idx])
                clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)
                s1, s2 = ratio * A, clipped * A
                pl = -np.minimum(s1, s2).mean()
                active = (s1 <= s2).astype(np.float64)
                dlogp = -A * ratio * active / m
                onehot = np.zeros_like(probs)
                onehot[np.arange(m), a] = 1.0
                dlogits = dlogp[:, None] * (onehot - probs)
                ent_row = -(probs * logp_all).sum(-1)
                dlogits += self.ent_coef / m * probs * (logp_all + ent_row[:, None])
                g_pi = self.pi.backward(dlogits)

                v = self.vf.forward(x, keep=True)[:, 0]
                vl = 0.5 * ((v - ret[idx]) ** 2).mean()
                g_vf = self.vf.backward(((v - ret[idx]) * self.vf_coef / m)[:, None])

                grads = g_pi + g_vf
                clip_grad_norm(grads, self.max_grad_norm)
                self.opt.step(grads)

                kl = float(((ratio - 1) - np.log(ratio)).mean())
                stats["pl"].append(pl)
                stats["vl"].append(vl)
                stats["ent"].append(float(ent_row.mean()))
                stats["kl"].append(kl)
                stats["cf"].append(float((np.abs(ratio - 1) > self.clip).mean()))
                yield None
                if self.target_kl and kl > 1.5 * self.target_kl:
                    stop = True
                    break
            if stop:
                break
        if self.norm is not None:
            self.norm.update(batch["obs"])
        mean = lambda k: float(np.mean(stats[k]))
        yield dict(policy_loss=mean("pl"), value_loss=mean("vl"), entropy=mean("ent"), approx_kl=mean("kl"),
                   clip_frac=mean("cf"), explained_var=explained, lr=float(self.opt.lr))

    def inspect(self, obs_row):
        x = self._prep(np.asarray(obs_row, np.float64)[None])
        acts = self.pi.activations(x)
        layers = [dict(name="input", values=np.asarray(obs_row, float))]
        for i, a in enumerate(acts[1:-1]):
            layers.append(dict(name="hidden %d" % (i + 1), values=a[0].copy()))
        layers.append(dict(name="logits", values=acts[-1][0].copy()))
        return dict(layers=layers, probs=softmax(acts[-1])[0], value=float(self.vf.forward(x)[0, 0]))

    def state_dict(self):
        return dict(pi=[p.copy() for p in self.pi.params()], vf=[p.copy() for p in self.vf.params()],
                    opt=self.opt.state_dict(), norm=self.norm.state_dict() if self.norm else None,
                    rng=self.rng.bit_generator.state)

    def load_state_dict(self, s):
        for dst, src in zip(self.pi.params() + self.vf.params(), s["pi"] + s["vf"]):
            dst[...] = src
        self.opt.load_state_dict(s["opt"])
        if self.norm is not None and s.get("norm") is not None:
            self.norm.load_state_dict(s["norm"])
        if s.get("rng") is not None:
            self.rng.bit_generator.state = s["rng"]
