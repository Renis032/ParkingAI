import numpy as np

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, obs_dim, n_actions, seed=None, **config):
        super().__init__(obs_dim, n_actions, **config)
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=False):
        n = len(obs)
        probs = np.full((n, self.n_actions), 1.0 / self.n_actions)
        actions = self.rng.integers(self.n_actions, size=n)
        return actions, {"probs": probs, "value": np.zeros(n), "logp": np.log(probs[:, 0])}

    def value(self, obs):
        return np.zeros(len(obs))

    def update(self, batch):
        yield None
        yield {"policy_loss": 0.0, "value_loss": 0.0, "entropy": float(np.log(self.n_actions)),
               "approx_kl": 0.0, "clip_frac": 0.0, "explained_var": 0.0}
