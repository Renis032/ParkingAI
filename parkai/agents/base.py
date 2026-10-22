import numpy as np


class Agent:
    name = "base"

    def __init__(self, obs_dim, n_actions, **config):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = dict(config)

    def act(self, obs, deterministic=False):
        raise NotImplementedError

    def value(self, obs):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def inspect(self, obs_row):
        actions, ex = self.act(obs_row[None], deterministic=True)
        return {"layers": [{"name": "input", "values": np.asarray(obs_row, float)}],
                "probs": ex["probs"][0], "value": float(ex["value"][0])}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
