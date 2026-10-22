import importlib.util

from .random_agent import RandomAgent

AGENTS = {"random": RandomAgent}

if importlib.util.find_spec(".ppo", __package__) is not None:
    from .ppo import PPOAgent
    AGENTS["ppo"] = PPOAgent

DEFAULT_AGENT = "ppo" if "ppo" in AGENTS else "random"


def make_agent(name, obs_dim, n_actions, **config):
    return AGENTS[name](obs_dim, n_actions, **config)
