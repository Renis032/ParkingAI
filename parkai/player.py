import numpy as np

from .env import ParkingEnv
from .trainer import build_frame, finish_episode


class Player:
    def __init__(self, trainer, deterministic=True, difficulty=None, seed=None):
        self.trainer = trainer
        self.agent = trainer.agent
        self.deterministic = deterministic
        self.env = ParkingEnv(trainer.world, 1, trainer.env.sim, trainer.env.reward_cfg, seed=seed)
        self.env.difficulty = trainer.env.difficulty if difficulty is None else difficulty
        self.frames = []
        self.finished = None
        self.reset()

    def reset(self, pose=None):
        self.env.reset()
        if pose is not None:
            self.env.set_pose(0, *pose)
        self.frames = []
        self.finished = None

    def set_world(self, world):
        self.env.set_world(world)
        self.reset()

    @property
    def obs(self):
        return self.env.obs[0]

    def state(self):
        return self.env.state(0)

    def step(self):
        if self.finished is not None:
            return self.finished
        actions, ex = self.agent.act(self.env.obs, deterministic=self.deterministic)
        pre = self.env.state(0)
        obs0 = self.env.obs[0].copy()
        _, _, done, info = self.env.step(actions)
        self.frames.append(build_frame(pre, obs0, actions[0], ex, info, 0, self.frames))
        if done[0]:
            self.finished = finish_episode(self.frames, info, 0, self.env.world, self.trainer.world_version,
                                           self.trainer.updates)
        return self.finished

    def current_frame(self):
        if self.finished is not None:
            return self.frames[-1]
        actions, ex = self.agent.act(self.env.obs, deterministic=True)
        frame = dict(self.env.state(0))
        frame.update({"obs": self.env.obs[0].copy(), "action": int(actions[0]), "probs": ex["probs"][0],
                      "value": float(ex["value"][0]), "reward": 0.0,
                      "components": np.zeros(6), "collision": np.zeros(5, bool),
                      "return": self.frames[-1]["return"] if self.frames else 0.0})
        return frame

    def inspect(self):
        return self.agent.inspect(self.env.obs[0])
