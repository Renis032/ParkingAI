import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np

from .agents import make_agent, DEFAULT_AGENT, AGENTS
from .env import ParkingEnv, SimConfig, RewardConfig
from .world import World


def build_frame(pre_state, obs_row, action, ex, info, i, frames):
    frame = dict(pre_state)
    frame.update({"obs": np.array(obs_row, copy=True), "action": int(action), "probs": ex["probs"][i].copy(),
                  "value": float(ex["value"][i]), "reward": float(info["components"][i].sum()),
                  "components": info["components"][i].copy(), "collision": info["collision"][i].copy()})
    frame["return"] = (frames[-1]["return"] if frames else 0.0) + frame["reward"]
    return frame


def finish_episode(frames, info, i, world, world_version, update):
    summary = next(e for e in info["episodes"] if e["env"] == i)
    prev = frames[-1]
    last = dict(info["terminal_states"][i])
    last.update({"obs": info["terminal_obs"][i].copy(), "action": -1, "probs": prev["probs"],
                 "value": 0.0, "reward": 0.0, "components": np.zeros_like(prev["components"]),
                 "collision": info["collision"][i].copy(), "return": prev["return"], "terminal": True})
    frames.append(last)
    return {"summary": summary, "frames": frames, "world": world, "world_version": world_version, "update": update}


@dataclass
class TrainConfig:
    agent: str = DEFAULT_AGENT
    num_envs: int = 16
    rollout_len: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    curriculum: bool = True
    start_difficulty: float = 0.0
    curriculum_window: int = 100
    curriculum_up: float = 0.7
    curriculum_step: float = 0.1
    record_keep: int = 40
    seed: int = 0


class Trainer:
    def __init__(self, world: World, train: TrainConfig = None, sim: SimConfig = None, reward: RewardConfig = None,
                 agent=None, agent_config=None):
        self.cfg = train or TrainConfig()
        self.world = world
        self.env = ParkingEnv(world, self.cfg.num_envs, sim, reward, seed=self.cfg.seed)
        self.env.difficulty = self.cfg.start_difficulty if self.cfg.curriculum else self.env.sim.difficulty
        self.agent = agent or make_agent(self.cfg.agent, self.env.obs_dim, self.env.n_actions, **(agent_config or {}))
        self.history = {"episodes": [], "updates": []}
        self.total_steps = 0
        self.updates = 0
        self.train_time = 0.0
        self.world_version = 0
        self.recent_episodes = deque(maxlen=self.cfg.record_keep)
        self.best_episode = None
        self.live = None
        self._frames = []
        self._new_buffer()
        self._update_gen = None
        self._sps = deque(maxlen=20)

    def _new_buffer(self):
        t, n, o = self.cfg.rollout_len, self.cfg.num_envs, self.env.obs_dim
        self._buf = {"obs": np.zeros((t, n, o), np.float32), "actions": np.zeros((t, n), np.int64),
                     "logp": np.zeros((t, n)), "values": np.zeros((t, n)), "rewards": np.zeros((t, n)),
                     "dones": np.zeros((t, n), bool)}
        self._pos = 0

    @property
    def updating(self):
        return self._update_gen is not None

    def run(self, seconds, max_steps=None):
        start = time.perf_counter()
        deadline = start + seconds
        steps0 = self.total_steps
        done_steps = 0
        while time.perf_counter() < deadline:
            if self._update_gen is not None:
                self._advance_update(deadline)
            elif max_steps is not None and done_steps >= max_steps:
                break
            else:
                self._env_step()
                done_steps += 1
        elapsed = time.perf_counter() - start
        self.train_time += elapsed
        if elapsed > 0:
            self._sps.append((self.total_steps - steps0) / elapsed)

    def steps_per_second(self):
        return float(np.mean(self._sps)) if self._sps else 0.0

    def _advance_update(self, deadline):
        for m in self._update_gen:
            if isinstance(m, dict):
                self._update_gen = None
                self._finish_update(m)
                return
            if time.perf_counter() >= deadline:
                return
        self._update_gen = None

    def _finish_update(self, metrics):
        self.updates += 1
        rec = dict(metrics)
        rec.update({"update": self.updates, "steps": self.total_steps, "difficulty": float(self.env.difficulty),
                    "success_rate": self.success_rate(), "sps": self.steps_per_second()})
        self.history["updates"].append(rec)

    def _env_step(self):
        env, b, p = self.env, self._buf, self._pos
        obs = env.obs
        actions, ex = self.agent.act(obs)
        pre = env.state(0)
        obs0 = obs[0].copy()
        new_obs, rewards, dones, info = env.step(actions)
        rewards = rewards.copy()
        trunc = info["truncated"]
        if trunc.any():
            rewards[trunc] += self.cfg.gamma * self.agent.value(info["terminal_obs"][trunc])
        b["obs"][p], b["actions"][p], b["logp"][p] = obs, actions, ex["logp"]
        b["values"][p], b["rewards"][p], b["dones"][p] = ex["value"], rewards, dones
        self._pos += 1
        self.total_steps += env.n
        self._record(pre, obs0, actions[0], ex, info)
        for ep in info["episodes"]:
            ep["steps"] = self.total_steps
            ep["update"] = self.updates
            self.history["episodes"].append(ep)
        if info["episodes"]:
            self._curriculum()
        if self._pos == self.cfg.rollout_len:
            batch = dict(self._buf)
            batch["last_values"] = self.agent.value(env.obs)
            batch["gamma"], batch["lam"] = self.cfg.gamma, self.cfg.gae_lambda
            self._new_buffer()
            self._update_gen = self.agent.update(batch)

    def _record(self, pre, obs0, action0, ex, info):
        frame = build_frame(pre, obs0, action0, ex, info, 0, self._frames)
        self._frames.append(frame)
        self.live = frame
        if 0 in info.get("terminal_states", {}):
            ep = finish_episode(self._frames, info, 0, self.world, self.world_version, self.updates)
            self.recent_episodes.append(ep)
            if self.best_episode is None or ep["summary"]["return"] > self.best_episode["summary"]["return"]:
                self.best_episode = ep
            self._frames = []

    def _curriculum(self):
        if not self.cfg.curriculum or self.env.difficulty >= 1.0:
            return
        eps = [e for e in self.history["episodes"][-self.cfg.curriculum_window:]
               if abs(e["difficulty"] - self.env.difficulty) < 1e-9]
        if len(eps) >= self.cfg.curriculum_window and np.mean([e["success"] for e in eps]) >= self.cfg.curriculum_up:
            self.env.difficulty = min(1.0, round(self.env.difficulty + self.cfg.curriculum_step, 6))

    def success_rate(self, window=100):
        eps = self.history["episodes"][-window:]
        return float(np.mean([e["success"] for e in eps])) if eps else 0.0

    def summary(self, window=100):
        eps = self.history["episodes"][-window:]
        mean = lambda k: float(np.mean([e[k] for e in eps])) if eps else 0.0
        return {
            "steps": self.total_steps, "episodes": len(self.history["episodes"]), "updates": self.updates,
            "success_rate": mean("success"), "mean_return": mean("return"),
            "collision_rate": float(np.mean([e["collision"] is not None for e in eps])) if eps else 0.0,
            "mean_inside": mean("inside"), "mean_angle_err": mean("angle_err"),
            "difficulty": float(self.env.difficulty), "sps": self.steps_per_second(), "train_time": self.train_time,
            "agent": self.agent.name,
        }

    def set_world(self, world: World):
        self.world = world
        self.world_version += 1
        self.env.set_world(world)
        self._new_buffer()
        self._frames = []
        self.live = None

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "format": 1, "agent_name": self.agent.name, "agent_config": self.agent.config,
            "agent_state": self.agent.state_dict(), "train": asdict(self.cfg), "sim": asdict(self.env.sim),
            "reward": asdict(self.env.reward_cfg), "world": self.world.to_dict(), "history": self.history,
            "total_steps": self.total_steps, "updates": self.updates, "train_time": self.train_time,
            "difficulty": float(self.env.difficulty),
            "recent_episodes": [self._pack(e) for e in self.recent_episodes],
            "best_episode": self._pack(self.best_episode) if self.best_episode else None,
        }
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @staticmethod
    def _pack(ep):
        out = dict(ep)
        out["world"] = ep["world"].to_dict()
        return out

    @staticmethod
    def _unpack(ep):
        out = dict(ep)
        out["world"] = World.from_dict(ep["world"])
        return out

    @classmethod
    def load(cls, path, world: World = None):
        with open(path, "rb") as f:
            data = pickle.load(f)
        train = TrainConfig(**data["train"])
        train.agent = data["agent_name"] if data["agent_name"] in AGENTS else train.agent
        w = world or World.from_dict(data["world"])
        t = cls(w, train, SimConfig(**data["sim"]), RewardConfig(**data["reward"]), agent_config=data["agent_config"])
        t.agent.load_state_dict(data["agent_state"])
        t.history = data["history"]
        t.total_steps, t.updates, t.train_time = data["total_steps"], data["updates"], data["train_time"]
        t.env.difficulty = data["difficulty"]
        t.env.episode_count = t.history["episodes"][-1]["episode"] if t.history["episodes"] else 0
        t.recent_episodes.extend(cls._unpack(e) for e in data["recent_episodes"])
        t.best_episode = cls._unpack(data["best_episode"]) if data["best_episode"] else None
        return t
