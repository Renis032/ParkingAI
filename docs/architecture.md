# ParkingAI architecture and data contracts

Stack: Python 3.8, numpy, pygame. Nothing else. The neural network and PPO are written in plain numpy.

## Modules

| Module | Role |
|---|---|
| `parkai/geometry.py` | Vectorized rectangles, segment intersection, ray casting, path interpolation |
| `parkai/world.py` | Editable lot model (`World`, `Slot`, `ParkedCar`, `Obstacle`, `Spawn`, `MovingObject`), JSON load/save, `default_world()` |
| `parkai/env.py` | `ParkingEnv`: N parallel cars on one lot, kinematic bicycle model, lidar, collisions, reward. `SimConfig`, `RewardConfig` |
| `parkai/agents/` | `Agent` API (`base.py`), `RandomAgent`, `PPOAgent` (`ppo.py`, `nn.py`) |
| `parkai/trainer.py` | `Trainer`: time-sliced training loop, history, episode recording, curriculum, checkpoints |
| `parkai/player.py` | `Player`: single car driven by the current policy (WATCH mode) |
| `parkai/headless.py` | Training without a window |
| `parkai/ui/` | pygame app: `app.py`, `lot_view.py`, `editor.py`, `controls.py`, `widgets.py`, `panels.py`, `theme.py` |

## Coordinates

Meters. Origin is the top-left corner of the lot, x to the right, y down. Angles are radians, 0 = +x, positive = clockwise on screen. A rectangle is `(x, y, angle, length, width)` with `(x, y)` its center and `length` along `angle`. Lot size is `world.width x world.height`, and the lot border acts as a wall.

A `Slot.angle` points from the slot's entrance into the slot. The car may park nose-in or reverse-in when `SimConfig.allow_reverse_parking` is true.

## Actions

9 discrete actions: `steer = a % 3 - 1` (-1 left, 0 straight, +1 right) and `throttle = a // 3 - 1` (-1 reverse, 0 brake, +1 forward). Names are in `env.ACTION_NAMES`, `STEER_NAMES` and `THROTTLE_NAMES`.

## Observation (`env.obs_names`, 39 values with 16 rays)

`goal_fwd`, `goal_side`: slot center in the car frame / 10. `slot_cos`, `slot_sin`: slot angle relative to the car heading. `speed`: v / max_speed. `steer`: steer / max_steer. `inside`: fraction of the car inside the target slot. `ray0..ray15`: clearance from the car body along ray i / ray_range, where 1 means nothing in range. Ray i points at heading `-pi + i * 2pi / n_rays`, so ray 8 points straight ahead and ray 0 points straight back. `rayN_moving`: 1 if that ray hits a moving object.

## Reward (`RewardConfig`, per step, components in `env.COMPONENT_NAMES`)

- `distance`: weight x the change in -min(dist, scale)/scale, so moving closer to the slot pays.
- `inside`: weight x the change in the fraction of the car inside the slot, plus `inside_per_step` x fraction.
- `straight`: weight x the change in fraction x straightness, where straightness = 1 - angle_err / straight_tolerance_deg, clipped to 0..1.
- `time`: -time per step.
- `collision`: minus the sum of penalties for every kind touched (wall, obstacle, parked_car = 1 point; moving_car = 10; pedestrian = 20).
- `success`: +success when inside >= 0.97, angle_err <= 6 degrees and |v| <= 0.3 m/s. The episode ends.

An episode ends on success, on collision (if `end_on_collision`), or on timeout (`max_steps`, a truncation that the trainer bootstraps).

## Agent API (`parkai/agents/base.py`)

- `act(obs (N,O), deterministic=False) -> actions (N,) int, {"probs": (N,9), "value": (N,), "logp": (N,)}`
- `value(obs) -> (N,)`
- `update(batch)` is a generator. It yields `None` after each chunk of work (one minibatch) so the UI stays responsive, and its last yield is a metrics dict with `policy_loss, value_loss, entropy, approx_kl, clip_frac, explained_var` (plus anything else).
- `batch`: `obs (T,N,O) float32, actions (T,N), logp (T,N), values (T,N), rewards (T,N), dones (T,N) bool, last_values (N,), gamma, lam`. Rewards at truncated steps already include `gamma * V(terminal_obs)`, and `dones` is true for both terminations and truncations.
- `inspect(obs_row) -> {"layers": [{"name": str, "values": 1-D array}, ...], "probs": (9,), "value": float}`: activations used to draw the network.
- `state_dict()` / `load_state_dict(d)`: picklable state, including optimizer state.

## Frame (`trainer.live`, `player.frames[i]`, `episode["frames"][i]`)

Keys: `x, y, angle, v, steer, t, step`; `slot` (5,); `moving` (M,5) rectangles of moving objects and `moving_kinds` (kind codes); `ray_ends` (R,2) world points; `ray_moving` (R,) bool; `inside` (0..1); `angle_err` (degrees); `obs` (O,); `action` (0..8, or -1 on the terminal frame); `probs` (9,); `value`; `reward`; `components` (6,); `collision` (5,) bool, ordered by `world.KIND_NAMES`; `return` (cumulative). The last frame of an episode has `terminal: True` and shows the final pose. The pose in a frame is the pose at which the decision was made.

## Episode record (`trainer.recent_episodes` deque of the last 40 episodes of training car 0, `trainer.best_episode`, `player.finished`)

`{"summary": {...}, "frames": [...], "world": World, "world_version": int, "update": int}`. Summary keys: `env, episode, return, length, success, collision` (worst kind name or None), `collisions` ({kind: count}), `inside, angle_err, timeout, components` ({name: sum}), `difficulty, steps, update`.

## History

`trainer.history["episodes"]` holds every episode summary. `trainer.history["updates"]` holds one dict per PPO update: the agent metrics plus `update, steps, difficulty, success_rate, sps`. `trainer.summary()` gives rolling numbers for the top bar.

## Trainer

- `run(seconds, max_steps=None)`: trains for a wall-clock budget, optionally capped at N vector steps, and resumes an interrupted update first.
- `set_world(world)`: keeps the agent and swaps the lot.
- `save(path)` / `Trainer.load(path, world=None)`: checkpoints are pickles holding weights, optimizer, history, configs, the world and recorded episodes.
- Curriculum: `env.difficulty` (0..1) blends the spawn from "just in front of the slot" to the user's spawn points. It goes up by 0.1 once the success rate over the last 100 episodes at the current level reaches 70%.

## Value map

`env.observe_poses(xs, ys, angles, slot_row)` builds observations for arbitrary car poses (speed 0). Feed them to `agent.value` / `agent.act(..., deterministic=True)` to draw what the agent thinks of every spot in the lot.

## Moving objects (future-ready)

`MovingObject(kind="car"|"pedestrian", path=[[x,y],...], speed, closed, phase, random_phase)` follows its path (closed loop, or back and forth) on each env's own clock. Moving objects already take part in collisions (with large penalties), lidar (`rayN_moving`) and rendering (`frame["moving"]`). Smarter behaviour (yielding, random routes) belongs in `ParkingEnv.moving_rects`.
