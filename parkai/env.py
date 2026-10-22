import math
from dataclasses import dataclass, asdict

import numpy as np

from . import geometry as geo
from .world import World, KIND_NAMES, CAR_LENGTH, CAR_WIDTH

N_ACTIONS = 9
STEER_NAMES = ["left", "straight", "right"]
THROTTLE_NAMES = ["reverse", "brake", "forward"]
ACTION_NAMES = [f"{THROTTLE_NAMES[a // 3]}+{STEER_NAMES[a % 3]}" for a in range(N_ACTIONS)]
COMPONENT_NAMES = ["distance", "inside", "straight", "time", "collision", "success"]


def decode_action(a):
    a = np.asarray(a)
    return a % 3 - 1, a // 3 - 1


@dataclass
class SimConfig:
    dt: float = 0.1
    max_steps: int = 400
    car_length: float = CAR_LENGTH
    car_width: float = CAR_WIDTH
    wheelbase: float = 2.7
    max_steer: float = 0.6
    steer_rate: float = 3.0
    accel: float = 2.5
    brake: float = 4.0
    max_speed: float = 4.0
    max_reverse: float = 2.5
    n_rays: int = 16
    ray_range: float = 12.0
    allow_reverse_parking: bool = True
    straight_tolerance_deg: float = 45.0
    success_inside: float = 0.97
    success_angle_deg: float = 6.0
    success_speed: float = 0.3
    end_on_collision: bool = True
    difficulty: float = 1.0


@dataclass
class RewardConfig:
    distance: float = 2.0
    distance_scale: float = 20.0
    inside: float = 3.0
    straight: float = 3.0
    inside_per_step: float = 0.0
    time: float = 0.002
    success: float = 10.0
    wall: float = 1.0
    obstacle: float = 1.0
    parked_car: float = 1.0
    moving_car: float = 10.0
    pedestrian: float = 20.0

    def penalties(self):
        return np.array([self.wall, self.obstacle, self.parked_car, self.moving_car, self.pedestrian])


def obs_names(n_rays):
    names = ["goal_fwd", "goal_side", "slot_cos", "slot_sin", "speed", "steer", "inside"]
    names += [f"ray{i}" for i in range(n_rays)]
    names += [f"ray{i}_moving" for i in range(n_rays)]
    return names


class ParkingEnv:
    def __init__(self, world: World, num_envs=16, sim: SimConfig = None, reward: RewardConfig = None, seed=None):
        self.world = world
        self.n = num_envs
        self.sim = sim or SimConfig()
        self.reward_cfg = reward or RewardConfig()
        self.rng = np.random.default_rng(seed)
        self.difficulty = self.sim.difficulty
        s = self.sim
        self.n_rays = s.n_rays
        self.obs_names = obs_names(s.n_rays)
        self.obs_dim = len(self.obs_names)
        self.n_actions = N_ACTIONS

        self.ray_angles = np.linspace(-np.pi, np.pi, s.n_rays, endpoint=False)
        ca, sa = np.abs(np.cos(self.ray_angles)), np.abs(np.sin(self.ray_angles))
        with np.errstate(divide="ignore"):
            self.ray_body = np.minimum(np.where(ca > 1e-9, s.car_length / 2 / ca, np.inf),
                                       np.where(sa > 1e-9, s.car_width / 2 / sa, np.inf))
        gx = (np.arange(9) + 0.5) / 9 - 0.5
        gy = (np.arange(5) + 0.5) / 5 - 0.5
        g = np.stack(np.meshgrid(gx * s.car_length, gy * s.car_width), -1).reshape(-1, 2)
        self.body_points = g

        self._build_world()
        z = lambda: np.zeros(self.n)
        self.x, self.y, self.th, self.v, self.steer, self.t = z(), z(), z(), z(), z(), z()
        self.steps = np.zeros(self.n, dtype=np.int64)
        self.slot = np.zeros((self.n, 5))
        self.phase = np.zeros((self.n, len(self.movers)))
        self.potential = z()
        self.prev_terms = np.zeros((self.n, 3))
        self.ep_return = z()
        self.ep_components = np.zeros((self.n, len(COMPONENT_NAMES)))
        self.ep_collisions = np.zeros((self.n, len(KIND_NAMES)), dtype=np.int64)
        self.ray_dist = np.zeros((self.n, s.n_rays))
        self.ray_moving = np.zeros((self.n, s.n_rays), dtype=bool)
        self.inside = z()
        self.angle_err = z()
        self.episode_count = 0
        self.obs = self.reset()

    def _build_world(self):
        w = self.world
        self.static_segs, self.static_seg_kinds = w.static_segments()
        rects, kinds = w.static_rects()
        self.static_rects, self.static_kinds = rects, kinds
        self.static_corners = geo.rect_corners(*rects.T) if len(rects) else np.zeros((0, 4, 2))
        self.static_edges = geo.poly_edges(self.static_corners).reshape(-1, 4) if len(rects) else np.zeros((0, 4))
        self.targets = w.target_slots()
        self.movers = []
        for m in w.moving:
            pts = np.array(m.path, dtype=float).reshape(-1, 2)
            if len(pts) == 0:
                continue
            if len(pts) == 1:
                pts = np.vstack([pts, pts])
            if m.closed:
                pts = np.vstack([pts, pts[:1]])
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            length, width = m.size()
            self.movers.append(dict(points=pts, cum=cum, closed=m.closed, speed=m.speed, phase=m.phase,
                                    random_phase=m.random_phase, length=length, width=width, kind=m.kind_code()))

    def set_world(self, world: World):
        self.world = world
        self._build_world()
        self.phase = np.zeros((self.n, len(self.movers)))
        self.obs = self.reset()
        return self.obs

    def moving_rects(self, t=None, phase=None):
        t = self.t if t is None else np.asarray(t, float)
        phase = self.phase if phase is None else phase
        m = len(self.movers)
        out = np.zeros(t.shape + (m, 5))
        kinds = np.zeros(m, dtype=np.int64)
        for j, mv in enumerate(self.movers):
            s = mv["speed"] * t + phase[..., j]
            x, y, a = geo.path_pose(mv["points"], mv["cum"], s, mv["closed"])
            out[..., j, 0], out[..., j, 1], out[..., j, 2] = x, y, a
            out[..., j, 3], out[..., j, 4] = mv["length"], mv["width"]
            kinds[j] = mv["kind"]
        return out, kinds

    def collisions(self, cx, cy, cth, dyn=None, dyn_kinds=None):
        s = self.sim
        n = len(cx)
        hits = np.zeros((n, len(KIND_NAMES)), dtype=bool)
        corners = geo.rect_corners(cx, cy, cth, s.car_length, s.car_width)
        edges = geo.poly_edges(corners)
        out = ((corners[..., 0] < 0) | (corners[..., 0] > self.world.width) |
               (corners[..., 1] < 0) | (corners[..., 1] > self.world.height)).any(1)
        hits[:, 0] = out
        ns = len(self.static_rects)
        if ns:
            r = self.static_rects
            e = geo.segments_intersect(edges[:, :, None, :], self.static_edges[None, None, :, :])
            e = e.reshape(n, 4, ns, 4).any(axis=(1, 3))
            cin = geo.points_in_rects(corners[:, :, None, :], r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]).any(1)
            vin = geo.points_in_rects(self.static_corners[None], cx[:, None, None], cy[:, None, None],
                                      cth[:, None, None], s.car_length, s.car_width).any(2)
            hit_rect = e | cin | vin
            for k in np.unique(self.static_kinds):
                hits[:, k] |= hit_rect[:, self.static_kinds == k].any(1)
        if dyn is not None and dyn.shape[1]:
            m = dyn.shape[1]
            dc = geo.rect_corners(dyn[..., 0], dyn[..., 1], dyn[..., 2], dyn[..., 3], dyn[..., 4])
            de = geo.poly_edges(dc).reshape(n, m * 4, 4)
            e = geo.segments_intersect(edges[:, :, None, :], de[:, None, :, :])
            e = e.reshape(n, 4, m, 4).any(axis=(1, 3))
            cin = geo.points_in_rects(corners[:, :, None, :], dyn[:, None, :, 0], dyn[:, None, :, 1],
                                      dyn[:, None, :, 2], dyn[:, None, :, 3], dyn[:, None, :, 4]).any(1)
            vin = geo.points_in_rects(dc, cx[:, None, None], cy[:, None, None], cth[:, None, None],
                                      s.car_length, s.car_width).any(2)
            hit_dyn = e | cin | vin
            for k in np.unique(dyn_kinds):
                hits[:, k] |= hit_dyn[:, dyn_kinds == k].any(1)
        return hits

    def sense(self, cx, cy, cth, dyn=None):
        s = self.sim
        origins = np.stack([cx, cy], -1)
        ang = cth[:, None] + self.ray_angles[None, :]
        dirs = np.stack([np.cos(ang), np.sin(ang)], -1)
        dist, _ = geo.cast_rays(origins, dirs, self.static_segs, s.ray_range + self.ray_body.max())
        moving = np.zeros_like(dist, dtype=bool)
        if dyn is not None and dyn.shape[1]:
            m = dyn.shape[1]
            dc = geo.rect_corners(dyn[..., 0], dyn[..., 1], dyn[..., 2], dyn[..., 3], dyn[..., 4])
            de = geo.poly_edges(dc).reshape(len(cx), m * 4, 4)
            dd, _ = geo.cast_rays(origins, dirs, de, s.ray_range + self.ray_body.max())
            moving = dd < dist
            dist = np.minimum(dist, dd)
        dist = np.minimum(dist, s.ray_range + self.ray_body[None, :])
        return dist, moving

    def parking_terms(self, cx, cy, cth, slot):
        s = self.sim
        pts = geo.to_world(self.body_points[None], cx[:, None], cy[:, None], cth[:, None])
        inside = geo.points_in_rects(pts, slot[:, 0:1], slot[:, 1:2], slot[:, 2:3], slot[:, 3:4], slot[:, 4:5]).mean(1)
        d = geo.wrap_angle(cth - slot[:, 2])
        if s.allow_reverse_parking:
            err = np.degrees(np.arccos(np.clip(np.abs(np.cos(d)), 0, 1)))
        else:
            err = np.degrees(np.abs(d))
        dist = np.hypot(cx - slot[:, 0], cy - slot[:, 1])
        return inside, err, dist

    def observe(self, cx, cy, cth, v, steer, slot, dyn=None):
        s = self.sim
        rays, moving = self.sense(cx, cy, cth, dyn)
        inside, err, dist = self.parking_terms(cx, cy, cth, slot)
        local = geo.to_local(slot[:, :2], cx, cy, cth)
        rel = slot[:, 2] - cth
        clearance = np.clip((rays - self.ray_body[None, :]) / s.ray_range, 0, 1)
        obs = np.concatenate([
            np.clip(local / 10.0, -3, 3),
            np.cos(rel)[:, None], np.sin(rel)[:, None],
            (v / s.max_speed)[:, None], (steer / s.max_steer)[:, None],
            inside[:, None],
            clearance, moving.astype(float),
        ], 1)
        return obs, rays, moving, inside, err, dist

    def observe_poses(self, cx, cy, cth, slot_row, env_index=0):
        n = len(cx)
        slot = np.repeat(np.asarray(slot_row, float)[None], n, 0)
        dyn = None
        if self.movers:
            d, _ = self.moving_rects(self.t[env_index:env_index + 1], self.phase[env_index:env_index + 1])
            dyn = np.repeat(d, n, 0)
        z = np.zeros(n)
        return self.observe(np.asarray(cx, float), np.asarray(cy, float), np.asarray(cth, float), z, z, slot, dyn)[0]

    def _potential_terms(self, inside, err, dist):
        r = self.reward_cfg
        straight = np.clip(1 - err / self.sim.straight_tolerance_deg, 0, 1)
        return np.stack([-np.minimum(dist, r.distance_scale) / r.distance_scale, inside, inside * straight], 1)

    def _approach_pose(self, slot, rng, dyn_one=None, dyn_kinds=None):
        s = self.sim
        sx, sy, sa, sl, sw = slot
        ax, ay = math.cos(sa), math.sin(sa)
        nx, ny = -ay, ax
        back = sl / 2 + s.car_length / 2 + 1.0
        side = sw / 2 + s.car_width / 2 + 1.0
        ahead = s.car_length * 0.6
        behind = (sx - ax * back, sy - ay * back, sa)
        beside = [(sx + nx * side * k + ax * ahead, sy + ny * side * k + ay * ahead, sa) for k in (1, -1)]
        free = lambda p: not self.collisions(np.array([p[0]]), np.array([p[1]]), np.array([p[2]]),
                                             dyn_one, dyn_kinds).any()
        if free(behind):
            return np.array(behind)
        ok = [p for p in beside if free(p)]
        if ok:
            return np.array(ok[rng.integers(len(ok))])
        for radius in np.linspace(4.0, max(self.world.width, self.world.height), 8):
            for _ in range(10):
                r, a = rng.uniform(3.0, radius), rng.uniform(-np.pi, np.pi)
                p = (sx + r * math.cos(a), sy + r * math.sin(a), rng.uniform(-np.pi, np.pi))
                if free(p):
                    return np.array(p)
        return np.array(behind)

    def _sample_spawn(self, slot, rng, dyn_one=None, dyn_kinds=None):
        appr = self._approach_pose(slot, rng, dyn_one, dyn_kinds)
        if self.world.spawns:
            sp = self.world.spawns[rng.integers(len(self.world.spawns))]
            user = np.array([sp.x, sp.y, sp.angle])
            jp, ja = sp.jitter_pos, sp.jitter_angle
        else:
            user = np.array([self.world.width / 2, self.world.height / 2, 0.0])
            jp, ja = 3.0, math.pi
        d = float(np.clip(self.difficulty, 0, 1))
        base = appr + (user - appr) * d
        base[2] = appr[2] + geo.wrap_angle(user[2] - appr[2]) * d
        scale = 0.3 + 0.7 * d
        free = lambda p: not self.collisions(np.array([p[0]]), np.array([p[1]]), np.array([p[2]]),
                                             dyn_one, dyn_kinds).any()
        for _ in range(30):
            p = base + rng.uniform(-1, 1, 3) * np.array([jp, jp, ja]) * scale
            if free(p):
                return p[0], p[1], geo.wrap_angle(p[2])
        for p in (base, user, appr):
            if free(p):
                break
        return p[0], p[1], geo.wrap_angle(p[2])

    def reset(self, ids=None):
        ids = np.arange(self.n) if ids is None else np.asarray(ids)
        if len(ids) == 0:
            return self.obs
        s = self.sim
        for i in ids:
            if self.targets:
                sl = self.targets[self.rng.integers(len(self.targets))]
                self.slot[i] = (sl.x, sl.y, sl.angle, sl.length, sl.width)
            else:
                self.slot[i] = (self.world.width / 2, self.world.height / 2, 0.0, 5.5, 2.7)
            for j, mv in enumerate(self.movers):
                period = mv["cum"][-1] * (1 if mv["closed"] else 2)
                self.phase[i, j] = mv["phase"] + (self.rng.uniform(0, period) if mv["random_phase"] else 0.0)
            self.t[i] = 0.0
            dyn_one, kinds = (None, None)
            if self.movers:
                d, kinds = self.moving_rects(self.t[i:i + 1], self.phase[i:i + 1])
                dyn_one = d
            self.x[i], self.y[i], self.th[i] = self._sample_spawn(self.slot[i], self.rng, dyn_one, kinds)
        self.v[ids] = 0
        self.steer[ids] = 0
        self.steps[ids] = 0
        self.ep_return[ids] = 0
        self.ep_components[ids] = 0
        self.ep_collisions[ids] = 0
        dyn = self.moving_rects(self.t[ids], self.phase[ids])[0] if self.movers else None
        obs, rays, moving, inside, err, dist = self.observe(self.x[ids], self.y[ids], self.th[ids], self.v[ids],
                                                            self.steer[ids], self.slot[ids], dyn)
        self.prev_terms[ids] = self._potential_terms(inside, err, dist)
        self.ray_dist[ids], self.ray_moving[ids] = rays, moving
        self.inside[ids], self.angle_err[ids] = inside, err
        if len(ids) == self.n:
            self.obs = obs
        else:
            self.obs[ids] = obs
        return self.obs

    def set_pose(self, i, x, y, angle):
        self.x[i], self.y[i], self.th[i] = x, y, geo.wrap_angle(angle)
        self.v[i] = self.steer[i] = 0.0
        self.steps[i] = 0
        self.ep_return[i] = 0
        self.ep_components[i] = 0
        self.ep_collisions[i] = 0
        sl = slice(i, i + 1)
        dyn = self.moving_rects(self.t[sl], self.phase[sl])[0] if self.movers else None
        obs, rays, moving, inside, err, dist = self.observe(self.x[sl], self.y[sl], self.th[sl], self.v[sl],
                                                            self.steer[sl], self.slot[sl], dyn)
        self.obs[i] = obs[0]
        self.prev_terms[i] = self._potential_terms(inside, err, dist)[0]
        self.ray_dist[i], self.ray_moving[i] = rays[0], moving[0]
        self.inside[i], self.angle_err[i] = inside[0], err[0]
        return self.obs

    def pose_collides(self, x, y, angle, i=0):
        dyn, kinds = (None, None)
        if self.movers:
            dyn, kinds = self.moving_rects(self.t[i:i + 1], self.phase[i:i + 1])
        return bool(self.collisions(np.array([x]), np.array([y]), np.array([angle]), dyn, kinds).any())

    def step(self, actions):
        s, r = self.sim, self.reward_cfg
        steer_cmd, thr = decode_action(actions)
        target = steer_cmd * s.max_steer
        self.steer += np.clip(target - self.steer, -s.steer_rate * s.dt, s.steer_rate * s.dt)
        v = self.v
        coast = -np.sign(v) * np.minimum(s.brake, np.abs(v) / s.dt)
        drive = thr * np.where(thr * v < 0, s.brake, s.accel)
        a = np.where(thr == 0, coast, drive)
        v_new = np.clip(v + a * s.dt, -s.max_reverse, s.max_speed)
        b = s.wheelbase / 2
        rx = self.x - b * np.cos(self.th)
        ry = self.y - b * np.sin(self.th)
        rx = rx + v_new * np.cos(self.th) * s.dt
        ry = ry + v_new * np.sin(self.th) * s.dt
        th_new = geo.wrap_angle(self.th + v_new / s.wheelbase * np.tan(self.steer) * s.dt)
        x_new = rx + b * np.cos(th_new)
        y_new = ry + b * np.sin(th_new)

        self.t += s.dt
        self.steps += 1
        dyn, dyn_kinds = self.moving_rects() if self.movers else (None, None)
        hits = self.collisions(x_new, y_new, th_new, dyn, dyn_kinds)
        collided = hits.any(1)
        if not s.end_on_collision:
            x_new = np.where(collided, self.x, x_new)
            y_new = np.where(collided, self.y, y_new)
            th_new = np.where(collided, self.th, th_new)
            v_new = np.where(collided, 0.0, v_new)
        self.x, self.y, self.th, self.v = x_new, y_new, th_new, v_new

        obs, rays, moving, inside, err, dist = self.observe(self.x, self.y, self.th, self.v, self.steer, self.slot, dyn)
        terms = self._potential_terms(inside, err, dist)
        delta = terms - self.prev_terms
        self.prev_terms = terms
        success = (inside >= s.success_inside) & (err <= s.success_angle_deg) & (np.abs(self.v) <= s.success_speed)
        comp = np.zeros((self.n, len(COMPONENT_NAMES)))
        comp[:, 0] = r.distance * delta[:, 0]
        comp[:, 1] = r.inside * delta[:, 1] + r.inside_per_step * inside
        comp[:, 2] = r.straight * delta[:, 2]
        comp[:, 3] = -r.time
        comp[:, 4] = -(hits * r.penalties()[None, :]).sum(1)
        comp[:, 5] = r.success * success
        reward = comp.sum(1)

        terminated = success | (collided & s.end_on_collision)
        truncated = (self.steps >= s.max_steps) & ~terminated
        done = terminated | truncated
        self.ep_return += reward
        self.ep_components += comp
        self.ep_collisions += hits
        self.ray_dist, self.ray_moving = rays, moving
        self.inside, self.angle_err = inside, err

        info = {"components": comp, "inside": inside.copy(), "angle_err": err.copy(), "collision": hits,
                "success": success, "terminated": terminated, "truncated": truncated,
                "terminal_obs": obs.copy(), "episodes": []}
        ids = np.nonzero(done)[0]
        pen = r.penalties()
        for i in ids:
            kinds_hit = np.nonzero(self.ep_collisions[i])[0]
            worst = KIND_NAMES[kinds_hit[np.argmax(pen[kinds_hit])]] if len(kinds_hit) else None
            self.episode_count += 1
            info["episodes"].append({
                "env": int(i), "episode": self.episode_count, "return": float(self.ep_return[i]),
                "length": int(self.steps[i]), "success": bool(success[i]), "collision": worst,
                "collisions": {KIND_NAMES[k]: int(self.ep_collisions[i, k]) for k in kinds_hit},
                "inside": float(inside[i]), "angle_err": float(err[i]),
                "timeout": bool(truncated[i]),
                "components": dict(zip(COMPONENT_NAMES, self.ep_components[i].tolist())),
                "difficulty": float(self.difficulty),
            })
        self.obs = obs
        if len(ids):
            info["terminal_states"] = {int(i): self.state(i) for i in ids}
            self.reset(ids)
        return self.obs, reward, done, info

    def state(self, i=0):
        dyn = self.moving_rects(self.t[i:i + 1], self.phase[i:i + 1])[0][0] if self.movers else np.zeros((0, 5))
        ang = self.th[i] + self.ray_angles
        ends = np.stack([self.x[i] + np.cos(ang) * self.ray_dist[i], self.y[i] + np.sin(ang) * self.ray_dist[i]], -1)
        return {
            "x": float(self.x[i]), "y": float(self.y[i]), "angle": float(self.th[i]),
            "v": float(self.v[i]), "steer": float(self.steer[i]), "t": float(self.t[i]), "step": int(self.steps[i]),
            "slot": self.slot[i].copy(), "moving": dyn, "moving_kinds": [m["kind"] for m in self.movers],
            "ray_ends": ends, "ray_moving": self.ray_moving[i].copy(),
            "inside": float(self.inside[i]), "angle_err": float(self.angle_err[i]),
        }

    def config_dict(self):
        return {"sim": asdict(self.sim), "reward": asdict(self.reward_cfg)}
