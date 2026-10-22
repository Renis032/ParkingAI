import math
import time

import numpy as np
import pygame

from .. import geometry as geo
from ..world import PEDESTRIAN
from ..env import ACTION_NAMES
from . import theme as T


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c[:3])


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


HEAT_STOPS = [(0.0, (46, 40, 120)), (0.35, (36, 120, 190)), (0.65, (90, 200, 130)), (1.0, (252, 232, 80))]


def heat_colors(t):
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,))
    for (t0, c0), (t1, c1) in zip(HEAT_STOPS[:-1], HEAT_STOPS[1:]):
        m = (t >= t0) & (t <= t1)
        f = ((t - t0) / (t1 - t0))[..., None]
        out = np.where(m[..., None], np.array(c0) + (np.array(c1) - np.array(c0)) * f, out)
    return out.astype(np.uint8)


def dashed_line(surf, color, p0, p1, dash=9, gap=6, width=2):
    x0, y0 = p0
    x1, y1 = p1
    d = math.hypot(x1 - x0, y1 - y0)
    if d < 1:
        return
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    s = 0.0
    while s < d:
        e = min(s + dash, d)
        pygame.draw.line(surf, color, (x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e), width)
        s += dash + gap


def arrow(surf, color, p0, p1, head=7, width=2):
    pygame.draw.line(surf, color, p0, p1, width)
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (-0.5, 0.5):
        pygame.draw.line(surf, color, p1, (p1[0] - head * math.cos(a + s), p1[1] - head * math.sin(a + s)), width)


class LotView:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, 100, 100)
        self.scale = 10.0
        self.ox = 0.0
        self.oy = 0.0
        self.manual = False
        self._fit_key = None
        self.pan = None
        self.show_rays = True
        self.show_trail = True
        self.show_ghosts = False
        self.show_value = False
        self.show_policy = False
        self.show_spawns = True
        self.show_grid = False
        self.effects = []
        self.maps = None
        self.maps_time = 0.0
        self._static = None
        self._static_key = None
        self._static_world = None
        self._sprites = {}
        self._overlay = None
        self._heat_scaled = None

    def set_rect(self, rect, world):
        self.rect = pygame.Rect(rect)
        key = (self.rect.size, world.width, world.height)
        if key != self._fit_key and (not self.manual or self._fit_key is None):
            self.fit(world)
        self._fit_key = key

    def fit(self, world):
        self.manual = False
        m = 24
        self.scale = max(2.0, min((self.rect.w - 2 * m) / world.width, (self.rect.h - 2 * m) / world.height))
        self.ox = (self.rect.w - world.width * self.scale) / 2
        self.oy = (self.rect.h - world.height * self.scale) / 2

    def w2s(self, x, y):
        return (self.rect.x + self.ox + x * self.scale, self.rect.y + self.oy + y * self.scale)

    def s2w(self, px, py):
        return ((px - self.rect.x - self.ox) / self.scale, (py - self.rect.y - self.oy) / self.scale)

    def poly(self, x, y, a, l, w):
        c = geo.rect_corners(x, y, a, l, w)
        return [self.w2s(p[0], p[1]) for p in c]

    def zoom(self, factor, pos):
        wx, wy = self.s2w(*pos)
        self.scale = max(3.0, min(150.0, self.scale * factor))
        self.ox = pos[0] - self.rect.x - wx * self.scale
        self.oy = pos[1] - self.rect.y - wy * self.scale
        self.manual = True

    def handle_event(self, e, mouse):
        if e.type == pygame.MOUSEWHEEL and self.rect.collidepoint(mouse):
            self.zoom(1.12 ** e.y, mouse)
            return True
        if e.type == pygame.MOUSEBUTTONDOWN and e.button in (2, 3) and self.rect.collidepoint(e.pos):
            self.pan = e.pos
            return True
        if e.type == pygame.MOUSEMOTION and self.pan is not None:
            self.ox += e.pos[0] - self.pan[0]
            self.oy += e.pos[1] - self.pan[1]
            self.pan = e.pos
            self.manual = True
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button in (2, 3) and self.pan is not None:
            self.pan = None
            return True
        return False

    def sprite(self, color, length_px, width_px, headlights=True):
        L, W = max(int(length_px), 6), max(int(width_px), 4)
        key = (tuple(color[:3]), L, W)
        s = self._sprites.get(key)
        if s is None:
            s = pygame.Surface((L, W), pygame.SRCALPHA)
            r = max(2, int(W * 0.3))
            pygame.draw.rect(s, color, (0, 0, L, W), border_radius=r)
            pygame.draw.rect(s, shade(color, 0.6), (0, 0, L, W), 1, border_radius=r)
            glass = (28, 34, 46)
            wx, ww = int(L * 0.56), max(2, int(L * 0.15))
            pygame.draw.polygon(s, glass, [(wx, int(W * .14)), (wx + ww, int(W * .24)), (wx + ww, int(W * .76)),
                                           (wx, int(W * .86))])
            pygame.draw.rect(s, glass, (int(L * 0.13), int(W * .2), max(1, int(L * 0.07)), int(W * .6)))
            pygame.draw.rect(s, shade(color, 1.15), (int(L * 0.3), int(W * .18), int(L * 0.22), int(W * .64)),
                             border_radius=2)
            if headlights and L > 14:
                for yy in (0.18, 0.82):
                    pygame.draw.circle(s, (255, 244, 190), (L - 3, int(W * yy)), max(1, W // 9))
            if len(self._sprites) > 400:
                self._sprites.clear()
            self._sprites[key] = s
        return s

    def blit_car(self, surf, x, y, a, l, w, color, alpha=None):
        s = self.sprite(color, l * self.scale, w * self.scale)
        r = pygame.transform.rotate(s, -math.degrees(a))
        if alpha is not None:
            r.set_alpha(alpha)
        cx, cy = self.w2s(x, y)
        surf.blit(r, r.get_rect(center=(int(cx), int(cy))))

    def _draw_static(self, surf, world, rev):
        key = (rev, round(self.scale, 3), round(self.ox, 1), round(self.oy, 1), self.rect.size, self.show_spawns,
               self.show_grid)
        if self._static is None or self._static_world is not world or key != self._static_key:
            self._static = self._render_static(world)
            self._static_key = key
            self._static_world = world
        surf.blit(self._static, self.rect.topleft)

    def _render_static(self, world):
        s = pygame.Surface(self.rect.size)
        s.fill(T.BG)
        saved = self.rect
        self.rect = pygame.Rect(0, 0, saved.w, saved.h)
        try:
            self._render_static_into(s, world)
        finally:
            self.rect = saved
        return s

    def _render_static_into(self, s, world):
        sc = self.scale
        x0, y0 = self.w2s(0, 0)
        lot = pygame.Rect(int(x0), int(y0), int(world.width * sc), int(world.height * sc))
        pygame.draw.rect(s, T.ASPHALT, lot)
        if self.show_grid and sc >= 6:
            for i in range(1, int(world.width) + 1):
                px = int(x0 + i * sc)
                pygame.draw.line(s, (68, 72, 82) if i % 5 else (84, 88, 100), (px, lot.y), (px, lot.bottom))
            for j in range(1, int(world.height) + 1):
                py = int(y0 + j * sc)
                pygame.draw.line(s, (68, 72, 82) if j % 5 else (84, 88, 100), (lot.x, py), (lot.right, py))
        for sl in world.slots:
            pts = self.poly(sl.x, sl.y, sl.angle, sl.length, sl.width)
            if sl.target:
                pygame.draw.polygon(s, lerp(T.ASPHALT, T.TARGET_SLOT, 0.22), pts)
                pygame.draw.polygon(s, T.TARGET_SLOT, pts, 2)
            else:
                for a, b in ((0, 1), (1, 2), (3, 0)):
                    pygame.draw.line(s, T.SLOT_LINE, pts[a], pts[b], 2)
        for p in world.parked_cars:
            self.blit_car(s, p.x, p.y, p.angle, p.length, p.width, p.color)
        for o in world.obstacles:
            pts = self.poly(o.x, o.y, o.angle, o.length, o.width)
            pygame.draw.polygon(s, T.OBSTACLE, pts)
            pygame.draw.polygon(s, shade(T.OBSTACLE, 0.6), pts, 2)
            if o.length < 3 and o.width < 3 and sc > 8:
                pygame.draw.line(s, shade(T.OBSTACLE, 0.6), pts[0], pts[2], 1)
                pygame.draw.line(s, shade(T.OBSTACLE, 0.6), pts[1], pts[3], 1)
        pygame.draw.rect(s, T.LOT_EDGE, lot.inflate(6, 6), 4)
        if self.show_spawns:
            for sp in world.spawns:
                c = self.w2s(sp.x, sp.y)
                pygame.draw.circle(s, shade(T.SPAWN, 0.8), (int(c[0]), int(c[1])), max(4, int(sp.jitter_pos * sc)), 1)
                tip = self.w2s(sp.x + math.cos(sp.angle) * 1.6, sp.y + math.sin(sp.angle) * 1.6)
                for da in (-sp.jitter_angle, sp.jitter_angle):
                    e = self.w2s(sp.x + math.cos(sp.angle + da) * 1.3, sp.y + math.sin(sp.angle + da) * 1.3)
                    pygame.draw.line(s, shade(T.SPAWN, 0.65), c, e, 1)
                arrow(s, T.SPAWN, c, tip, 8, 3)
                pygame.draw.circle(s, T.SPAWN, (int(c[0]), int(c[1])), 4)
        for m in world.moving:
            self._draw_path(s, m)

    def _draw_path(self, s, m):
        col = T.PEDESTRIAN if m.kind == "pedestrian" else T.MOVING_CAR
        dim = shade(col, 0.8)
        pts = [self.w2s(p[0], p[1]) for p in m.path]
        loop = pts + pts[:1] if (m.closed and len(pts) > 2) else pts
        for a, b in zip(loop[:-1], loop[1:]):
            dashed_line(s, dim, a, b)
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            ang = math.atan2(b[1] - a[1], b[0] - a[0])
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 40:
                for k in (-0.45, 0.45):
                    pygame.draw.line(s, dim, (mx, my), (mx - 8 * math.cos(ang + k), my - 8 * math.sin(ang + k)), 2)
        for p in pts:
            pygame.draw.circle(s, col, (int(p[0]), int(p[1])), 4)
            pygame.draw.circle(s, (20, 20, 20), (int(p[0]), int(p[1])), 4, 1)

    def static_movers(self, world):
        rows, kinds = [], []
        for m in world.moving:
            if not m.path:
                continue
            p0 = m.path[0]
            p1 = m.path[1] if len(m.path) > 1 else p0
            l, w = m.size()
            rows.append([p0[0], p0[1], math.atan2(p1[1] - p0[1], p1[0] - p0[0]), l, w])
            kinds.append(m.kind_code())
        return np.array(rows).reshape(-1, 5), kinds

    def draw_mover(self, surf, row, kind, alpha=None):
        x, y, a, l, w = row
        if kind == PEDESTRIAN:
            c = self.w2s(x, y)
            r = max(8, int(max(l, w) * 0.5 * self.scale))
            pygame.draw.circle(surf, T.PEDESTRIAN, (int(c[0]), int(c[1])), r)
            pygame.draw.circle(surf, (60, 50, 10), (int(c[0]), int(c[1])), r, 2)
            tip = (c[0] + math.cos(a) * (r + 5), c[1] + math.sin(a) * (r + 5))
            pygame.draw.line(surf, (60, 50, 10), c, tip, 3)
        else:
            self.blit_car(surf, x, y, a, l, w, T.MOVING_CAR, alpha)

    def add_effect(self, x, y, angle, kind, color=None):
        self.effects.append({"x": x, "y": y, "a": angle, "kind": kind, "color": color or
                             (T.GOOD if kind == "ok" else T.BAD), "t0": time.time()})
        del self.effects[:-8]

    def _get_overlay(self):
        if self._overlay is None or self._overlay.get_size() != self.rect.size:
            self._overlay = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        self._overlay.fill((0, 0, 0, 0))
        return self._overlay

    def update_maps(self, env, agent, frame, now):
        if not (self.show_value or self.show_policy) or env is None or frame is None:
            self.maps = None
            return
        m = self.maps
        same = m is not None and m["world"] is env.world
        if same and now - self.maps_time < 2.0:
            return
        self.maps_time = now
        try:
            self.maps = self._compute_maps(env, agent, frame)
        except Exception:
            self.maps = None

    def _compute_maps(self, env, agent, frame):
        W, H = env.world.width, env.world.height
        nx, ny = int(math.ceil(W)), int(math.ceil(H))
        cw, ch = W / nx, H / ny
        gx, gy = np.meshgrid((np.arange(nx) + 0.5) * cw, (np.arange(ny) + 0.5) * ch)
        fx, fy = gx.ravel(), gy.ravel()
        th = np.full(fx.shape, frame["angle"])
        blocked = env.collisions(fx, fy, th).any(1)
        obs = env.observe_poses(fx, fy, th, frame["slot"])
        val = np.asarray(agent.value(obs), float)
        acts = None
        if self.show_policy:
            acts = np.asarray(agent.act(obs, deterministic=True)[0])
        ok = val[~blocked] if (~blocked).any() else val
        lo, hi = np.percentile(ok, 2), np.percentile(ok, 98)
        t = np.clip((val - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = heat_colors(t.reshape(ny, nx))
        surf = pygame.Surface((nx, ny), pygame.SRCALPHA)
        px = pygame.surfarray.pixels3d(surf)
        px[:] = rgb.transpose(1, 0, 2)
        del px
        al = pygame.surfarray.pixels_alpha(surf)
        al[:] = np.where(blocked.reshape(ny, nx), 0, 150).T
        del al
        self._heat_scaled = None
        return {"world": env.world, "nx": nx, "ny": ny, "cw": cw, "ch": ch, "fx": fx, "fy": fy, "acts": acts,
                "blocked": blocked, "surf": surf, "lo": float(lo), "hi": float(hi), "theta": float(frame["angle"])}

    def _draw_maps(self, surf, ov, world):
        m = self.maps
        if m is None or m["world"] is not world:
            return
        if self.show_value:
            size = (int(world.width * self.scale), int(world.height * self.scale))
            if self._heat_scaled is None or self._heat_scaled[0] != size:
                self._heat_scaled = (size, pygame.transform.scale(m["surf"], size))
            x0, y0 = self.w2s(0, 0)
            surf.blit(self._heat_scaled[1], (int(x0), int(y0)))
        if self.show_policy and m["acts"] is not None and m["cw"] * self.scale >= 9:
            L = 0.36 * m["cw"] * self.scale
            for i in range(len(m["fx"])):
                if m["blocked"][i]:
                    continue
                a = int(m["acts"][i])
                steer, thr = a % 3 - 1, a // 3 - 1
                c = self.w2s(m["fx"][i], m["fy"][i])
                c = (c[0] - self.rect.x, c[1] - self.rect.y)
                if thr == 0:
                    pygame.draw.circle(ov, (230, 230, 230, 200), (int(c[0]), int(c[1])), 2)
                    continue
                ang = m["theta"] + (steer * 0.6 if thr > 0 else math.pi - steer * 0.6)
                col = (255, 255, 255, 230) if thr > 0 else (255, 170, 70, 230)
                d = (math.cos(ang) * L, math.sin(ang) * L)
                arrow(ov, col, (c[0] - d[0] * 0.5, c[1] - d[1] * 0.5), (c[0] + d[0] * 0.5, c[1] + d[1] * 0.5), 4, 1)

    def draw_legend(self, surf):
        m = self.maps
        if m is None or not self.show_value:
            return
        box = pygame.Rect(self.rect.right - 190, self.rect.y + 8, 180, 46)
        pygame.draw.rect(surf, T.PANEL, box, border_radius=5)
        pygame.draw.rect(surf, T.BORDER, box, 1, border_radius=5)
        T.text(surf, "Value map (heading %d deg)" % round(math.degrees(m["theta"])), (box.x + 8, box.y + 5), 12,
               T.TEXT_DIM)
        bar = pygame.Rect(box.x + 8, box.y + 22, box.w - 16, 8)
        cols = heat_colors(np.linspace(0, 1, bar.w))
        for i in range(bar.w):
            pygame.draw.line(surf, tuple(int(v) for v in cols[i]), (bar.x + i, bar.y), (bar.x + i, bar.bottom - 1))
        T.text(surf, "%.1f" % m["lo"], (bar.x, bar.bottom + 3), 12, T.TEXT_DIM)
        T.text(surf, "%.1f" % m["hi"], (bar.right, bar.bottom + 3), 12, T.TEXT_DIM, anchor="topright")

    def draw(self, surf, world, frame=None, rev=0, trail=None, env=None, agent=None, ghosts=None, outcome=None):
        clip = surf.get_clip()
        surf.set_clip(self.rect)
        now = time.time()
        self._draw_static(surf, world, rev)
        ov = self._get_overlay()
        self.update_maps(env, agent, frame, now)
        self._draw_maps(surf, ov, world)
        if frame is not None:
            self._draw_goal(surf, ov, frame["slot"], now, outcome)
        if ghosts is not None and self.show_ghosts:
            for i in range(1, ghosts.n):
                self.blit_car(surf, ghosts.x[i], ghosts.y[i], ghosts.th[i], 4.5, 1.9, T.AGENT_CAR, 70)
        if frame is not None and len(frame["moving"]):
            for row, k in zip(frame["moving"], frame["moving_kinds"]):
                self.draw_mover(surf, row, k)
        elif frame is None:
            rows, kinds = self.static_movers(world)
            for row, k in zip(rows, kinds):
                self.draw_mover(surf, row, k)
        if frame is not None:
            if self.show_trail and trail and len(trail) > 1:
                n = len(trail)
                pts = [self.w2s(p[0], p[1]) for p in trail]
                for i in range(1, n):
                    a = int(40 + 190 * i / n)
                    p0 = (pts[i - 1][0] - self.rect.x, pts[i - 1][1] - self.rect.y)
                    p1 = (pts[i][0] - self.rect.x, pts[i][1] - self.rect.y)
                    pygame.draw.line(ov, T.AGENT_CAR + (a,), p0, p1, 3)
            if self.show_rays:
                c = self.w2s(frame["x"], frame["y"])
                c = (c[0] - self.rect.x, c[1] - self.rect.y)
                for end, mv in zip(frame["ray_ends"], frame["ray_moving"]):
                    e = self.w2s(end[0], end[1])
                    e = (e[0] - self.rect.x, e[1] - self.rect.y)
                    col = (T.RAY_MOVING if mv else T.RAY) + (150 if mv else 110,)
                    pygame.draw.line(ov, col, c, e, 1)
                    pygame.draw.circle(ov, col[:3] + (230,), (int(e[0]), int(e[1])), 2)
        surf.blit(ov, self.rect.topleft)
        if frame is not None:
            self._draw_agent(surf, frame, now, outcome)
        self._draw_effects(surf, now)
        surf.set_clip(clip)
        self.draw_legend(surf)

    def _draw_goal(self, surf, ov, slot, now, outcome):
        x, y, a, l, w = [float(v) for v in slot]
        pts = self.poly(x, y, a, l, w)
        local = [(p[0] - self.rect.x, p[1] - self.rect.y) for p in pts]
        pulse = 0.5 + 0.5 * math.sin(now * 3.0)
        alpha = 200 if outcome == "success" else int(55 + 55 * pulse)
        pygame.draw.polygon(ov, T.TARGET_SLOT + (alpha,), local)
        pygame.draw.polygon(surf, T.TARGET_SLOT if outcome != "success" else (150, 255, 180), pts, 4)
        c = self.w2s(x, y)
        d = min(l * 0.3, 1.6)
        arrow(surf, (200, 255, 215), self.w2s(x - math.cos(a) * d, y - math.sin(a) * d),
              self.w2s(x + math.cos(a) * d, y + math.sin(a) * d), 9, 2)
        T.text(surf, "GOAL", (c[0], c[1] + 14), 12, (210, 255, 225), bold=True,
               anchor="center")

    def _draw_agent(self, surf, frame, now, outcome):
        x, y, a, steer = frame["x"], frame["y"], frame["angle"], frame["steer"]
        hit = frame["collision"].any() if "collision" in frame else False
        L, W = 4.5, 1.9
        wheel_col = (18, 18, 22)
        for sx in (-1, 1):
            for fx, ang in ((1.35, a + steer), (-1.35, a)):
                c = geo.to_world(np.array([fx, sx * (W / 2 - 0.02)]), x, y, a)
                surf_pts = self.poly(c[0], c[1], ang, 0.75, 0.3)
                pygame.draw.polygon(surf, wheel_col, surf_pts)
        self.blit_car(surf, x, y, a, L, W, T.AGENT_CAR)
        pts = self.poly(x, y, a, L, W)
        if hit:
            pygame.draw.polygon(surf, T.BAD, pts, 3)
        elif outcome == "success":
            pygame.draw.polygon(surf, (150, 255, 180), pts, 3)
        else:
            pygame.draw.polygon(surf, shade(T.AGENT_CAR, 0.5), pts, 1)

    def _draw_effects(self, surf, now):
        keep = []
        for e in self.effects:
            age = now - e["t0"]
            if age > 1.0:
                continue
            keep.append(e)
            f = 1 - age
            c = self.w2s(e["x"], e["y"])
            col = lerp(T.ASPHALT, e["color"], f)
            r = int((1.2 + age * (5 if e["kind"] == "hit" else 4)) * self.scale)
            pygame.draw.circle(surf, col, (int(c[0]), int(c[1])), r, max(1, int(4 * f)))
            pygame.draw.polygon(surf, col, self.poly(e["x"], e["y"], e["a"], 4.5, 1.9), 3)
        self.effects = keep

    def draw_hud(self, surf, frame, lines=None):
        if frame is None:
            return
        act = frame.get("action", -1)
        parts = [
            "speed %+.1f m/s" % frame["v"], "steer %+.2f" % frame["steer"], "inside %d%%" % round(frame["inside"] * 100),
            "angle err %.1f deg" % frame["angle_err"], "return %+.2f" % frame.get("return", 0.0),
            "action %s" % (ACTION_NAMES[act] if 0 <= act < len(ACTION_NAMES) else "-"),
        ]
        if lines:
            parts = lines + parts
        s = "   ".join(parts)
        img = T.font(14).render(s, True, T.TEXT)
        box = img.get_rect(bottomleft=(self.rect.x + 10, self.rect.bottom - 8)).inflate(16, 10)
        bg = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(bg, (18, 20, 26, 200), bg.get_rect(), border_radius=6)
        surf.blit(bg, box.topleft)
        surf.blit(img, img.get_rect(center=box.center))

    def banner(self, surf, text, color, sub=None):
        img = T.font(30, True).render(text, True, (12, 16, 24))
        simg = T.font(14).render(sub, True, (12, 16, 24)) if sub else None
        w = max(img.get_width(), simg.get_width() if simg else 0) + 44
        h = img.get_height() + (simg.get_height() + 6 if simg else 0) + 20
        box = pygame.Rect(0, 0, w, h)
        box.midtop = (self.rect.centerx, self.rect.y + 12)
        pygame.draw.rect(surf, color, box, border_radius=10)
        surf.blit(img, img.get_rect(midtop=(box.centerx, box.y + 10)))
        if simg:
            surf.blit(simg, simg.get_rect(midbottom=(box.centerx, box.bottom - 9)))


def outcome_of(summary):
    if summary["success"]:
        return "success", "Parked!", T.GOOD, "inside %d%%, angle error %.1f deg" % (round(summary["inside"] * 100),
                                                                                  summary["angle_err"])
    if summary["collision"]:
        k = summary["collision"]
        return "collision", "Hit " + k.replace("_", " "), T.BAD, "inside %d%%" % round(summary["inside"] * 100)
    return "timeout", "Timeout", T.WARN, "inside %d%%, angle error %.1f deg" % (round(summary["inside"] * 100),
                                                                              summary["angle_err"])
