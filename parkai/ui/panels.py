import math
import time

import numpy as np
import pygame

from ..env import ACTION_NAMES, COMPONENT_NAMES, SimConfig
from ..world import KIND_NAMES
from . import theme
from .theme import text
from .widgets import (LineChart, NetworkView, ProbabilityGrid, RadarDial, SignedBarList, Sparkline,
                      StatRows, clamp, empty_state, fmt, mix, section_header)

_SIM = SimConfig()
_OUT_LABELS = [n.replace("forward", "fwd").replace("reverse", "rev").replace("brake", "brk")
               .replace("+left", " L").replace("+straight", " S").replace("+right", " R") for n in ACTION_NAMES]
_KIND_COLORS = dict(theme.KIND_COLORS)
_KIND_COLORS["wall"] = (170, 176, 195)
_SECTION_GAP = 4
_HEADER = 17


class InspectorPanel:
    SECTIONS = [
        ("status", 60, 0.0),
        ("actions", 14 + 3 * 22, 1.0),
        ("value", 26, 0.4),
        ("rewards", 14 + 6 * 12, 0.5),
        ("sensors", 96, 1.0),
        ("network", 120, 2.0),
    ]

    def __init__(self, obs_names, n_rays):
        self.n_rays = n_rays
        names = list(obs_names)
        self.i_ray = names.index("ray0") if "ray0" in names else 7
        self.i_rmov = names.index("ray0_moving") if "ray0_moving" in names else self.i_ray + n_rays
        self.i_goal = names.index("goal_fwd") if "goal_fwd" in names else 0
        self.stats = StatRows(2)
        self.grid = ProbabilityGrid()
        self.spark = Sparkline()
        self.bars = SignedBarList()
        self.radar = RadarDial()
        self.net = NetworkView(32, _OUT_LABELS)
        self._cache_key = None
        self._values = None
        self._cum = None

    def _episode_cache(self, frames):
        if not frames:
            return None, None
        key = (id(frames), len(frames))
        if key != self._cache_key:
            self._values = np.array([f.get("value", 0.0) for f in frames], dtype=np.float64)
            self._cum = np.sum([f["components"] for f in frames], axis=0) if "components" in frames[0] else None
            self._cache_key = key
        return self._values, self._cum

    def draw(self, surface, rect, frame, inspect=None, episode_frames=None):
        pygame.draw.rect(surface, theme.PANEL, rect, border_radius=6)
        pygame.draw.rect(surface, theme.BORDER, rect, 1, border_radius=6)
        inner = rect.inflate(-20, -16)
        if frame is None:
            text(surface, "AI INSPECTOR", (inner.x, inner.y), 11, theme.TEXT_DIM, bold=True)
            empty_state(surface, rect, "no data yet - start training or watch the AI")
            return
        chosen = frame.get("action", -1)
        chosen = chosen if chosen is not None and chosen >= 0 else None
        probs = None
        if frame.get("probs") is not None:
            probs = np.asarray(frame["probs"]).ravel()
        elif inspect is not None and inspect.get("probs") is not None:
            probs = np.asarray(inspect["probs"]).ravel()
        value = frame.get("value")
        if value is None or (frame.get("terminal") and inspect and inspect.get("value") is not None):
            value = inspect.get("value", 0.0) if inspect else 0.0
        values, cum = self._episode_cache(episode_frames)

        avail = inner.h
        chosen_secs = []
        used = 0
        for name, min_body, w in self.SECTIONS:
            need = _HEADER + min_body + (_SECTION_GAP if chosen_secs else 0)
            if used + need <= avail:
                chosen_secs.append([name, min_body, w])
                used += need
        extra = avail - used
        tw = sum(s[2] for s in chosen_secs)
        if tw > 0 and extra > 0:
            for s in chosen_secs:
                s[1] += int(extra * s[2] / tw)

        y = inner.y
        for name, body_h, _ in chosen_secs:
            sec = pygame.Rect(inner.x, y, inner.w, _HEADER + body_h)
            y = sec.bottom + _SECTION_GAP
            getattr(self, "_draw_" + name)(surface, sec, frame, probs, chosen, value, values, cum, inspect)

    def _draw_status(self, surface, sec, frame, probs, chosen, value, values, cum, inspect):
        done = frame.get("terminal")
        body = section_header(surface, sec, "status", "terminal" if done else "step %d" % frame.get("step", 0),
                              theme.WARN if done else theme.TEXT)
        err = float(frame.get("angle_err", 0.0))
        straight = clamp(1 - err / _SIM.straight_tolerance_deg, 0, 1)
        inside = float(frame.get("inside", 0.0))
        rew = float(frame.get("reward", 0.0))
        ret = float(frame.get("return", 0.0))
        items = [
            ("speed", "%+.2f m/s" % frame.get("v", 0.0), None),
            ("steer", "%+.0f°" % math.degrees(frame.get("steer", 0.0)), None),
            ("inside", "%d%%" % round(inside * 100), theme.GOOD if inside > 0.97 else None),
            ("straight", "%d%%" % round(straight * 100), theme.GOOD if straight > 0.9 else None),
            ("angle err", "%.0f°" % err, None),
            ("time", "%.1f s" % frame.get("t", 0.0), None),
            ("return", "%+.2f" % ret, theme.GOOD if ret > 0 else theme.BAD if ret < 0 else None),
            ("reward", "%+.3f" % rew, theme.GOOD if rew > 0 else theme.BAD if rew < 0 else None),
        ]
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        self.stats.draw(surface, body, [items[i] for i in order], row_h=max(14, min(18, body.h // 4)))

    def _draw_actions(self, surface, sec, frame, probs, chosen, value, values, cum, inspect):
        right = None
        if chosen is not None:
            right = "chose " + ACTION_NAMES[chosen]
        body = section_header(surface, sec, "action probabilities", right, theme.WARN)
        self.grid.draw(surface, body, probs, chosen)

    def _draw_value(self, surface, sec, frame, probs, chosen, value, values, cum, inspect):
        body = section_header(surface, sec, "value estimate V(s)", "%+.3f" % float(value), theme.ACCENT)
        self.spark.draw(surface, body, values if values is not None and len(values) > 1 else None)

    def _draw_rewards(self, surface, sec, frame, probs, chosen, value, values, cum, inspect):
        tot = float(frame.get("reward", 0.0))
        body = section_header(surface, sec, "reward breakdown", "step %+.3f" % tot,
                              theme.GOOD if tot > 0 else theme.BAD if tot < 0 else theme.TEXT_DIM)
        comp = np.asarray(frame.get("components", np.zeros(len(COMPONENT_NAMES))), dtype=np.float64)
        c = cum if cum is not None else comp
        rows = [(COMPONENT_NAMES[i], float(comp[i]), float(c[i])) for i in range(len(COMPONENT_NAMES))]
        self.bars.draw(surface, body, rows)

    def _draw_sensors(self, surface, sec, frame, probs, chosen, value, values, cum, inspect):
        obs = np.asarray(frame["obs"]) if frame.get("obs") is not None else None
        body = section_header(surface, sec, "sensors", "%d rays" % self.n_rays)
        if obs is None:
            return
        n = self.n_rays
        clear = obs[self.i_ray:self.i_ray + n]
        moving = frame.get("ray_moving")
        if moving is None:
            moving = obs[self.i_rmov:self.i_rmov + n] > 0.5
        gf, gs = float(obs[self.i_goal]) * 10, float(obs[self.i_goal + 1]) * 10
        side = min(body.h, body.w * 0.55)
        dial = pygame.Rect(body.x, body.y, int(side), body.h)
        self.radar.draw(surface, dial, clear, moving, (gf, gs), _SIM.ray_range)
        x = dial.right + 10
        w = body.right - x
        if w < 70:
            return
        dist = math.hypot(gf, gs)
        bearing = math.degrees(math.atan2(gs, gf))
        nearest = float(np.min(clear)) * _SIM.ray_range
        n_mov = int(np.sum(moving))
        rows = [
            ("goal dist", "%.1f m" % dist, theme.GOOD),
            ("goal bearing", "%+.0f°" % bearing, theme.GOOD),
            ("nearest hit", "%.1f m" % nearest if nearest < _SIM.ray_range * 0.999 else "clear",
             theme.BAD if nearest < 1.0 else theme.TEXT),
            ("moving hits", "%d" % n_mov, theme.RAY_MOVING if n_mov else theme.TEXT_DIM),
        ]
        yy = body.y + 4
        for lab, val, col in rows:
            text(surface, lab, (x, yy), 12, theme.TEXT_DIM)
            text(surface, val, (body.right, yy - 1), 14, col, bold=True, anchor="topright")
            yy += 19 if body.h >= 130 else 17
        yy += 4
        for col, lab in ((theme.RAY, "static clearance"), (theme.RAY_MOVING, "moving object"), (theme.GOOD, "goal direction")):
            if yy + 14 > body.bottom:
                break
            pygame.draw.circle(surface, col, (x + 4, yy + 6), 4)
            text(surface, lab, (x + 14, yy), 11, theme.TEXT_DIM)
            yy += 16

    def _draw_network(self, surface, sec, frame, probs, chosen, value, values, cum, inspect):
        layers = inspect.get("layers") if inspect else None
        body = section_header(surface, sec, "network activations", "blue −   orange +", theme.TEXT_DIM)
        if not layers:
            empty_state(surface, body, "agent exposes no activations")
            return
        hl = chosen
        if hl is None and probs is not None:
            hl = int(np.argmax(probs))
        self.net.draw(surface, body, layers, hl)


def _roll_at(col, w, idx):
    c = col.c
    hi = idx + 1
    lo = np.maximum(0, hi - w)
    return (c[hi] - c[lo]) / (hi - lo)


def _sample(n, m=800):
    if n <= m:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, m).astype(np.int64))


class _Col:
    def __init__(self, dtype=np.float64):
        self.a = np.zeros(1024, dtype=dtype)
        self.c = np.zeros(1025)
        self.n = 0

    def extend(self, vals):
        k = len(vals)
        if self.n + k > len(self.a):
            grow = max(len(self.a), k)
            self.a = np.concatenate([self.a, np.zeros(grow, dtype=self.a.dtype)])
            self.c = np.concatenate([self.c, np.zeros(grow)])
        vals = np.nan_to_num(np.asarray(vals, dtype=np.float64))
        self.a[self.n:self.n + k] = vals
        self.c[self.n + 1:self.n + k + 1] = self.c[self.n] + np.cumsum(vals)
        self.n += k

    @property
    def v(self):
        return self.a[:self.n]


class ChartsPanel:
    TABS = ["Rewards", "Success", "Parking quality", "Learning", "Curriculum"]
    WINDOW = 100
    REFRESH = 0.25

    def __init__(self):
        self.tab = 0
        self._tab_rects = []
        self._tab_local = []
        self._ep = {k: _Col() for k in ("ret", "len", "succ", "coll", "tout", "inside", "err", "kind")}
        self._ep["comp"] = [_Col() for _ in COMPONENT_NAMES]
        self._ep["kinds"] = [_Col() for _ in KIND_NAMES]
        self._ep_n = 0
        self._up = {}
        self._up_n = 0
        self._ep_first = None
        self._up_first = None
        self._data_ver = (0, 0)
        self._last_refresh = 0.0
        self._img = None
        self._img_key = None
        self.last_draw_ms = 0.0
        self._charts = {}

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, r in enumerate(self._tab_rects):
                if r.collidepoint(event.pos):
                    self.tab = i
                    return True
        return False

    def _ingest(self, history):
        eps = history.get("episodes", [])
        ups = history.get("updates", [])
        if (len(eps) < self._ep_n) or (len(ups) < self._up_n) or \
                (eps and self._ep_first is not eps[0]) or (ups and self._up_first is not ups[0]):
            tab = self.tab
            self.__init__()
            self.tab = tab
        if eps:
            self._ep_first = eps[0]
        if ups:
            self._up_first = ups[0]
        if len(eps) > self._ep_n:
            new = eps[self._ep_n:]
            E = self._ep
            E["ret"].extend([e["return"] for e in new])
            E["len"].extend([e["length"] for e in new])
            E["succ"].extend([1.0 if e["success"] else 0.0 for e in new])
            E["coll"].extend([1.0 if e.get("collision") else 0.0 for e in new])
            E["tout"].extend([1.0 if e.get("timeout") else 0.0 for e in new])
            E["inside"].extend([e["inside"] for e in new])
            E["err"].extend([e["angle_err"] for e in new])
            for j, name in enumerate(COMPONENT_NAMES):
                E["comp"][j].extend([e["components"].get(name, 0.0) for e in new])
            for j, name in enumerate(KIND_NAMES):
                E["kinds"][j].extend([1.0 if e.get("collision") == name else 0.0 for e in new])
            self._ep_n = len(eps)
        if len(ups) > self._up_n:
            new = ups[self._up_n:]
            keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "explained_var", "difficulty", "sps", "update")
            for k in keys:
                col = self._up.setdefault(k, _Col())
                col.extend([float(u.get(k, np.nan)) for u in new])
            self._up_n = len(ups)

    def _ep_series(self, key, window=None, kind="line", color=theme.ACCENT, name="", scale=1.0, width=2):
        col = self._ep[key] if isinstance(key, str) else key
        arr = col.v
        idx = _sample(len(arr))
        if kind == "points":
            return {"name": name, "x": idx + 1, "y": arr[idx] * scale, "color": color, "kind": "points", "last": None}
        idx = idx[idx >= min(10, len(arr) - 1)]
        sm = _roll_at(col, window or self.WINDOW, idx)
        last = float(_roll_at(col, window or self.WINDOW, np.array([len(arr) - 1]))[0] * scale) if len(arr) else None
        return {"name": name, "x": idx + 1, "y": sm * scale, "color": color, "width": width, "last": last}

    def _up_series(self, key, color, name="", smooth=0, width=2, faint=False):
        col = self._up.get(key)
        arr = col.v if col is not None else np.zeros(0)
        idx = _sample(len(arr))
        x = idx + 1
        y = _roll_at(col, smooth, idx) if smooth else arr[idx]
        if faint:
            color = mix(theme.PANEL, color, 0.45)
        return {"name": name, "x": x, "y": y, "color": color, "width": width,
                "last": float(arr[-1]) if len(arr) else None}

    def _build_tab(self, tab):
        W = self.WINDOW
        C = theme.SERIES
        pct = lambda v: "%.0f%%" % (v * 100)
        if tab == 0:
            ret = [self._ep_series("ret", kind="points", color=mix(theme.PANEL, C[0], 0.5)),
                   self._ep_series("ret", W, color=theme.WARN, name="mean%d" % W)]
            comps = [self._ep_series(self._ep["comp"][j], W, color=C[j], name=COMPONENT_NAMES[j][:4], width=2)
                     for j in range(len(COMPONENT_NAMES))]
            ln = [self._ep_series("len", W, color=C[4], name="steps")]
            return [(LineChart("Episode return", xlabel="episode"), ret, 1.5),
                    (LineChart("Return components (mean %d)" % W, xlabel="episode"), comps, 1.5),
                    (LineChart("Episode length", xlabel="episode"), ln, 1.0)]
        if tab == 1:
            outcome = [self._ep_series("succ", W, color=theme.GOOD, name="success"),
                       self._ep_series("coll", W, color=theme.BAD, name="collision"),
                       self._ep_series("tout", W, color=theme.WARN, name="timeout")]
            kinds = [self._ep_series(self._ep["kinds"][j], W, color=_KIND_COLORS[KIND_NAMES[j]],
                                     name=KIND_NAMES[j].replace("_car", "")[:6]) for j in range(len(KIND_NAMES))]
            return [(LineChart("Outcome rates (rolling %d)" % W, ylim=(0, 1), y_fmt=pct), outcome, 1.2),
                    (LineChart("Collision rate by kind", ylim=(0, 1), y_fmt=pct), kinds, 1.2)]
        if tab == 2:
            ins = [self._ep_series("inside", kind="points", color=mix(theme.PANEL, C[1], 0.45)),
                   self._ep_series("inside", W, color=theme.GOOD, name="mean%d" % W)]
            err = [self._ep_series("err", kind="points", color=mix(theme.PANEL, C[2], 0.45)),
                   self._ep_series("err", W, color=theme.WARN, name="mean%d" % W)]
            return [(LineChart("Final inside slot", ylim=(0, 1), y_fmt=pct), ins, 1.0),
                    (LineChart("Final angle error (deg)"), err, 1.0)]
        if tab == 3:
            loss = [self._up_series("policy_loss", C[0], faint=True, width=1),
                    self._up_series("policy_loss", C[0], "policy", smooth=10),
                    self._up_series("value_loss", C[3], faint=True, width=1),
                    self._up_series("value_loss", C[3], "value", smooth=10)]
            ent = [self._up_series("entropy", C[2], faint=True, width=1),
                   self._up_series("entropy", C[2], "entropy", smooth=10)]
            kl = [self._up_series("approx_kl", C[4], faint=True, width=1),
                  self._up_series("approx_kl", C[4], "kl", smooth=10)]
            ev = [self._up_series("explained_var", C[5], faint=True, width=1),
                  self._up_series("explained_var", C[5], "expl.var", smooth=10)]
            return [(LineChart("Losses", xlabel="update"), loss, 1.0),
                    (LineChart("Policy entropy", xlabel="update"), ent, 1.0),
                    (LineChart("Approx KL", xlabel="update"), kl, 1.0),
                    (LineChart("Explained variance", xlabel="update"), ev, 1.0)]
        diff = [self._up_series("difficulty", C[4], "difficulty")]
        sps = [self._up_series("sps", C[0], faint=True, width=1), self._up_series("sps", C[0], "steps/s", smooth=10)]
        return [(LineChart("Curriculum difficulty", ylim=(0, 1.05), xlabel="update"), diff, 1.0),
                (LineChart("Throughput (steps/sec)", xlabel="update"), sps, 1.0)]

    def draw(self, surface, rect, history):
        t0 = time.perf_counter()
        now = t0
        ver = (len(history.get("episodes", [])), len(history.get("updates", [])))
        if ver != self._data_ver and (now - self._last_refresh > self.REFRESH or self._img_key is None):
            self._ingest(history)
            self._data_ver = (self._ep_n, self._up_n)
            self._last_refresh = now
        key = (self.tab, rect.size, self._data_ver)
        if key != self._img_key:
            self._render(rect.size)
            self._img_key = key
        surface.blit(self._img, rect.topleft)
        self._tab_rects = [r.move(rect.x, rect.y) for r in self._tab_local]
        self.last_draw_ms = (time.perf_counter() - t0) * 1000

    def _render(self, size):
        if self._img is None or self._img.get_size() != size:
            self._img = pygame.Surface(size).convert()
        img = self._img
        img.fill(theme.PANEL)
        pygame.draw.rect(img, theme.BORDER, img.get_rect(), 1, border_radius=6)
        x = 10
        self._tab_local = []
        for i, name in enumerate(self.TABS):
            tw = theme.font(13, True).size(name)[0] + 22
            r = pygame.Rect(x, 6, tw, 22)
            self._tab_local.append(r)
            on = i == self.tab
            pygame.draw.rect(img, theme.ACCENT if on else theme.PANEL_LIGHT, r, border_radius=11)
            text(img, name, r.center, 13, (12, 16, 24) if on else theme.TEXT_DIM, bold=on, anchor="center")
            x = r.right + 6
        info = "%d episodes, %d updates" % (self._ep_n, self._up_n)
        text(img, info, (size[0] - 12, 17), 12, theme.TEXT_DIM, anchor="midright")
        charts = self._build_tab(self.tab)
        area = pygame.Rect(8, 34, size[0] - 16, size[1] - 34 - 8)
        gap = 8
        tot = sum(c[2] for c in charts)
        usable = area.w - gap * (len(charts) - 1)
        cx = area.x
        for chart, series, wgt in charts:
            w = int(usable * wgt / tot)
            chart.draw(img, pygame.Rect(cx, area.y, w, area.h), series)
            cx += w + gap
