import math

import numpy as np
import pygame

from . import theme
from .theme import text


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def mix(a, b, t):
    t = clamp(t, 0.0, 1.0)
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    a = abs(v)
    if a >= 1e5:
        return "%.2e" % v
    if a >= 1000:
        return "%d" % round(v)
    if a >= 100:
        return "%.0f" % v
    if a >= 10:
        return "%.1f" % v
    if a >= 1 or a == 0:
        return "%.2f" % v if digits >= 2 else "%.1f" % v
    if a >= 0.001:
        return "%.4f" % v
    return "%.1e" % v


def fmt_tick(v, step):
    if abs(v) >= 1e6:
        return "%.1fM" % (v / 1e6)
    if abs(v) >= 1e4:
        return "%.0fk" % (v / 1e3) if step >= 1000 else "%.1fk" % (v / 1e3)
    if step >= 1:
        return "%d" % round(v)
    dec = max(0, int(math.ceil(-math.log10(step))))
    return "%.*f" % (dec, v)


def nice_ticks(lo, hi, n=5):
    if hi - lo < 1e-12:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    v = first
    while v <= hi + step * 1e-6:
        ticks.append(v)
        v += step
    return ticks, step


def section_header(surface, rect, title, right=None, right_color=theme.TEXT_DIM):
    text(surface, title.upper(), (rect.x, rect.y), 11, theme.TEXT_DIM, bold=True)
    if right is not None:
        text(surface, right, (rect.right, rect.y), 12, right_color, anchor="topright")
    y = rect.y + 14
    pygame.draw.line(surface, theme.BORDER, (rect.x, y), (rect.right, y))
    return pygame.Rect(rect.x, y + 3, rect.w, max(0, rect.h - 17))


def empty_state(surface, rect, msg):
    text(surface, msg, rect.center, 13, theme.TEXT_DIM, anchor="center")


class StatRows:
    def __init__(self, columns=2):
        self.columns = columns

    def draw(self, surface, rect, items, row_h=18):
        n = len(items)
        per = int(math.ceil(n / float(self.columns)))
        col_w = rect.w // self.columns
        for i, (label, value, color) in enumerate(items):
            c, r = divmod(i, per)
            x = rect.x + c * col_w
            y = rect.y + r * row_h
            text(surface, label, (x, y), 12, theme.TEXT_DIM)
            text(surface, value, (x + col_w - 10, y - 1), 14, color or theme.TEXT, bold=True, anchor="topright")


class Sparkline:
    def draw(self, surface, rect, values, color=theme.ACCENT, zero_line=True):
        pygame.draw.rect(surface, theme.PANEL_LIGHT, rect, border_radius=3)
        if values is None or len(values) < 2:
            text(surface, "collecting...", rect.center, 12, theme.TEXT_DIM, anchor="center")
            return
        v = np.asarray(values, dtype=np.float64)
        if len(v) > rect.w:
            idx = np.linspace(0, len(v) - 1, rect.w).astype(int)
            v = v[idx]
        lo, hi = float(v.min()), float(v.max())
        if zero_line:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        pad = 4
        w, h = rect.w - 2 * pad, rect.h - 2 * pad
        xs = rect.x + pad + np.linspace(0, w, len(v))
        ys = rect.y + pad + (1 - (v - lo) / (hi - lo)) * h
        if zero_line and lo < 0 < hi:
            zy = int(rect.y + pad + (1 - (0 - lo) / (hi - lo)) * h)
            pygame.draw.line(surface, theme.BORDER, (rect.x + 2, zy), (rect.right - 3, zy))
        pygame.draw.lines(surface, color, False, list(zip(xs.tolist(), ys.tolist())), 2)
        pygame.draw.circle(surface, theme.TEXT, (int(xs[-1]), int(ys[-1])), 3)
        text(surface, fmt(hi), (rect.x + 4, rect.y + 1), 10, theme.TEXT_DIM)
        text(surface, fmt(lo), (rect.x + 4, rect.bottom - 2), 10, theme.TEXT_DIM, anchor="bottomleft")


class SignedBarList:
    def draw(self, surface, rect, rows, headers=("step", "episode")):
        n = len(rows)
        if n == 0:
            return
        label_w = 64
        val_w = 46
        gap = 8
        zone_w = (rect.w - label_w - 2 * val_w - gap) // 2
        header_h = 14
        row_h = clamp((rect.h - header_h) // n, 12, 20)
        x1 = rect.x + label_w
        x2 = x1 + zone_w + val_w + gap
        text(surface, headers[0], (x1 + zone_w // 2, rect.y), 11, theme.TEXT_DIM, anchor="midtop")
        text(surface, headers[1], (x2 + zone_w // 2, rect.y), 11, theme.TEXT_DIM, anchor="midtop")
        s1 = max([abs(r[1]) for r in rows] + [1e-6])
        s2 = max([abs(r[2]) for r in rows] + [1e-6])
        y = rect.y + header_h
        for label, a, b in rows:
            text(surface, label, (rect.x, y + row_h // 2), 11, theme.TEXT, anchor="midleft")
            for x, val, sc in ((x1, a, s1), (x2, b, s2)):
                mid = x + zone_w // 2
                half = zone_w // 2 - 1
                bh = max(4, row_h - 7)
                by = y + (row_h - bh) // 2
                pygame.draw.rect(surface, theme.PANEL_LIGHT, (x, by, zone_w, bh), border_radius=2)
                L = int(abs(val) / sc * half)
                if abs(val) > 1e-9:
                    L = max(L, 1)
                    col = theme.GOOD if val > 0 else theme.BAD
                    bx = mid if val > 0 else mid - L
                    pygame.draw.rect(surface, col, (bx, by, L, bh))
                pygame.draw.line(surface, theme.TEXT_DIM, (mid, by - 1), (mid, by + bh))
                col = theme.TEXT if abs(val) > 1e-9 else theme.TEXT_DIM
                text(surface, "%+.3f" % val if abs(val) < 100 else "%+.0f" % val,
                     (x + zone_w + val_w, y + row_h // 2), 11, col, anchor="midright")
            y += row_h


class ProbabilityGrid:
    def __init__(self, steer_names=None, throttle_names=None):
        self.steer_names = steer_names or ["left", "straight", "right"]
        self.throttle_names = throttle_names or ["reverse", "brake", "forward"]

    def draw(self, surface, rect, probs, chosen=None):
        label_w = 52
        head_h = 14
        cw = (rect.w - label_w) // 3
        ch = max(10, (rect.h - head_h) // 3)
        for c in range(3):
            text(surface, self.steer_names[c], (rect.x + label_w + c * cw + cw // 2, rect.y), 11,
                 theme.TEXT_DIM, anchor="midtop")
        pmax = float(max(probs)) if probs is not None else 1.0
        for r in range(3):
            thr = 2 - r
            y = rect.y + head_h + r * ch
            text(surface, self.throttle_names[thr], (rect.x, y + ch // 2), 12, theme.TEXT_DIM, anchor="midleft")
            for c in range(3):
                a = thr * 3 + c
                p = float(probs[a]) if probs is not None else 0.0
                cell = pygame.Rect(rect.x + label_w + c * cw + 2, y + 2, cw - 4, ch - 4)
                shade = mix(theme.PANEL_LIGHT, theme.ACCENT, (p / max(pmax, 1e-9)) * 0.55 + p * 0.45)
                pygame.draw.rect(surface, shade, cell, border_radius=4)
                lum = shade[0] * 0.3 + shade[1] * 0.59 + shade[2] * 0.11
                tc = (15, 18, 24) if lum > 125 else theme.TEXT
                text(surface, "%d%%" % round(p * 100), cell.center, 14, tc, bold=True, anchor="center")
                if chosen is not None and a == chosen:
                    pygame.draw.rect(surface, theme.WARN, cell, 2, border_radius=4)


class RadarDial:
    def __init__(self):
        self._overlay = None

    def draw(self, surface, rect, clearance, moving=None, goal=None, ray_range_m=None):
        cx, cy = rect.center
        R = min(rect.w, rect.h) // 2 - 9
        if R < 20:
            return
        pygame.draw.circle(surface, theme.PANEL_LIGHT, (cx, cy), R)
        for f in (1 / 3.0, 2 / 3.0):
            pygame.draw.circle(surface, theme.BORDER, (cx, cy), int(R * f), 1)
        pygame.draw.circle(surface, theme.BORDER, (cx, cy), R, 1)
        pygame.draw.line(surface, theme.BORDER, (cx, cy - R), (cx, cy + R))
        pygame.draw.line(surface, theme.BORDER, (cx - R, cy), (cx + R, cy))
        text(surface, "F", (cx, cy - R - 2), 11, theme.TEXT_DIM, anchor="midbottom")
        if ray_range_m:
            text(surface, "%dm" % round(ray_range_m), (rect.right - 2, rect.bottom - 1), 10, theme.TEXT_DIM, anchor="bottomright")
        car_w, car_l = max(6, int(R * 0.2)), max(10, int(R * 0.36))
        r0 = car_l * 0.5
        if clearance is not None and len(clearance) > 0:
            n = len(clearance)
            pts = []
            for i in range(n):
                th = -math.pi + i * 2 * math.pi / n
                rr = r0 + clamp(float(clearance[i]), 0, 1) * (R - r0)
                pts.append((cx + math.sin(th) * rr, cy - math.cos(th) * rr))
            size = (rect.w, rect.h)
            if self._overlay is None or self._overlay.get_size() != size:
                self._overlay = pygame.Surface(size, pygame.SRCALPHA)
            self._overlay.fill((0, 0, 0, 0))
            local = [(x - rect.x, y - rect.y) for x, y in pts]
            pygame.draw.polygon(self._overlay, theme.RAY + (45,), local)
            surface.blit(self._overlay, rect.topleft)
            for i, (px, py) in enumerate(pts):
                hit = moving is not None and bool(moving[i])
                col = theme.RAY_MOVING if hit else theme.RAY
                th = -math.pi + i * 2 * math.pi / n
                sx, sy = cx + math.sin(th) * r0, cy - math.cos(th) * r0
                pygame.draw.line(surface, col, (sx, sy), (px, py), 2 if hit else 1)
                pygame.draw.circle(surface, col, (int(px), int(py)), 3 if hit else 2)
            pygame.draw.polygon(surface, theme.RAY, pts, 1)
        car = pygame.Rect(0, 0, car_w, car_l)
        car.center = (cx, cy)
        pygame.draw.rect(surface, theme.AGENT_CAR, car, border_radius=3)
        pygame.draw.rect(surface, theme.TEXT, (car.x + 2, car.y + 2, car_w - 4, max(2, car_l // 5)), border_radius=1)
        if goal is not None:
            fwd, side = goal
            d = math.hypot(fwd, side)
            if d > 1e-6:
                ux, uy = side / d, -fwd / d
                tip_r = R * 0.86
                tip = (cx + ux * tip_r, cy + uy * tip_r)
                base = (cx + ux * r0 * 1.15, cy + uy * r0 * 1.15)
                pygame.draw.line(surface, theme.GOOD, base, tip, 3)
                px, py = -uy, ux
                head = [tip, (tip[0] - ux * 9 + px * 5, tip[1] - uy * 9 + py * 5),
                        (tip[0] - ux * 9 - px * 5, tip[1] - uy * 9 - py * 5)]
                pygame.draw.polygon(surface, theme.GOOD, head)


class NetworkView:
    def __init__(self, max_neurons=32, output_labels=None):
        self.max_neurons = max_neurons
        self.output_labels = output_labels
        self._conn = None
        self._conn_key = None

    @staticmethod
    def reduce(values, m):
        v = np.asarray(values, dtype=np.float64).ravel()
        if len(v) <= m:
            return v
        edges = np.linspace(0, len(v), m + 1).astype(int)[:-1]
        counts = np.diff(np.append(edges, len(v)))
        return np.add.reduceat(v, edges) / counts

    @staticmethod
    def color(v, scale):
        t = clamp(abs(v) / scale, 0.0, 1.0) ** 0.8
        base = (70, 150, 255) if v < 0 else (255, 150, 50)
        return mix((40, 44, 56), base, t)

    def draw(self, surface, rect, layers, highlight=None):
        if not layers:
            empty_state(surface, rect, "no network activations")
            return
        n = len(layers)
        out_labels = self.output_labels if n > 1 and len(layers[-1]["values"]) == len(self.output_labels or []) else None
        label_w = 46 if out_labels else 0
        top = 28
        body = pygame.Rect(rect.x + 6, rect.y + top, rect.w - 12 - label_w, rect.h - top)
        if body.h < 20:
            return
        cols = [self.reduce(l["values"], self.max_neurons) for l in layers]
        sizes = [len(np.asarray(l["values"]).ravel()) for l in layers]
        xs = [body.x + (body.w * (2 * i + 1)) // (2 * n) for i in range(n)]
        def pitch_of(m):
            return min(body.h / float(m), 17.0)

        def ypos(k, m):
            pitch = pitch_of(m)
            return body.y + (body.h - pitch * m) / 2 + pitch * (k + 0.5)

        key = (tuple(len(c) for c in cols), tuple(xs), body.y, body.h)
        if self._conn_key != key:
            conn = pygame.Surface((rect.w + 4, rect.h + 4), pygame.SRCALPHA)
            for i in range(n - 1):
                for a in range(len(cols[i])):
                    for b in range(len(cols[i + 1])):
                        pygame.draw.line(conn, (150, 160, 190, 14),
                                         (xs[i] - rect.x, ypos(a, len(cols[i])) - rect.y),
                                         (xs[i + 1] - rect.x, ypos(b, len(cols[i + 1])) - rect.y))
            self._conn, self._conn_key = conn, key
        surface.blit(self._conn, rect.topleft)
        for i, (c, l) in enumerate(zip(cols, layers)):
            scale = max(float(np.abs(c).max()), 1e-6)
            name = str(l.get("name", ""))
            text(surface, name, (xs[i], rect.y), 11, theme.TEXT, bold=True, anchor="midtop")
            lab = "%d" % sizes[i] + (" (x%d)" % int(math.ceil(sizes[i] / float(len(c)))) if sizes[i] > len(c) else "")
            text(surface, lab, (xs[i], rect.y + 13), 10, theme.TEXT_DIM, anchor="midtop")
            pitch = pitch_of(len(c))
            rad = clamp(int(pitch * 0.42), 2, 7)
            for k, v in enumerate(c):
                y = int(ypos(k, len(c)))
                pygame.draw.circle(surface, self.color(v, scale), (xs[i], y), rad)
                if out_labels and i == n - 1:
                    sel = highlight is not None and k == highlight
                    pygame.draw.circle(surface, theme.WARN if sel else theme.BORDER, (xs[i], y), rad + 1, 1)
                    text(surface, out_labels[k], (xs[i] + rad + 5, y), 11 if pitch >= 11 else 10,
                         theme.WARN if sel else theme.TEXT_DIM, bold=sel, anchor="midleft")


class LineChart:
    def __init__(self, title="", ylim=None, y_fmt=None, xlabel="episode"):
        self.title = title
        self.ylim = ylim
        self.xlabel = xlabel
        self.y_fmt = y_fmt

    def draw(self, surface, rect, series, empty_msg="no data yet"):
        pygame.draw.rect(surface, theme.PANEL, rect, border_radius=4)
        pygame.draw.rect(surface, theme.BORDER, rect, 1, border_radius=4)
        text(surface, self.title, (rect.x + 8, rect.y + 5), 13, theme.TEXT, bold=True)
        lx = rect.right - 8
        limit = rect.x + 12 + theme.font(13, True).size(self.title)[0] + 8
        for s in reversed(series):
            if not s["name"]:
                continue
            last = s.get("last")
            if last is None and len(s["y"]):
                last = float(s["y"][-1])
            lab = s["name"] + (" " + (self.y_fmt(last) if self.y_fmt else fmt(last)) if last is not None else "")
            if lx - theme.font(11).size(lab)[0] - 16 < limit:
                break
            r = text(surface, lab, (lx, rect.y + 6), 11, s["color"], anchor="topright")
            pygame.draw.circle(surface, s["color"], (r.x - 6, r.centery), 3)
            lx = r.x - 16
        left, bottom = 40, 18
        plot = pygame.Rect(rect.x + left, rect.y + 26, rect.w - left - 10, rect.h - 26 - bottom)
        if plot.w < 30 or plot.h < 20:
            return
        data = [s for s in series if len(s["y"]) > 0]
        if not data:
            empty_state(surface, plot, empty_msg)
            return
        xmin = min(float(s["x"][0]) for s in data)
        xmax = max(float(s["x"][-1]) for s in data)
        if xmax - xmin < 1e-9:
            xmax = xmin + 1
        if self.ylim:
            ylo, yhi = self.ylim
        else:
            ylo = min(float(np.min(s["y"])) for s in data)
            yhi = max(float(np.max(s["y"])) for s in data)
            if yhi - ylo < 1e-9:
                yhi, ylo = yhi + 0.5, ylo - 0.5
            pad = (yhi - ylo) * 0.06
            ylo, yhi = ylo - pad, yhi + pad
        yt, ystep = nice_ticks(ylo, yhi, max(2, plot.h // 34))
        xt, xstep = nice_ticks(xmin, xmax, max(2, plot.w // 90))
        for v in yt:
            y = plot.bottom - (v - ylo) / (yhi - ylo) * plot.h
            pygame.draw.line(surface, theme.PANEL_LIGHT if v != 0 else theme.BORDER, (plot.x, y), (plot.right, y))
            text(surface, fmt_tick(v, ystep), (plot.x - 5, y), 11, theme.TEXT_DIM, anchor="midright")
        for v in xt:
            x = plot.x + (v - xmin) / (xmax - xmin) * plot.w
            pygame.draw.line(surface, theme.PANEL_LIGHT, (x, plot.y), (x, plot.bottom))
            text(surface, fmt_tick(v, xstep), (x, plot.bottom + 3), 11, theme.TEXT_DIM, anchor="midtop")
        text(surface, self.xlabel, (plot.right, rect.bottom - 2), 10, theme.TEXT_DIM, anchor="bottomright")
        prev = surface.get_clip()
        surface.set_clip(plot.inflate(2, 2))
        for s in data:
            xs = plot.x + (np.asarray(s["x"], dtype=np.float64) - xmin) / (xmax - xmin) * plot.w
            ys = plot.bottom - (np.clip(np.asarray(s["y"], dtype=np.float64), ylo - (yhi - ylo), yhi + (yhi - ylo)) - ylo) / (yhi - ylo) * plot.h
            if s.get("kind") == "points":
                col = s["color"]
                for x, y in zip(xs.tolist(), ys.tolist()):
                    surface.fill(col, (int(x), int(y), 2, 2))
            elif len(xs) >= 2:
                pygame.draw.lines(surface, s["color"], False, list(zip(xs.tolist(), ys.tolist())), s.get("width", 2))
            else:
                pygame.draw.circle(surface, s["color"], (int(xs[0]), int(ys[0])), 3)
        surface.set_clip(prev)
