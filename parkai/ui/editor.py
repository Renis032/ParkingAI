import copy
import math
import os
import time

import numpy as np
import pygame

from .. import geometry as geo
from ..world import World, Slot, ParkedCar, Obstacle, Spawn, MovingObject, default_world
from . import theme as T
from .lot_view import arrow, dashed_line

TOOLS = [("select", "Select", "1"), ("slot", "Slot", "2"), ("car", "Parked car", "3"), ("obstacle", "Obstacle", "4"),
         ("spawn", "Spawn", "5"), ("mcar", "Moving car", "6"), ("ped", "Pedestrian", "7")]
CAR_COLORS = [[178, 64, 64], [64, 120, 178], [90, 160, 90], [200, 170, 60], [140, 90, 170], [90, 90, 100],
              [210, 210, 215], [60, 60, 66]]
LISTS = {"slot": "slots", "car": "parked_cars", "obstacle": "obstacles", "spawn": "spawns", "mover": "moving"}
LABELS = {"slot": "Slot", "car": "Parked car", "obstacle": "Obstacle", "spawn": "Spawn", "mover": "Moving"}
ROT_STEP = math.radians(15)


class Editor:
    def __init__(self, world, lot_path, toast):
        self.lot_path = lot_path
        self.toast = toast
        self.world = world
        self.rev = 0
        self.rev0 = 0
        self.tool = "select"
        self.sel = None
        self.wp = None
        self.snap_on = True
        self.snap = 0.25
        self.drag = None
        self.pend = None
        self.rot = -math.pi / 2
        self.cursor = (0.0, 0.0)
        self.hover = None
        self.undo = []
        self.last_click = (0.0, (0.0, 0.0))
        self.color_i = 0
        self.mover_speed = 1.5
        self.mover_closed = False

    def set_world(self, world):
        self.world = world
        self.rev += 1
        self.rev0 = self.rev
        self.sel = self.wp = self.drag = self.pend = self.hover = None
        self.undo = []

    @property
    def dirty(self):
        return self.rev != self.rev0

    def touch(self):
        self.rev += 1

    def push_undo(self):
        self.undo.append(copy.deepcopy(self.world))
        del self.undo[:-60]

    def do_undo(self):
        if not self.undo:
            self.toast("Nothing to undo")
            return
        self.world = self.undo.pop()
        self.sel = self.wp = self.drag = None
        self.touch()
        self.toast("Undo")

    def sn(self, v):
        return round(v / self.snap) * self.snap if self.snap_on else v

    def snp(self, x, y):
        return self.sn(x), self.sn(y)

    def get(self, sel=None):
        sel = sel or self.sel
        if sel is None:
            return None
        lst = getattr(self.world, LISTS[sel[0]])
        return lst[sel[1]] if 0 <= sel[1] < len(lst) else None

    def set_tool(self, tool):
        self.tool = tool
        self.pend = None
        self.drag = None
        if tool != "select":
            self.sel = self.wp = None

    def pick(self, wx, wy, scale):
        w = self.world
        hits = []
        r = max(0.5, 8.0 / scale)
        for i, s in enumerate(w.spawns):
            if math.hypot(wx - s.x, wy - s.y) < max(0.9, r):
                hits.append(("spawn", i, None))
        for i, m in enumerate(w.moving):
            best = None
            for k, p in enumerate(m.path):
                if math.hypot(wx - p[0], wy - p[1]) < max(0.7, r):
                    best = k
                    break
            if best is not None:
                hits.append(("mover", i, best))
                continue
            pts = np.array(m.path, float)
            for k in range(len(pts) - (0 if m.closed and len(pts) > 2 else 1)):
                a, b = pts[k], pts[(k + 1) % len(pts)]
                d = b - a
                t = 0 if d @ d < 1e-9 else float(np.clip(((wx - a[0]) * d[0] + (wy - a[1]) * d[1]) / (d @ d), 0, 1))
                if math.hypot(wx - (a[0] + d[0] * t), wy - (a[1] + d[1] * t)) < max(0.5, r * 0.8):
                    hits.append(("mover", i, None))
                    break
        pt = np.array([wx, wy])
        for kind, lst in (("car", w.parked_cars), ("obstacle", w.obstacles), ("slot", w.slots)):
            for i in range(len(lst) - 1, -1, -1):
                o = lst[i]
                if geo.points_in_rects(pt, o.x, o.y, o.angle, o.length, o.width):
                    hits.append((kind, i, None))
        return hits

    def choose(self, hits, cycle):
        if not hits:
            return None
        keys = [(h[0], h[1]) for h in hits]
        if self.sel in keys:
            i = keys.index(self.sel)
            return hits[(i + 1) % len(hits)] if cycle else hits[i]
        return hits[0]

    def mouse_down(self, pos, view, mods):
        wx, wy = view.s2w(*pos)
        x, y = self.snp(wx, wy)
        self.cursor = (x, y)
        t = self.tool
        if t == "select":
            hit = self.choose(self.pick(wx, wy, view.scale), bool(mods & pygame.KMOD_SHIFT))
            if hit is None:
                self.sel = self.wp = None
                return
            kind, idx, wp = hit
            self.sel, self.wp = (kind, idx), wp
            o = self.get()
            if kind == "mover":
                self.drag = {"start": (wx, wy), "pts": [list(p) for p in o.path], "moved": False}
            else:
                self.drag = {"off": (o.x - wx, o.y - wy), "moved": False}
        elif t in ("slot", "car"):
            self.push_undo()
            if t == "slot":
                target = not any(s.target for s in self.world.slots)
                self.world.slots.append(Slot(x, y, geo.wrap_angle(self.rot), target=target))
                self.sel = ("slot", len(self.world.slots) - 1)
            else:
                col = list(CAR_COLORS[self.color_i % len(CAR_COLORS)])
                self.color_i += 1
                self.world.parked_cars.append(ParkedCar(x, y, geo.wrap_angle(self.rot), color=col))
                self.sel = ("car", len(self.world.parked_cars) - 1)
            self.touch()
        elif t == "obstacle":
            self.pend = {"a": (x, y), "b": (x, y)}
        elif t == "spawn":
            self.pend = {"pos": (x, y), "angle": 0.0, "set": False}
        elif t in ("mcar", "ped"):
            now = time.time()
            double = self.pend is not None and now - self.last_click[0] < 0.4 and \
                math.hypot(x - self.last_click[1][0], y - self.last_click[1][1]) < 0.6
            self.last_click = (now, (x, y))
            if double:
                self.finish_mover()
            elif self.pend is None:
                self.pend = {"pts": [[x, y]]}
            else:
                self.pend["pts"].append([x, y])

    def mouse_move(self, pos, view, mods, buttons):
        wx, wy = view.s2w(*pos)
        self.cursor = self.snp(wx, wy)
        if self.tool == "select" and self.drag is None:
            h = self.pick(wx, wy, view.scale)
            self.hover = (h[0][0], h[0][1]) if h else None
        if self.drag is not None and buttons:
            o = self.get()
            if o is None:
                return
            if not self.drag["moved"]:
                self.push_undo()
                self.drag["moved"] = True
            if self.sel[0] == "mover":
                if self.wp is not None:
                    o.path[self.wp] = [self.sn(wx), self.sn(wy)]
                else:
                    dx, dy = self.sn(wx - self.drag["start"][0]), self.sn(wy - self.drag["start"][1])
                    o.path = [[p[0] + dx, p[1] + dy] for p in self.drag["pts"]]
            else:
                o.x, o.y = self.sn(wx + self.drag["off"][0]), self.sn(wy + self.drag["off"][1])
            self.touch()
        if self.pend is not None:
            if "b" in self.pend:
                self.pend["b"] = self.cursor
            elif "pos" in self.pend and buttons:
                dx, dy = wx - self.pend["pos"][0], wy - self.pend["pos"][1]
                if math.hypot(dx, dy) > 0.4:
                    self.pend["angle"] = math.atan2(dy, dx)
                    self.pend["set"] = True

    def mouse_up(self, pos, view, mods):
        self.drag = None
        p = self.pend
        if p is None:
            return
        if "a" in p:
            (ax, ay), (bx, by) = p["a"], p["b"]
            l, w = abs(bx - ax), abs(by - ay)
            if l < 0.2 and w < 0.2:
                l = w = 1.0
                cx, cy = ax, ay
            else:
                l, w = max(l, 0.3), max(w, 0.3)
                cx, cy = (ax + bx) / 2, (ay + by) / 2
            self.push_undo()
            self.world.obstacles.append(Obstacle(cx, cy, 0.0, l, w))
            self.sel = ("obstacle", len(self.world.obstacles) - 1)
            self.pend = None
            self.touch()
        elif "pos" in p:
            self.push_undo()
            self.world.spawns.append(Spawn(p["pos"][0], p["pos"][1], geo.wrap_angle(p["angle"])))
            self.sel = ("spawn", len(self.world.spawns) - 1)
            self.pend = None
            self.touch()

    def finish_mover(self):
        p = self.pend
        if p is None or "pts" not in p:
            return
        pts = p["pts"]
        while len(pts) > 1 and math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]) < 0.1:
            pts.pop()
        if len(pts) < 2:
            self.toast("A path needs at least 2 waypoints", T.WARN)
            self.pend = None
            return
        self.push_undo()
        kind = "pedestrian" if self.tool == "ped" else "car"
        speed = self.mover_speed if kind == "car" else min(self.mover_speed, 1.5)
        self.world.moving.append(MovingObject(kind=kind, path=pts, speed=speed, closed=self.mover_closed and len(pts) > 2))
        self.sel = ("mover", len(self.world.moving) - 1)
        self.wp = None
        self.pend = None
        self.touch()
        self.toast("Added moving %s with %d waypoints" % (kind, len(pts)))

    def rotate(self, delta):
        if self.pend is not None and "pts" not in self.pend and "a" not in self.pend:
            return
        if self.tool in ("slot", "car") and self.sel is None:
            self.rot = geo.wrap_angle(self.rot + delta)
            return
        o = self.get()
        if o is None:
            if self.tool in ("slot", "car"):
                self.rot = geo.wrap_angle(self.rot + delta)
            return
        self.push_undo()
        if self.sel[0] == "mover":
            pts = np.array(o.path, float)
            c = pts.mean(0)
            cs, sn = math.cos(delta), math.sin(delta)
            d = pts - c
            o.path = np.stack([c[0] + d[:, 0] * cs - d[:, 1] * sn, c[1] + d[:, 0] * sn + d[:, 1] * cs], 1).tolist()
        else:
            o.angle = geo.wrap_angle(o.angle + delta)
            if self.tool in ("slot", "car"):
                self.rot = o.angle
        self.touch()

    def delete(self):
        if self.sel is None:
            return
        lst = getattr(self.world, LISTS[self.sel[0]])
        if 0 <= self.sel[1] < len(lst):
            self.push_undo()
            del lst[self.sel[1]]
            self.toast("Deleted %s" % LABELS[self.sel[0]].lower())
            self.touch()
        self.sel = self.wp = self.drag = None

    def duplicate(self):
        o = self.get()
        if o is None:
            return
        self.push_undo()
        n = copy.deepcopy(o)
        if self.sel[0] == "mover":
            n.path = [[p[0] + 1, p[1] + 1] for p in n.path]
        else:
            n.x += 1.0
            n.y += 1.0
        if self.sel[0] == "slot":
            n.target = False
        lst = getattr(self.world, LISTS[self.sel[0]])
        lst.append(n)
        self.sel = (self.sel[0], len(lst) - 1)
        self.touch()

    def nudge(self, dx, dy):
        o = self.get()
        if o is None:
            return
        self.push_undo()
        if self.sel[0] == "mover":
            o.path = [[p[0] + dx, p[1] + dy] for p in o.path]
        else:
            o.x += dx
            o.y += dy
        self.touch()

    def toggle_target(self):
        if self.sel and self.sel[0] == "slot":
            self.push_undo()
            s = self.get()
            s.target = not s.target
            self.touch()

    def change_speed(self, d):
        o = self.get() if self.sel and self.sel[0] == "mover" else None
        if o is not None:
            self.push_undo()
            o.speed = round(max(0.25, min(10.0, o.speed + d)), 2)
            self.touch()
        else:
            self.mover_speed = round(max(0.25, min(10.0, self.mover_speed + d)), 2)

    def toggle_closed(self):
        o = self.get() if self.sel and self.sel[0] == "mover" else None
        if o is not None:
            self.push_undo()
            o.closed = not o.closed
            self.touch()
        else:
            self.mover_closed = not self.mover_closed

    def wheel(self, dy, mods):
        if mods & pygame.KMOD_SHIFT:
            self.rotate(ROT_STEP * dy)
            return True
        return False

    def key(self, e):
        mods = e.mod
        ctrl = mods & pygame.KMOD_CTRL
        shift = mods & pygame.KMOD_SHIFT
        k = e.key
        for name, _, num in TOOLS:
            if k == ord(num) and not ctrl:
                self.set_tool(name)
                return True
        if k == pygame.K_s and ctrl:
            self.save()
        elif k == pygame.K_z and ctrl:
            self.do_undo()
        elif k == pygame.K_d and ctrl:
            self.duplicate()
        elif k in (pygame.K_DELETE, pygame.K_BACKSPACE):
            self.delete()
        elif k in (pygame.K_q, pygame.K_e):
            step = math.radians(1) if shift else ROT_STEP
            self.rotate(-step if k == pygame.K_q else step)
        elif k == pygame.K_r and not ctrl:
            if self.sel is not None and self.tool == "select":
                self.rotate(math.pi / 2)
            else:
                self.rot = geo.wrap_angle(self.rot + math.pi / 2)
        elif k == pygame.K_t:
            self.toggle_target()
        elif k == pygame.K_g:
            self.snap_on = not self.snap_on
            self.toast("Grid snap %s" % ("on (0.25 m)" if self.snap_on else "off"))
        elif k == pygame.K_c:
            self.toggle_closed()
        elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.change_speed(0.25)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.change_speed(-0.25)
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.finish_mover()
        elif k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN) and self.sel:
            st = 1.0 if shift else self.snap
            self.nudge({pygame.K_LEFT: -st, pygame.K_RIGHT: st}.get(k, 0), {pygame.K_UP: -st, pygame.K_DOWN: st}.get(k, 0))
        elif k == pygame.K_ESCAPE:
            if self.pend is not None:
                self.pend = None
            elif self.sel is not None:
                self.sel = self.wp = None
            elif self.tool != "select":
                self.set_tool("select")
            else:
                return False
        else:
            return False
        return True

    def save(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.lot_path)), exist_ok=True)
            self.world.save(self.lot_path)
            self.toast("Saved lot to %s" % os.path.basename(self.lot_path), T.GOOD)
        except Exception as ex:
            self.toast("Save failed: %s" % ex, T.BAD)

    def load(self, path):
        try:
            w = World.load(path)
        except Exception as ex:
            self.toast("Load failed: %s" % ex, T.BAD)
            return
        self.push_undo()
        self.world = w
        self.sel = self.wp = self.pend = None
        self.touch()
        self.toast("Loaded %s" % os.path.basename(path), T.GOOD)

    def clear(self):
        self.push_undo()
        w = self.world
        w.slots, w.parked_cars, w.obstacles, w.spawns, w.moving = [], [], [], [], []
        self.sel = self.wp = self.pend = None
        self.touch()
        self.toast("Cleared the lot (Ctrl+Z to undo)")

    def reset_default(self):
        self.push_undo()
        self.world = default_world()
        self.sel = self.wp = self.pend = None
        self.touch()
        self.toast("Reset to the default lot (Ctrl+Z to undo)")

    def describe(self):
        o = self.get()
        if o is None:
            return None
        k = self.sel[0]
        n = "%s #%d" % (LABELS[k], self.sel[1] + 1)
        if k == "mover":
            c = np.array(o.path, float).mean(0)
            return "%s (%s)  %d waypoints  speed %.2f m/s  %s  centre (%.2f, %.2f)" % (
                n, o.kind, len(o.path), o.speed, "closed loop" if o.closed else "back and forth", c[0], c[1])
        s = "%s  x=%.2f  y=%.2f  angle=%d deg" % (n, o.x, o.y, round(math.degrees(o.angle)))
        if k in ("slot", "car", "obstacle"):
            s += "  size %.2f x %.2f m" % (o.length, o.width)
        if k == "slot":
            s += "  TARGET" if o.target else "  (not a target)"
        if k == "spawn":
            s += "  jitter %.1f m / %d deg" % (o.jitter_pos, round(math.degrees(o.jitter_angle)))
        return s

    def hint(self):
        t = self.tool
        if self.pend is not None and "pts" in self.pend:
            return "Click to add waypoints, Enter or double-click to finish, C closed loop, +/- speed, Esc cancel"
        return {
            "select": "Click to select (Shift+click cycles overlapping objects), drag to move, Q/E rotate, T target, Del delete, Ctrl+D duplicate",
            "slot": "Click to place a slot (R rotates 90 deg, Q/E 15 deg). The first slot becomes the target",
            "car": "Click to place a parked car (R rotates 90 deg)",
            "obstacle": "Click and drag to size an obstacle (a plain click makes 1 x 1 m)",
            "spawn": "Press to set the start position, drag to set the heading",
            "mcar": "Click waypoints for the car's path. Enter or double-click to finish",
            "ped": "Click waypoints for the pedestrian's path. Enter or double-click to finish",
        }[t]

    def draw_overlay(self, surf, view, mouse):
        sel = self.sel
        if self.hover is not None and self.hover != sel and self.tool == "select":
            self._outline(surf, view, self.hover, (255, 255, 255), 1)
        if sel is not None:
            self._outline(surf, view, sel, T.ACCENT, 3)
            o = self.get()
            if o is not None and sel[0] == "mover":
                for i, p in enumerate(o.path):
                    c = view.w2s(p[0], p[1])
                    col = T.WARN if i == self.wp else T.ACCENT
                    pygame.draw.circle(surf, col, (int(c[0]), int(c[1])), 7, 2)
                    T.text(surf, i + 1, (c[0] + 9, c[1] - 14), 12, T.TEXT)
        if not view.rect.collidepoint(mouse):
            return
        p = self.pend
        cx, cy = self.cursor
        if p is not None and "pts" in p:
            col = T.PEDESTRIAN if self.tool == "ped" else T.MOVING_CAR
            pts = [view.w2s(q[0], q[1]) for q in p["pts"]] + [view.w2s(cx, cy)]
            for a, b in zip(pts[:-1], pts[1:]):
                dashed_line(surf, col, a, b, 8, 5, 3)
            for i, q in enumerate(pts[:-1]):
                pygame.draw.circle(surf, col, (int(q[0]), int(q[1])), 6)
                T.text(surf, i + 1, (q[0] + 9, q[1] - 14), 12, T.TEXT)
        elif p is not None and "a" in p:
            (ax, ay), (bx, by) = p["a"], p["b"]
            pts = view.poly((ax + bx) / 2, (ay + by) / 2, 0, max(abs(bx - ax), 0.3), max(abs(by - ay), 0.3))
            pygame.draw.polygon(surf, T.OBSTACLE, pts, 3)
            T.text(surf, "%.2f x %.2f m" % (abs(bx - ax), abs(by - ay)), (mouse[0] + 14, mouse[1] + 14), 13, T.TEXT)
        elif p is not None and "pos" in p:
            c = view.w2s(*p["pos"])
            arrow(surf, T.SPAWN, c, view.w2s(p["pos"][0] + math.cos(p["angle"]) * 2, p["pos"][1] + math.sin(p["angle"]) * 2), 10, 3)
            pygame.draw.circle(surf, T.SPAWN, (int(c[0]), int(c[1])), 6)
        elif self.tool in ("slot", "car"):
            l, w = (5.5, 2.7) if self.tool == "slot" else (4.5, 1.9)
            pts = view.poly(cx, cy, self.rot, l, w)
            pygame.draw.polygon(surf, T.TARGET_SLOT if self.tool == "slot" else T.PARKED_CAR, pts, 3)
            c = view.w2s(cx, cy)
            tip = view.w2s(cx + math.cos(self.rot) * 1.5, cy + math.sin(self.rot) * 1.5)
            arrow(surf, T.TEXT, c, tip, 8, 2)
        elif self.tool == "obstacle":
            pts = view.poly(cx, cy, 0, 1, 1)
            pygame.draw.polygon(surf, T.OBSTACLE, pts, 2)
        elif self.tool == "spawn":
            c = view.w2s(cx, cy)
            pygame.draw.circle(surf, T.SPAWN, (int(c[0]), int(c[1])), 6, 2)
        elif self.tool in ("mcar", "ped"):
            c = view.w2s(cx, cy)
            pygame.draw.circle(surf, T.PEDESTRIAN if self.tool == "ped" else T.MOVING_CAR, (int(c[0]), int(c[1])), 6, 2)
        T.text(surf, "%.2f, %.2f" % (cx, cy), (mouse[0] + 14, mouse[1] - 18), 12, T.TEXT_DIM)

    def _outline(self, surf, view, sel, color, width):
        o = self.get(sel)
        if o is None:
            return
        k = sel[0]
        if k == "mover":
            pts = [view.w2s(p[0], p[1]) for p in o.path]
            if len(pts) > 1:
                pygame.draw.lines(surf, color, bool(o.closed and len(pts) > 2), pts, max(1, width - 1))
            return
        if k == "spawn":
            c = view.w2s(o.x, o.y)
            pygame.draw.circle(surf, color, (int(c[0]), int(c[1])), 12, width)
            return
        pts = view.poly(o.x, o.y, o.angle, o.length, o.width)
        pygame.draw.polygon(surf, color, pts, width)
        if width > 1:
            for p in pts:
                pygame.draw.rect(surf, color, (p[0] - 3, p[1] - 3, 7, 7))
            c = view.w2s(o.x, o.y)
            arrow(surf, color, c, view.w2s(o.x + math.cos(o.angle) * min(o.length * 0.5, 1.5),
                                           o.y + math.sin(o.angle) * min(o.length * 0.5, 1.5)), 7, 2)

    def draw_toolbar(self, surf, rect, ctl):
        x = rect.x + 8
        for name, label, num in TOOLS:
            w = 28 + len(label) * 8
            if ctl.button(surf, (x, rect.y + 3, w, rect.h - 6), "%s %s" % (num, label), active=self.tool == name,
                          tip=label):
                self.set_tool(name)
            x += w + 4

    def draw_panel(self, surf, rect, ctl, app):
        pygame.draw.rect(surf, T.PANEL, rect)
        pygame.draw.line(surf, T.BORDER, rect.topleft, rect.bottomleft)
        x, y, w = rect.x + 16, rect.y + 14, rect.w - 32
        T.text(surf, "Lot editor", (x, y), 20, T.TEXT, bold=True)
        T.text(surf, os.path.basename(self.lot_path) + ("  (unsaved changes)" if self.dirty else ""),
               (rect.right - 16, y + 4), 13, T.WARN if self.dirty else T.TEXT_DIM, anchor="topright")
        y += 34
        wd = self.world
        nw = ctl.stepper(surf, (x, y, w, 24), "Lot width (m)", wd.width, 1, 10, 100, "{:.0f}")
        y += 30
        nh = ctl.stepper(surf, (x, y, w, 24), "Lot height (m)", wd.height, 1, 10, 100, "{:.0f}")
        if nw != wd.width or nh != wd.height:
            self.push_undo()
            wd.width, wd.height = nw, nh
            self.touch()
        y += 34
        self.snap_on = ctl.toggle(surf, (x, y, 150, 26), "Grid snap 0.25 m (G)", self.snap_on)
        app.view.show_grid = ctl.toggle(surf, (x + 158, y, 120, 26), "Show grid", app.view.show_grid)
        y += 40
        bw = (w - 12) // 3
        if ctl.button(surf, (x, y, bw, 28), "Save lot", color=(40, 90, 60), tip="Ctrl+S"):
            self.save()
        if ctl.button(surf, (x + bw + 6, y, bw, 28), "Load lot", tip="Ctrl+O"):
            app.open_lot_list()
        if ctl.button(surf, (x + 2 * (bw + 6), y, bw, 28), "Undo", enabled=bool(self.undo), tip="Ctrl+Z"):
            self.do_undo()
        y += 34
        if ctl.button(surf, (x, y, (w - 6) // 2, 28), "Clear lot"):
            self.clear()
        if ctl.button(surf, (x + (w - 6) // 2 + 6, y, (w - 6) // 2, 28), "Reset to default"):
            self.reset_default()
        y += 44
        pygame.draw.line(surf, T.BORDER, (x, y - 8), (x + w, y - 8))
        n_t = len(wd.target_slots())
        rows = [("Slots", "%d  (%d usable targets)" % (len(wd.slots), n_t), T.TEXT if n_t else T.BAD),
                ("Parked cars", str(len(wd.parked_cars)), T.TEXT), ("Obstacles", str(len(wd.obstacles)), T.TEXT),
                ("Spawn points", str(len(wd.spawns)) if wd.spawns else "0  (agent starts at the centre)",
                 T.TEXT if wd.spawns else T.WARN),
                ("Moving objects", "%d" % len(wd.moving), T.TEXT)]
        for lab, val, col in rows:
            T.text(surf, lab, (x, y), 14, T.TEXT_DIM)
            T.text(surf, val, (x + w, y), 14, col, anchor="topright")
            y += 20
        if n_t == 0:
            T.text(surf, "No free slot to park in! Add a slot and clear its cars.", (x, y + 2), 13, T.BAD)
        y += 28
        pygame.draw.line(surf, T.BORDER, (x, y - 8), (x + w, y - 8))
        T.text(surf, "Selection", (x, y), 16, T.TEXT, bold=True)
        y += 26
        o = self.get()
        if o is None:
            for line in ["Nothing selected.", "Pick a tool (1-7), or Select and click an object.",
                         "", "Wheel zoom, right/middle drag pans.", "Shift+wheel or Q/E rotates by 15 deg.",
                         "Arrow keys nudge (Shift = 1 m)."]:
                T.text(surf, line, (x, y), 13, T.TEXT_DIM)
                y += 18
            return
        k = self.sel[0]
        T.text(surf, "%s #%d" % (LABELS[k], self.sel[1] + 1), (x, y), 15, T.ACCENT, bold=True)
        y += 26
        changed = False

        def step(label, value, st, lo, hi, fmt="{:.2f}"):
            nonlocal y, changed
            v = ctl.stepper(surf, (x, y, w, 24), label, value, st, lo, hi, fmt)
            y += 30
            if v != value:
                if not changed:
                    self.push_undo()
                changed = True
            return v

        if k in ("slot", "car", "obstacle"):
            o.length = step("Length (m)", o.length, 0.25, 0.25, 40.0)
            o.width = step("Width (m)", o.width, 0.1 if k != "obstacle" else 0.25, 0.25, 20.0)
        if k == "slot":
            if ctl.toggle(surf, (x, y, w, 26), "Target slot (T)", o.target) != o.target:
                self.toggle_target()
            y += 32
        elif k == "car":
            if ctl.button(surf, (x, y, w, 26), "Next colour"):
                self.push_undo()
                self.color_i += 1
                o.color = list(CAR_COLORS[self.color_i % len(CAR_COLORS)])
                changed = True
            y += 32
        elif k == "spawn":
            o.jitter_pos = step("Position jitter (m)", o.jitter_pos, 0.25, 0.0, 10.0)
            deg = step("Heading jitter (deg)", round(math.degrees(o.jitter_angle)), 5, 0, 180, "{:.0f}")
            o.jitter_angle = math.radians(deg)
        elif k == "mover":
            o.speed = step("Speed (m/s)", o.speed, 0.25, 0.25, 10.0)
            if ctl.toggle(surf, (x, y, w, 26), "Closed loop (C)", o.closed) != o.closed:
                self.push_undo()
                o.closed = not o.closed
                changed = True
            y += 32
            o.random_phase = ctl.toggle(surf, (x, y, w, 26), "Random start position per car", o.random_phase)
            y += 32
        if changed:
            self.touch()
        if ctl.button(surf, (x, y, (w - 6) // 2, 28), "Duplicate (Ctrl+D)"):
            self.duplicate()
        if ctl.button(surf, (x + (w - 6) // 2 + 6, y, (w - 6) // 2, 28), "Delete (Del)", color=(120, 50, 46)):
            self.delete()
