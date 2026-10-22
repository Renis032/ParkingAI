import glob
import math
import os
import sys
import time
import traceback

import pygame

from ..player import Player
from ..trainer import Trainer
from ..env import ACTION_NAMES
from . import theme as T
from .controls import Controls, ListPopup, mix
from .editor import Editor
from .lot_view import LotView, arrow, outcome_of

try:
    from .panels import InspectorPanel
except ImportError:
    InspectorPanel = None
try:
    from .panels import ChartsPanel
except ImportError:
    ChartsPanel = None

BUILD, TRAIN, WATCH, REPLAY = 0, 1, 2, 3
MODE_NAMES = ["BUILD", "TRAIN", "WATCH", "REPLAY"]
TRAIN_SPEEDS = ["Real-time", "Fast", "Turbo", "Max"]
WATCH_SPEEDS = [1, 2, 4, 8]
REPLAY_SPEEDS = [0.5, 1, 2, 4]
TOP_H = 70
STATUS_H = 24
RIGHT_W = 400
CHARTS_H = 240
BAR_H = 30
LIST_W = 270
AUTOSAVE_SECONDS = 120.0

HELP = [
    ("Everywhere", [
        ("F2 F3 F4 F5, Tab", "switch mode, cycle modes"),
        ("H or F1", "this help"),
        ("Mouse wheel", "zoom the lot"),
        ("Right / middle drag", "pan the lot"),
        ("F", "fit the lot to the window"),
        ("L / T / G / V / P / O", "rays / trail / ghosts / value map / policy / spawns"),
        ("Ctrl+S", "save checkpoint (BUILD: save lot)"),
        ("Ctrl+O", "load checkpoint (BUILD: load lot)"),
        ("Esc (twice)", "quit, autosaves the checkpoint"),
    ]),
    ("TRAIN", [
        ("Space", "start / pause training"),
        ("1 2 3 4", "speed: Real-time, Fast, Turbo, Max"),
    ]),
    ("WATCH", [
        ("Drag in the lot", "place start pose (drag = heading)"),
        ("Space", "pause"),
        ("N / R", "new spawn / restart same pose"),
        ("S", "stochastic vs deterministic actions"),
        ("A", "auto-repeat"),
        ("1 2 3 4", "speed 1x 2x 4x 8x"),
    ]),
    ("REPLAY", [
        ("Click a row, Up / Down", "choose an episode"),
        ("Space", "play / pause"),
        ("Left / Right, Home / End", "step a frame / jump to start or end"),
        ("1 2 3 4", "speed 0.5x 1x 2x 4x"),
        ("Drag the scrubber", "seek"),
    ]),
    ("BUILD", [
        ("1 .. 7", "tools: select, slot, car, obstacle, spawn, mover, ped"),
        ("Click / drag", "select / move (Shift+click cycles overlaps)"),
        ("Q / E, Shift+wheel", "rotate 15 deg (Shift+Q/E 1 deg), R = 90 deg"),
        ("Del / Backspace", "delete the selection"),
        ("T", "toggle target slot"),
        ("G", "grid snap 0.25 m"),
        ("Enter / double-click", "finish path; C closed loop; +/- speed"),
        ("Arrows", "nudge the selection (Shift = 1 m)"),
        ("Ctrl+D / Ctrl+Z", "duplicate / undo"),
        ("Ctrl+S / Ctrl+O", "save / load lot"),
    ]),
]


class App:
    def __init__(self, trainer, lot_path, checkpoint_path, size=(1600, 900), fit_display=True):
        pygame.init()
        pygame.display.set_caption("ParkingAI")
        if fit_display:
            try:
                info = pygame.display.Info()
                if info.current_w > 0 and info.current_h > 0:
                    size = (min(size[0], info.current_w - 40), min(size[1], info.current_h - 80))
            except Exception:
                pass
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.trainer = trainer
        self.lot_path = lot_path
        self.checkpoint_path = checkpoint_path
        self.ctl = Controls()
        self.view = LotView()
        self.editor = None
        self.player = None
        self.running = True
        self.mode = TRAIN
        self.training = True
        self.train_speed = 1
        self.watch_speed = 0
        self.replay_speed = 1
        self.rt_acc = 0.0
        self.last_t = time.perf_counter()
        self.last_draw = 0.0
        self.last_save_tt = trainer.train_time
        self.toast_msg = None
        self.popup = None
        self.show_help = False
        self.quit_armed = 0.0
        self.trail = []
        self._ep_seen = None
        self.capture = None
        self.summary = None
        self.summary_t = 0.0
        self.fps = 0.0
        self.insp = None
        self.insp_frame = None
        self.insp_t = 0.0
        self.panel_err = {}
        self.inspector = None
        self.charts = None
        self.watch_paused = False
        self.watch_acc = 0.0
        self.watch_frame = None
        self.auto_repeat = True
        self.finish_t = 0.0
        self.place = None
        self.rp_ep = None
        self.rp_i = 0
        self.rp_playing = True
        self.rp_acc = 0.0
        self.rp_scroll = 0
        self.mods = pygame.key.get_mods
        self._entered = False
        self.make_panels()
        self.layout()
        self.set_mode(TRAIN)

    def make_panels(self):
        env = self.trainer.env
        self.inspector = self.charts = None
        if InspectorPanel is not None:
            try:
                self.inspector = InspectorPanel(env.obs_names, env.n_rays)
            except Exception as ex:
                self.panel_err["inspector"] = (time.time(), "InspectorPanel: %s" % ex)
        if ChartsPanel is not None:
            try:
                self.charts = ChartsPanel()
            except Exception as ex:
                self.panel_err["charts"] = (time.time(), "ChartsPanel: %s" % ex)

    def toast(self, msg, color=None, seconds=4.0):
        self.toast_msg = (msg, color or T.TEXT, time.time() + seconds)

    def layout(self):
        self.screen = pygame.display.get_surface()
        W, H = self.screen.get_size()
        right = RIGHT_W if W >= 1250 else 300
        charts_h = CHARTS_H if H >= 800 else max(110, int(H * 0.26))
        self.top = pygame.Rect(0, 0, W, TOP_H)
        self.status = pygame.Rect(0, H - STATUS_H, W, STATUS_H)
        self.right = pygame.Rect(W - right, TOP_H, right, H - TOP_H - STATUS_H)
        self.charts_rect = pygame.Rect(0, H - STATUS_H - charts_h, W - right, charts_h)
        self.center = pygame.Rect(0, TOP_H, W - right, self.charts_rect.y - TOP_H)
        c = self.center
        bars = BAR_H * (2 if self.mode == BUILD else 1)
        self.bar_rect = pygame.Rect(c.x, c.y, c.w, BAR_H)
        self.tool_rect = pygame.Rect(c.x, c.y + BAR_H, c.w, BAR_H)
        x0 = c.x + (LIST_W if self.mode == REPLAY else 0)
        bottom = 0
        if self.mode == BUILD:
            bottom = 28
        elif self.mode == REPLAY:
            bottom = 62
        self.list_rect = pygame.Rect(c.x, c.y, LIST_W, c.h)
        self.canvas = pygame.Rect(x0, c.y + bars, c.right - x0, c.h - bars - bottom)
        self.bottom_rect = pygame.Rect(x0, self.canvas.bottom, c.right - x0, bottom)
        if self.mode == REPLAY:
            self.bar_rect = pygame.Rect(x0, c.y, c.right - x0, BAR_H)

    def current_world(self):
        if self.mode == BUILD and self.editor is not None:
            return self.editor.world
        if self.mode == REPLAY and self.rp_ep is not None:
            return self.rp_ep["world"]
        return self.trainer.world

    def set_mode(self, mode):
        if mode == self.mode and self._entered:
            return
        if self.mode == BUILD:
            self.leave_build()
        self.mode = mode
        self._enter(mode)

    def _enter(self, mode):
        self._entered = True
        self.capture = None
        self.place = None
        self.trail = []
        self.view.maps = None
        self.view.effects = []
        self.layout()
        if mode == BUILD:
            self.training = False
            if self.editor is None:
                self.editor = Editor(self.trainer.world.copy(), self.lot_path, self.toast)
            else:
                self.editor.set_world(self.trainer.world.copy())
            self.view.fit(self.editor.world)
        elif mode == TRAIN:
            self.training = True
            self.last_save_tt = max(self.last_save_tt, self.trainer.train_time)
            self._ep_seen = self.trainer.recent_episodes[-1] if self.trainer.recent_episodes else None
        elif mode == WATCH:
            self.training = False
            self.ensure_player()
            self.watch_frame = None
        elif mode == REPLAY:
            self.training = False
            items = self.replay_items()
            if self.rp_ep is None or all(ep is not self.rp_ep for _, ep in items):
                self.rp_ep = items[0][1] if items else None
                self.rp_i = 0
            self.rp_playing = True
            self.rp_acc = 0.0
            self.rp_i = 0
        pygame.display.set_caption("ParkingAI - %s" % MODE_NAMES[mode])
        self.view.manual = False
        self._fit_key_reset()

    def _fit_key_reset(self):
        self.view._fit_key = None

    def leave_build(self):
        ed = self.editor
        if ed is None or not ed.dirty:
            return
        self.trainer.set_world(ed.world.copy())
        self.player = None
        self.rp_ep = None
        ed.rev0 = ed.rev
        self.toast("Lot applied to the trainer (agent weights kept)", T.GOOD)
        if not ed.world.target_slots():
            self.toast("Warning: this lot has no free target slot, nothing to park in!", T.BAD, 8.0)
        elif not ed.world.spawns:
            self.toast("Lot applied. Note: no spawn points, cars start in the centre", T.WARN, 6.0)

    def replay_items(self):
        tr = self.trainer
        items = []
        if tr.best_episode is not None:
            items.append(("best", tr.best_episode))
        items.extend(("", ep) for ep in reversed(tr.recent_episodes) if ep is not tr.best_episode)
        return items

    def ensure_player(self):
        p = self.player
        if p is None or p.trainer is not self.trainer or p.env.world is not self.trainer.world:
            self.player = Player(self.trainer, deterministic=True if p is None else p.deterministic)
            self.watch_frame = None
            self.finish_t = 0.0
        return self.player

    def new_watch_episode(self, pose=None):
        p = self.ensure_player()
        p.reset(pose=pose)
        self.watch_frame = None
        self.finish_t = 0.0
        self.watch_acc = 0.0

    def get_watch_frame(self):
        p = self.ensure_player()
        if p.finished is not None:
            return p.frames[-1]
        if self.watch_frame is None:
            self.watch_frame = p.current_frame()
        return self.watch_frame

    def watch_trail(self):
        p = self.player
        if p is None:
            return []
        pts = [(f["x"], f["y"]) for f in p.frames[-70:]]
        if p.finished is None:
            s = p.state()
            pts.append((s["x"], s["y"]))
        return pts

    def save_checkpoint(self, manual=True):
        tr = self.trainer
        try:
            if manual:
                d = os.path.dirname(os.path.abspath(self.checkpoint_path))
                path = os.path.join(d, "ckpt_%s_%dk.pkl" % (time.strftime("%Y%m%d_%H%M%S"), tr.total_steps // 1000))
            else:
                path = self.checkpoint_path
            tr.save(path)
            self.toast(("Saved checkpoint " if manual else "Autosaved checkpoint ") + os.path.basename(path), T.GOOD)
        except Exception as ex:
            self.toast("Checkpoint save failed: %s" % ex, T.BAD)

    def open_ckpt_list(self):
        d = os.path.dirname(os.path.abspath(self.checkpoint_path))
        files = sorted(glob.glob(os.path.join(d, "*.pkl")), key=os.path.getmtime, reverse=True)
        items = []
        for f in files:
            items.append((os.path.basename(f), "%s  %.1f MB" % (time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(f))),
                                                               os.path.getsize(f) / 1e6), f))
        self.popup = ("ckpt", ListPopup("Load checkpoint", items, "No checkpoints saved yet. Use Save."))

    def open_lot_list(self):
        d = os.path.dirname(os.path.abspath(self.lot_path))
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        items = [(os.path.basename(f), time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(f))), f) for f in files]
        self.popup = ("lot", ListPopup("Load lot", items, "No lots found"))

    def load_checkpoint(self, path):
        try:
            new = Trainer.load(path)
        except Exception as ex:
            self.toast("Load failed: %s" % ex, T.BAD)
            return
        self.editor = None
        self.trainer = new
        self.player = None
        self.rp_ep = None
        self.make_panels()
        self.last_save_tt = new.train_time
        self.mode = TRAIN
        self._enter(TRAIN)
        self.training = False
        self.toast("Loaded %s (%d steps). Press Space to resume training" % (os.path.basename(path), new.total_steps),
                   T.GOOD, 6.0)

    def shutdown(self):
        try:
            if self.mode == BUILD:
                self.leave_build()
            if self.trainer.total_steps > 0:
                self.trainer.save(self.checkpoint_path)
        except Exception as ex:
            print("autosave failed: %s" % ex, file=sys.stderr)
        pygame.quit()

    def request_quit(self, force=False):
        now = time.time()
        if force or now < self.quit_armed:
            self.running = False
        else:
            self.quit_armed = now + 2.0
            self.toast("Press Esc again to quit (the checkpoint is autosaved)", T.WARN, 2.0)

    def handle_event(self, e):
        if e.type == pygame.QUIT:
            self.running = False
            return
        self.ctl.feed(e)
        if self.popup is not None:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.popup = None
            return
        if self.show_help:
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_F1):
                self.show_help = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self.show_help = False
            return
        if self.charts is not None and self.charts_event(e):
            return
        if e.type == pygame.KEYDOWN:
            self.on_key(e)
            return
        self.on_mouse(e)

    def charts_event(self, e):
        r = self.charts_rect
        pos = getattr(e, "pos", None) if e.type != pygame.MOUSEWHEEL else self.ctl.mouse
        if e.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL):
            return False
        if pos is None or not (r.collidepoint(pos) or self.capture == "charts"):
            return False
        if e.type == pygame.MOUSEBUTTONDOWN:
            self.capture = "charts"
        elif e.type == pygame.MOUSEBUTTONUP:
            self.capture = None if self.capture == "charts" else self.capture
        try:
            return bool(self.charts.handle_event(e))
        except Exception:
            return False

    def on_key(self, e):
        k = e.key
        ctrl = e.mod & pygame.KMOD_CTRL
        if k == pygame.K_F1 or (k == pygame.K_h and not ctrl):
            self.show_help = True
            return
        if k in (pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5):
            self.set_mode(k - pygame.K_F2)
            return
        if k == pygame.K_TAB:
            self.set_mode((self.mode + 1) % 4)
            return
        if ctrl and k == pygame.K_s and self.mode != BUILD:
            self.save_checkpoint()
            return
        if ctrl and k in (pygame.K_o, pygame.K_l):
            self.open_lot_list() if self.mode == BUILD else self.open_ckpt_list()
            return
        if self.mode == BUILD:
            if self.editor.key(e):
                return
            if k == pygame.K_ESCAPE:
                self.request_quit()
            elif k == pygame.K_f:
                self.view.fit(self.editor.world)
            elif k == pygame.K_o:
                self.view.show_spawns = not self.view.show_spawns
            return
        v = self.view
        toggles = {pygame.K_l: "show_rays", pygame.K_t: "show_trail", pygame.K_v: "show_value",
                   pygame.K_p: "show_policy", pygame.K_o: "show_spawns"}
        if self.mode == TRAIN:
            toggles[pygame.K_g] = "show_ghosts"
        if k in toggles:
            setattr(v, toggles[k], not getattr(v, toggles[k]))
            return
        if k == pygame.K_f:
            v.fit(self.current_world())
            return
        if k == pygame.K_ESCAPE:
            self.request_quit()
        elif self.mode == TRAIN:
            if k == pygame.K_SPACE:
                self.training = not self.training
            elif pygame.K_1 <= k <= pygame.K_4:
                self.train_speed = k - pygame.K_1
        elif self.mode == WATCH:
            p = self.ensure_player()
            if k == pygame.K_SPACE:
                self.watch_paused = not self.watch_paused
            elif k == pygame.K_s:
                p.deterministic = not p.deterministic
                self.watch_frame = None
                self.toast("Actions are now %s" % ("deterministic (best action)" if p.deterministic else "sampled (stochastic)"))
            elif k == pygame.K_n:
                self.new_watch_episode()
            elif k == pygame.K_r:
                pose = self.start_pose()
                self.new_watch_episode(pose)
            elif k == pygame.K_a:
                self.auto_repeat = not self.auto_repeat
            elif pygame.K_1 <= k <= pygame.K_4:
                self.watch_speed = k - pygame.K_1
        elif self.mode == REPLAY:
            self.replay_key(k)

    def start_pose(self):
        p = self.player
        if p is not None and p.frames:
            f = p.frames[0]
            return (f["x"], f["y"], f["angle"])
        if p is not None:
            s = p.state()
            return (s["x"], s["y"], s["angle"])
        return None

    def replay_key(self, k):
        ep = self.rp_ep
        items = self.replay_items()
        n = len(ep["frames"]) if ep else 0
        if k == pygame.K_SPACE:
            if not self.rp_playing and self.rp_i >= n - 1:
                self.rp_i = 0
            self.rp_playing = not self.rp_playing
        elif k == pygame.K_LEFT:
            self.rp_i = max(0, self.rp_i - 1)
            self.rp_playing = False
        elif k == pygame.K_RIGHT:
            self.rp_i = min(max(n - 1, 0), self.rp_i + 1)
            self.rp_playing = False
        elif k == pygame.K_HOME:
            self.rp_i = 0
        elif k == pygame.K_END:
            self.rp_i = max(n - 1, 0)
            self.rp_playing = False
        elif k in (pygame.K_UP, pygame.K_DOWN) and items:
            idx = next((i for i, (_, e) in enumerate(items) if e is ep), 0)
            idx = max(0, min(len(items) - 1, idx + (1 if k == pygame.K_DOWN else -1)))
            self.select_replay(items[idx][1])
        elif pygame.K_1 <= k <= pygame.K_4:
            self.replay_speed = k - pygame.K_1
        elif k == pygame.K_ESCAPE:
            self.request_quit()

    def select_replay(self, ep):
        self.rp_ep = ep
        self.rp_i = 0
        self.rp_playing = True
        self.rp_acc = 0.0

    def on_mouse(self, e):
        v = self.view
        mouse = self.ctl.mouse
        mods = self.mods()
        if e.type == pygame.MOUSEWHEEL:
            if self.mode == BUILD and mods & pygame.KMOD_SHIFT and v.rect.collidepoint(mouse):
                self.editor.wheel(e.y, mods)
                return
            if self.mode == REPLAY and self.list_rect.collidepoint(mouse):
                self.rp_scroll = max(0, self.rp_scroll - e.y)
                return
            if v.rect.collidepoint(mouse):
                v.handle_event(e, mouse)
            return
        if v.handle_event(e, mouse):
            return
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.canvas.collidepoint(e.pos):
            self.capture = "lot"
            if self.mode == BUILD:
                self.editor.mouse_down(e.pos, v, mods)
            elif self.mode == WATCH:
                wx, wy = v.s2w(*e.pos)
                self.place = {"start": (wx, wy), "angle": None}
        elif e.type == pygame.MOUSEMOTION and self.capture == "lot":
            if self.mode == BUILD:
                self.editor.mouse_move(e.pos, v, mods, True)
            elif self.mode == WATCH and self.place is not None:
                wx, wy = v.s2w(*e.pos)
                sx, sy = self.place["start"]
                if math.hypot(wx - sx, wy - sy) > 0.5:
                    self.place["angle"] = math.atan2(wy - sy, wx - sx)
        elif e.type == pygame.MOUSEMOTION and self.mode == BUILD and self.canvas.collidepoint(e.pos):
            self.editor.mouse_move(e.pos, v, mods, False)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.capture == "lot":
            self.capture = None
            if self.mode == BUILD:
                self.editor.mouse_up(e.pos, v, mods)
            elif self.mode == WATCH and self.place is not None:
                self.finish_place()

    def finish_place(self):
        sx, sy = self.place["start"]
        ang = self.place["angle"]
        self.place = None
        p = self.ensure_player()
        if ang is None:
            slot = p.env.slot[0]
            ang = math.atan2(slot[1] - sy, slot[0] - sx)
        w = self.trainer.world
        if not (0 <= sx <= w.width and 0 <= sy <= w.height):
            self.toast("Start pose is outside the lot", T.WARN)
            return
        self.new_watch_episode((sx, sy, ang))
        if p.env.pose_collides(sx, sy, ang):
            self.toast("That start pose overlaps something, the run ends immediately", T.WARN)

    def update(self, dt):
        if self.popup is not None or self.show_help:
            return
        m = self.mode
        try:
            if m == TRAIN:
                self.update_train(dt)
            elif m == WATCH:
                self.update_watch(dt)
            elif m == REPLAY:
                self.update_replay(dt)
        except Exception as ex:
            traceback.print_exc()
            self.training = False
            self.toast("Error: %s (training paused)" % ex, T.BAD, 8.0)

    def update_train(self, dt):
        tr = self.trainer
        if self.training:
            sp = self.train_speed
            if sp == 0:
                if tr.updating:
                    tr.run(0.02, max_steps=0)
                self.rt_acc = min(self.rt_acc + dt * 10.0, 4.0)
                n = min(int(self.rt_acc), 3)
                self.rt_acc -= n
                for _ in range(n):
                    tr.run(0.03, max_steps=1)
            elif sp == 1:
                tr.run(0.03)
            elif sp == 2:
                tr.run(0.12)
            else:
                tr.run(0.1)
            if tr.train_time - self.last_save_tt >= AUTOSAVE_SECONDS:
                self.last_save_tt = tr.train_time
                self.save_checkpoint(manual=False)
        ep = tr.recent_episodes[-1] if tr.recent_episodes else None
        if ep is not self._ep_seen:
            self._ep_seen = ep
            if ep is not None:
                f, s = ep["frames"][-1], ep["summary"]
                if s["success"]:
                    self.view.add_effect(f["x"], f["y"], f["angle"], "ok")
                elif s["collision"]:
                    col = T.KIND_COLORS.get(s["collision"], T.BAD)
                    self.view.add_effect(f["x"], f["y"], f["angle"], "hit", T.BAD if col == T.LOT_EDGE else col)
        self.trail = [(f["x"], f["y"]) for f in tr._frames[-60:]]

    def update_watch(self, dt):
        p = self.ensure_player()
        if p.finished is not None:
            if self.auto_repeat:
                self.finish_t += dt
                if self.finish_t > 2.0:
                    self.new_watch_episode()
            return
        if self.watch_paused or self.place is not None:
            return
        self.watch_acc += dt * 10.0 * WATCH_SPEEDS[self.watch_speed]
        n = min(int(self.watch_acc), 40)
        self.watch_acc -= int(self.watch_acc)
        for _ in range(n):
            p.step()
            self.watch_frame = None
            if p.finished is not None:
                self.finish_t = 0.0
                break

    def update_replay(self, dt):
        ep = self.rp_ep
        if ep is None:
            return
        n = len(ep["frames"])
        if self.rp_playing:
            self.rp_acc += dt * 10.0 * REPLAY_SPEEDS[self.replay_speed]
            k = int(self.rp_acc)
            self.rp_acc -= k
            self.rp_i = min(n - 1, self.rp_i + k)
            if self.rp_i >= n - 1:
                self.rp_playing = False
        self.rp_i = max(0, min(n - 1, self.rp_i))

    def step(self, events=None):
        now = time.perf_counter()
        dt = min(0.1, now - self.last_t)
        self.last_t = now
        self.ctl.new_frame()
        evs = pygame.event.get() if events is None else events
        for e in evs:
            self.handle_event(e)
        self.update(dt)
        heavy = self.mode == TRAIN and self.training and self.train_speed == 3
        pressed = any(e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.KEYDOWN, pygame.MOUSEWHEEL) for e in evs)
        if not heavy or pressed or now - self.last_draw >= 0.33:
            self.draw()
            pygame.display.flip()
            self.last_draw = now
        self.clock.tick(60)
        self.fps = self.clock.get_fps()

    def draw(self):
        self.layout()
        surf = self.screen
        ctl = self.ctl
        saved_click = ctl.click
        modal = self.popup is not None or self.show_help
        if modal:
            ctl.click = None
        surf.fill(T.BG)
        self.draw_top(surf)
        self.draw_center(surf)
        self.draw_right(surf)
        self.draw_charts(surf)
        self.draw_status(surf)
        ctl.tooltip(surf)
        if modal:
            ctl.click = saved_click
        if self.popup is not None:
            res = self.popup[1].draw(surf, ctl)
            if res == "close":
                self.popup = None
            elif res is not None:
                kind = self.popup[0]
                self.popup = None
                if kind == "ckpt":
                    self.load_checkpoint(res)
                else:
                    self.editor.load(res)
        elif self.show_help:
            self.draw_help(surf)

    def get_summary(self):
        now = time.time()
        if self.summary is None or now - self.summary_t > 0.2:
            self.summary = self.trainer.summary()
            self.summary_t = now
        return self.summary

    def draw_top(self, surf):
        ctl = self.ctl
        pygame.draw.rect(surf, T.PANEL, self.top)
        pygame.draw.line(surf, T.BORDER, (0, TOP_H - 1), (self.top.w, TOP_H - 1))
        T.text(surf, "ParkingAI", (14, 20), 24, T.TEXT, bold=True)
        mode = ctl.tabs(surf, (140, 8, 4 * 96, 32), MODE_NAMES, self.mode)
        if mode != self.mode:
            self.set_mode(mode)
        x = 140 + 4 * 96 + 16
        playing = self.mode == TRAIN and self.training
        if ctl.button(surf, (x, 8, 96, 32), "Pause" if playing else "Train",
                      color=(150, 110, 30) if playing else (40, 110, 70), tip="Space (TRAIN mode)"):
            if self.mode != TRAIN:
                self.set_mode(TRAIN)
            else:
                self.training = not self.training
        x += 108
        if self.mode == TRAIN:
            labels, cur = TRAIN_SPEEDS, self.train_speed
        elif self.mode == WATCH:
            labels, cur = ["%dx" % s for s in WATCH_SPEEDS], self.watch_speed
        elif self.mode == REPLAY:
            labels, cur = ["%gx" % s for s in REPLAY_SPEEDS], self.replay_speed
        else:
            labels, cur = None, 0
        if labels:
            T.text(surf, "Speed", (x, 24), 14, T.TEXT_DIM, anchor="midleft")
            x += 46
            for i, lab in enumerate(labels):
                w = 24 + 8 * len(lab)
                if ctl.button(surf, (x, 8, w, 32), lab, active=i == cur, tip="key %d" % (i + 1)):
                    if self.mode == TRAIN:
                        self.train_speed = i
                    elif self.mode == WATCH:
                        self.watch_speed = i
                    else:
                        self.replay_speed = i
                x += w + 4
        x += 12
        if ctl.button(surf, (x, 8, 72, 32), "Save", tip="Save checkpoint (Ctrl+S)"):
            self.save_checkpoint()
        if ctl.button(surf, (x + 78, 8, 72, 32), "Load", tip="Load checkpoint (Ctrl+O)"):
            self.open_ckpt_list()
        if ctl.button(surf, (self.top.w - 44, 8, 34, 32), "?", tip="Help (H)", size=18):
            self.show_help = True
        s = self.get_summary()
        sr, cr = s["success_rate"], s["collision_rate"]
        parts = [("steps", "{:,}".format(s["steps"]), T.TEXT), ("episodes", "{:,}".format(s["episodes"]), T.TEXT),
                 ("success", "%d%%" % round(sr * 100), T.GOOD if sr > 0.5 else (T.WARN if sr > 0.15 else T.TEXT)),
                 ("collisions", "%d%%" % round(cr * 100), T.BAD if cr > 0.5 else T.TEXT),
                 ("difficulty", "%.1f" % s["difficulty"], T.TEXT), ("steps/s", "{:,.0f}".format(s["sps"]), T.TEXT),
                 ("agent", s["agent"], T.ACCENT)]
        x = 14
        for lab, val, col in parts:
            r = T.text(surf, lab + " ", (x, 53), 14, T.TEXT_DIM)
            r2 = T.text(surf, val, (r.right, 53), 14, col, bold=True)
            x = r2.right + 22
        state = "TRAINING" if (self.mode == TRAIN and self.training) else "paused"
        T.text(surf, state, (self.top.w - 14, 53), 14, T.GOOD if state == "TRAINING" else T.TEXT_DIM, bold=True,
               anchor="topright")

    def draw_center(self, surf):
        ctl, v = self.ctl, self.view
        world = self.current_world()
        v.set_rect(self.canvas, world)
        pygame.draw.rect(surf, mix(T.PANEL, T.BG, 0.4), self.bar_rect)
        self.draw_view_bar(surf)
        m = self.mode
        if m == BUILD:
            self.editor.draw_toolbar(surf, self.tool_rect, ctl)
            v.draw(surf, world, None, rev=self.editor.rev)
            clip = surf.get_clip()
            surf.set_clip(self.canvas)
            self.editor.draw_overlay(surf, v, ctl.mouse)
            surf.set_clip(clip)
            self.draw_prop_line(surf)
        elif m == TRAIN:
            f = self.trainer.live
            v.draw(surf, world, f, trail=self.trail, env=self.trainer.env, agent=self.trainer.agent,
                   ghosts=self.trainer.env)
            if f is not None:
                v.draw_hud(surf, f, ["ep step %d" % f["step"]])
            else:
                T.text(surf, "Press Train to start. Cars appear here.", self.canvas.center, 18, T.TEXT_DIM, anchor="center")
        elif m == WATCH:
            self.draw_watch(surf)
        else:
            self.draw_replay(surf)
        pygame.draw.rect(surf, T.BORDER, self.canvas, 1)

    def draw_view_bar(self, surf):
        ctl, v = self.ctl, self.view
        r = self.bar_rect
        specs = {
            BUILD: [("show_spawns", "Spawns (O)")],
            TRAIN: [("show_rays", "Rays (L)"), ("show_trail", "Trail (T)"), ("show_ghosts", "Ghost cars (G)"),
                    ("show_value", "Value map (V)"), ("show_policy", "Policy arrows (P)"), ("show_spawns", "Spawns (O)")],
            WATCH: [("show_rays", "Rays (L)"), ("show_trail", "Trail (T)"), ("show_value", "Value map (V)"),
                    ("show_policy", "Policy arrows (P)"), ("show_spawns", "Spawns (O)")],
            REPLAY: [("show_rays", "Rays (L)"), ("show_trail", "Trail (T)"), ("show_spawns", "Spawns (O)")],
        }[self.mode]
        x = r.x + 8
        for attr, lab in specs:
            w = 40 + 7 * len(lab)
            tip = None
            if attr == "show_value":
                tip = "Heatmap of the critic's value for the car at each spot (current heading)"
            elif attr == "show_policy":
                tip = "Arrow = what the agent would do from each spot (white forward, orange reverse, dot brake)"
            setattr(v, attr, ctl.toggle(surf, (x, r.y + 3, w, r.h - 6), lab, getattr(v, attr), tip=tip))
            x += w + 4
        if self.mode == WATCH:
            p = self.ensure_player()
            w = 150
            det = ctl.toggle(surf, (x, r.y + 3, w, r.h - 6), "Deterministic (S)", p.deterministic)
            if det != p.deterministic:
                p.deterministic = det
                self.watch_frame = None
            x += w + 4
            self.auto_repeat = ctl.toggle(surf, (x, r.y + 3, 130, r.h - 6), "Auto-repeat (A)", self.auto_repeat)
            x += 134
        if self.mode == BUILD:
            return
        if ctl.button(surf, (r.right - 62, r.y + 3, 54, r.h - 6), "Fit", tip="Fit lot to view (F)"):
            v.fit(self.current_world())

    def draw_prop_line(self, surf):
        r = self.bottom_rect
        pygame.draw.rect(surf, T.PANEL, r)
        ed = self.editor
        d = ed.describe()
        T.text(surf, d if d else ed.hint(), (r.x + 10, r.centery), 14, T.TEXT if d else T.TEXT_DIM, anchor="midleft")
        if ed.snap_on:
            T.text(surf, "snap 0.25 m", (r.right - 10, r.centery), 13, T.TEXT_DIM, anchor="midright")

    def draw_watch(self, surf):
        v, p = self.view, self.ensure_player()
        f = self.get_watch_frame()
        outcome = banner = None
        if p.finished is not None:
            outcome, title, col, sub = outcome_of(p.finished["summary"])
            banner = (title, col, sub)
        v.draw(surf, self.trainer.world, f, trail=self.watch_trail(), env=p.env, agent=self.trainer.agent, outcome=outcome)
        v.draw_hud(surf, f, ["t %.1f s" % f["t"]])
        if self.place is not None:
            sx, sy = self.place["start"]
            ang = self.place["angle"]
            c = v.w2s(sx, sy)
            if ang is None:
                pygame.draw.circle(surf, T.SPAWN, (int(c[0]), int(c[1])), 8, 2)
            else:
                v.blit_car(surf, sx, sy, ang, 4.5, 1.9, T.AGENT_CAR, 140)
                arrow(surf, T.SPAWN, c, v.w2s(sx + math.cos(ang) * 3.2, sy + math.sin(ang) * 3.2), 12, 3)
        if banner:
            if self.auto_repeat:
                banner = (banner[0], banner[1], banner[2] + "   (restarting in %.0f s)" % max(0, 2.0 - self.finish_t))
            v.banner(surf, banner[0], banner[1], banner[2])
        elif self.watch_paused:
            v.banner(surf, "Paused", T.WARN)
        elif p.frames == [] and self.place is None:
            T.text(surf, "Drag in the lot to choose a start pose", (v.rect.centerx, v.rect.bottom - 40), 15, T.TEXT_DIM,
                   anchor="center")

    def replay_frame(self):
        ep = self.rp_ep
        if ep is None:
            return None
        return ep["frames"][max(0, min(self.rp_i, len(ep["frames"]) - 1))]

    def draw_replay(self, surf):
        ctl, v = self.ctl, self.view
        self.draw_replay_list(surf)
        ep = self.rp_ep
        if ep is None:
            pygame.draw.rect(surf, T.ASPHALT, self.canvas)
            T.text(surf, "No recorded episodes yet", (self.canvas.centerx, self.canvas.centery - 10), 20, T.TEXT,
                   anchor="center")
            T.text(surf, "Train for a bit, then come back to replay episodes of training car 0", (self.canvas.centerx, self.canvas.centery + 16),
                   14, T.TEXT_DIM, anchor="center")
            return
        frames = ep["frames"]
        i = max(0, min(self.rp_i, len(frames) - 1))
        f = frames[i]
        last = i == len(frames) - 1
        outcome = None
        if last:
            outcome, title, col, sub = outcome_of(ep["summary"])
        trail = [(g["x"], g["y"]) for g in frames[max(0, i - 60):i + 1]]
        v.draw(surf, ep["world"], f, trail=trail, outcome=outcome)
        v.draw_hud(surf, f, ["t %.1f s" % f["t"]])
        if last:
            v.banner(surf, title, col, sub)
        self.draw_scrubber(surf, ep, i)

    def draw_replay_list(self, surf):
        ctl = self.ctl
        r = self.list_rect
        pygame.draw.rect(surf, T.PANEL, r)
        pygame.draw.line(surf, T.BORDER, r.topright, r.bottomright)
        T.text(surf, "Episodes of car 0", (r.x + 12, r.y + 9), 15, T.TEXT, bold=True)
        T.text(surf, "newest first", (r.right - 12, r.y + 11), 12, T.TEXT_DIM, anchor="topright")
        items = self.replay_items()
        rows = (r.h - 40) // 44
        self.rp_scroll = max(0, min(self.rp_scroll, max(0, len(items) - rows)))
        if not items:
            T.text(surf, "none yet", (r.x + 12, r.y + 46), 14, T.TEXT_DIM)
        for k in range(rows):
            idx = self.rp_scroll + k
            if idx >= len(items):
                break
            tag, ep = items[idx]
            s = ep["summary"]
            row = pygame.Rect(r.x + 6, r.y + 38 + k * 44, r.w - 12, 41)
            sel = ep is self.rp_ep
            hov = row.collidepoint(ctl.mouse)
            pygame.draw.rect(surf, T.PANEL_LIGHT if sel else (mix(T.PANEL, T.BORDER, 0.5) if hov else T.PANEL), row, border_radius=5)
            if sel:
                pygame.draw.rect(surf, T.ACCENT, row, 1, border_radius=5)
            self.outcome_icon(surf, (row.x + 18, row.centery), s)
            T.text(surf, ("BEST  " if tag else "") + "#%d" % s["episode"], (row.x + 38, row.y + 6), 15,
                   T.WARN if tag else T.TEXT, bold=True)
            T.text(surf, "%+.2f" % s["return"], (row.right - 8, row.y + 6), 15, T.GOOD if s["return"] > 0 else T.TEXT_DIM,
                   anchor="topright")
            if s["success"]:
                sub = "parked, %.1f deg, %d steps" % (s["angle_err"], s["length"])
            elif s["collision"]:
                sub = "hit %s, %d steps" % (s["collision"].replace("_", " "), s["length"])
            else:
                sub = "timeout, inside %d%%" % round(s["inside"] * 100)
            T.text(surf, sub, (row.x + 38, row.y + 23), 12, T.TEXT_DIM)
            if ctl.hit(row):
                self.select_replay(ep)

    def outcome_icon(self, surf, c, s):
        if s["success"]:
            pygame.draw.circle(surf, T.GOOD, c, 10)
            pygame.draw.lines(surf, (12, 16, 24), False, [(c[0] - 5, c[1]), (c[0] - 1, c[1] + 4), (c[0] + 5, c[1] - 4)], 2)
        elif s["collision"]:
            col = T.KIND_COLORS.get(s["collision"], T.BAD)
            col = T.BAD if col in (T.LOT_EDGE, T.OBSTACLE, T.PARKED_CAR) else col
            pygame.draw.circle(surf, col, c, 10)
            pygame.draw.line(surf, (12, 16, 24), (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4), 2)
            pygame.draw.line(surf, (12, 16, 24), (c[0] - 4, c[1] + 4), (c[0] + 4, c[1] - 4), 2)
        else:
            pygame.draw.circle(surf, T.WARN, c, 10)
            pygame.draw.line(surf, (12, 16, 24), c, (c[0], c[1] - 6), 2)
            pygame.draw.line(surf, (12, 16, 24), c, (c[0] + 4, c[1]), 2)

    def draw_scrubber(self, surf, ep, i):
        ctl = self.ctl
        r = self.bottom_rect
        pygame.draw.rect(surf, T.PANEL, r)
        n = len(ep["frames"])
        sl = pygame.Rect(r.x + 16, r.y + 6, r.w - 32, 20)
        val, changed = ctl.slider(surf, sl, "scrub", float(i), 0.0, float(max(n - 1, 1)))
        if changed:
            self.rp_i = int(round(val))
            self.rp_playing = False
        y = r.y + 30
        x = r.x + 16
        playing = self.rp_playing
        for lab, tip, fn in (("|<", "Start (Home)", lambda: setattr(self, "rp_i", 0)),
                             ("<", "Previous frame (Left)", lambda: self.replay_key(pygame.K_LEFT)),
                             ("Pause" if playing else "Play", "Space", lambda: self.replay_key(pygame.K_SPACE)),
                             (">", "Next frame (Right)", lambda: self.replay_key(pygame.K_RIGHT)),
                             (">|", "End", lambda: self.replay_key(pygame.K_END))):
            w = 64 if lab in ("Pause", "Play") else 36
            if ctl.button(surf, (x, y, w, 26), lab, tip=tip):
                fn()
            x += w + 4
        s = ep["summary"]
        f = ep["frames"][i]
        T.text(surf, "frame %d / %d    t %.1f s    return %+.2f    reward %+.3f" % (i + 1, n, f["t"], f["return"], f["reward"]),
               (x + 14, y + 13), 14, T.TEXT, anchor="midleft")
        T.text(surf, "episode #%d  difficulty %.1f  update %d" % (s["episode"], s["difficulty"], s["update"]),
               (r.right - 14, y + 13), 13, T.TEXT_DIM, anchor="midright")

    def draw_right(self, surf):
        r = self.right
        pygame.draw.rect(surf, T.PANEL, r)
        if self.mode == BUILD:
            self.editor.draw_panel(surf, r, self.ctl, self)
            return
        if self.mode == TRAIN:
            f = self.trainer.live
            frames = self.trainer._frames
        elif self.mode == WATCH:
            f = self.get_watch_frame()
            frames = self.player.frames if self.player else []
        else:
            f = self.replay_frame()
            frames = self.rp_ep["frames"][:self.rp_i + 1] if self.rp_ep else []
        if f is None:
            T.text(surf, "Waiting for the first step...", r.center, 15, T.TEXT_DIM, anchor="center")
            return
        now = time.time()
        if f is not self.insp_frame:
            try:
                self.insp = self.trainer.agent.inspect(f["obs"])
            except Exception:
                self.insp = None
            self.insp_frame = f
            self.insp_t = now
        if self.inspector is None or self.panel_blocked("inspector"):
            self.fallback_inspector(surf, r, f, self.panel_err.get("inspector", (0, ""))[1])
            return
        try:
            self.inspector.draw(surf, r, f, inspect=self.insp, episode_frames=frames)
        except Exception as ex:
            self.panel_failed("inspector", ex)
            self.fallback_inspector(surf, r, f, "Inspector error: %s" % ex)

    def panel_blocked(self, name):
        e = self.panel_err.get(name)
        return e is not None and time.time() - e[0] < 3.0

    def panel_failed(self, name, ex):
        if name not in self.panel_err:
            traceback.print_exc()
        self.panel_err[name] = (time.time(), "%s error: %s" % (name, ex))

    def fallback_inspector(self, surf, r, f, msg):
        T.text(surf, "Inspector unavailable" if InspectorPanel is None else msg or "Inspector unavailable",
               (r.x + 16, r.y + 16), 15, T.WARN)
        T.text(surf, "Action probabilities", (r.x + 16, r.y + 52), 14, T.TEXT_DIM)
        probs = f.get("probs")
        for k in range(len(ACTION_NAMES)):
            y = r.y + 76 + k * 22
            p = float(probs[k]) if probs is not None else 0.0
            T.text(surf, ACTION_NAMES[k], (r.x + 16, y), 13, T.TEXT)
            pygame.draw.rect(surf, T.BG, (r.x + 150, y + 2, 200, 12))
            pygame.draw.rect(surf, T.ACCENT, (r.x + 150, y + 2, int(200 * p), 12))
        T.text(surf, "value %.2f   reward %+.3f" % (f.get("value", 0.0), f.get("reward", 0.0)), (r.x + 16, r.y + 290), 14, T.TEXT)

    def draw_charts(self, surf):
        r = self.charts_rect
        pygame.draw.rect(surf, T.PANEL, r)
        pygame.draw.line(surf, T.BORDER, r.topleft, r.topright)
        e = self.panel_err.get("charts")
        if self.charts is None or (e is not None and time.time() - e[0] < 3.0):
            T.text(surf, e[1] if e else "Charts unavailable (parkai/ui/panels.py not found)", r.center, 15, T.WARN, anchor="center")
            return
        try:
            self.charts.draw(surf, r, self.trainer.history)
        except Exception as ex:
            self.panel_failed("charts", ex)

    def draw_status(self, surf):
        r = self.status
        pygame.draw.rect(surf, T.PANEL, r)
        now = time.time()
        if self.toast_msg and now < self.toast_msg[2]:
            msg, col = self.toast_msg[0], self.toast_msg[1]
        else:
            msg, col = {
                BUILD: self.editor.hint() if self.editor else "",
                TRAIN: "Space pause/resume   1-4 speed   V value map   P policy arrows   G ghost cars   H help",
                WATCH: "Drag in the lot to place a start pose   N new   R restart   S stochastic   Space pause   H help",
                REPLAY: "Click an episode   Space play/pause   Left/Right step   Home/End   H help",
            }[self.mode], T.TEXT_DIM
        T.text(surf, msg, (12, r.centery), 14, col, anchor="midleft")
        T.text(surf, "%d fps" % round(self.fps), (r.right - 12, r.centery), 13, T.TEXT_DIM, anchor="midright")

    def draw_help(self, surf):
        sw, sh = surf.get_size()
        dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 190))
        surf.blit(dim, (0, 0))
        box = pygame.Rect(0, 0, min(sw - 40, 1100), min(sh - 40, 640))
        box.center = (sw // 2, sh // 2)
        pygame.draw.rect(surf, T.PANEL, box, border_radius=10)
        pygame.draw.rect(surf, T.BORDER, box, 1, border_radius=10)
        T.text(surf, "Keyboard and mouse", (box.x + 24, box.y + 16), 22, T.TEXT, bold=True)
        T.text(surf, "H, F1 or Esc to close", (box.right - 24, box.y + 22), 14, T.TEXT_DIM, anchor="topright")
        cols = [[0, 1], [2, 3, 4]]
        colw = (box.w - 48) // 2
        for ci, secs in enumerate(cols):
            y = box.y + 60
            for si in secs:
                title, rows = HELP[si]
                T.text(surf, title, (box.x + 24 + ci * colw, y), 17, T.ACCENT, bold=True)
                y += 26
                for keys, desc in rows:
                    T.text(surf, keys, (box.x + 24 + ci * colw, y), 14, T.TEXT, bold=True)
                    T.text(surf, desc, (box.x + 24 + ci * colw + 190, y), 14, T.TEXT_DIM)
                    y += 21
                y += 10


def run_app(trainer, lot_path, checkpoint_path):
    max_seconds = float(os.environ.get("PARKAI_MAX_SECONDS", "0") or 0)
    app = App(trainer, lot_path, checkpoint_path)
    start = time.time()
    try:
        while app.running:
            app.step()
            if max_seconds and time.time() - start > max_seconds:
                break
    finally:
        app.shutdown()
