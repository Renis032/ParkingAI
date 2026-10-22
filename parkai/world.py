import copy
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np

from . import geometry as geo

WALL, OBSTACLE, PARKED_CAR, MOVING_CAR, PEDESTRIAN = 0, 1, 2, 3, 4
KIND_NAMES = ["wall", "obstacle", "parked_car", "moving_car", "pedestrian"]
MOVING_KINDS = {"car": MOVING_CAR, "pedestrian": PEDESTRIAN}

CAR_LENGTH = 4.5
CAR_WIDTH = 1.9
PEDESTRIAN_SIZE = 0.6


@dataclass
class Slot:
    x: float
    y: float
    angle: float
    length: float = 5.5
    width: float = 2.7
    target: bool = True


@dataclass
class ParkedCar:
    x: float
    y: float
    angle: float
    length: float = CAR_LENGTH
    width: float = CAR_WIDTH
    color: List[int] = field(default_factory=lambda: [120, 130, 150])


@dataclass
class Obstacle:
    x: float
    y: float
    angle: float
    length: float
    width: float


@dataclass
class Spawn:
    x: float
    y: float
    angle: float
    jitter_pos: float = 1.5
    jitter_angle: float = 0.35


@dataclass
class MovingObject:
    kind: str
    path: List[List[float]]
    speed: float = 1.5
    closed: bool = False
    phase: float = 0.0
    random_phase: bool = True
    length: float = 0.0
    width: float = 0.0

    def size(self):
        if self.length > 0 and self.width > 0:
            return self.length, self.width
        if self.kind == "pedestrian":
            return PEDESTRIAN_SIZE, PEDESTRIAN_SIZE
        return CAR_LENGTH, CAR_WIDTH

    def kind_code(self):
        return MOVING_KINDS.get(self.kind, MOVING_CAR)


@dataclass
class World:
    name: str = "untitled"
    width: float = 40.0
    height: float = 30.0
    slots: List[Slot] = field(default_factory=list)
    parked_cars: List[ParkedCar] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    spawns: List[Spawn] = field(default_factory=list)
    moving: List[MovingObject] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return World(
            name=d.get("name", "untitled"),
            width=float(d.get("width", 40.0)),
            height=float(d.get("height", 30.0)),
            slots=[Slot(**s) for s in d.get("slots", [])],
            parked_cars=[ParkedCar(**p) for p in d.get("parked_cars", [])],
            obstacles=[Obstacle(**o) for o in d.get("obstacles", [])],
            spawns=[Spawn(**s) for s in d.get("spawns", [])],
            moving=[MovingObject(**m) for m in d.get("moving", [])],
        )

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def load(path):
        with open(path) as f:
            return World.from_dict(json.load(f))

    def copy(self):
        return copy.deepcopy(self)

    def slot_occupied(self, slot):
        for p in self.parked_cars:
            if geo.points_in_rects(np.array([p.x, p.y]), slot.x, slot.y, slot.angle, slot.length, slot.width):
                return True
        return False

    def target_slots(self):
        free = [s for s in self.slots if not self.slot_occupied(s)]
        marked = [s for s in free if s.target]
        return marked or free

    def static_rects(self):
        rows = []
        for p in self.parked_cars:
            rows.append((p.x, p.y, p.angle, p.length, p.width, PARKED_CAR))
        for o in self.obstacles:
            rows.append((o.x, o.y, o.angle, o.length, o.width, OBSTACLE))
        if not rows:
            return np.zeros((0, 5)), np.zeros(0, dtype=np.int64)
        arr = np.array(rows, dtype=float)
        return arr[:, :5], arr[:, 5].astype(np.int64)

    def static_segments(self):
        w, h = self.width, self.height
        segs = [[0, 0, w, 0], [w, 0, w, h], [w, h, 0, h], [0, h, 0, 0]]
        kinds = [WALL] * 4
        rects, rkinds = self.static_rects()
        if len(rects):
            edges = geo.poly_edges(geo.rect_corners(*rects.T)).reshape(-1, 4)
            segs.extend(edges.tolist())
            kinds.extend(np.repeat(rkinds, 4).tolist())
        return np.array(segs, dtype=float), np.array(kinds, dtype=np.int64)


def default_world():
    w = World(name="default", width=36.0, height=26.0)
    row_y = 4.0
    slot_w = 2.7
    x0 = 6.0
    colors = [[178, 64, 64], [64, 120, 178], [90, 160, 90], [200, 170, 60], [140, 90, 170], [90, 90, 100]]
    for i in range(9):
        x = x0 + i * slot_w
        target = i == 4
        w.slots.append(Slot(x=x, y=row_y + 2.75, angle=-math.pi / 2, target=target))
        if i in (0, 1, 3, 5, 6, 8):
            w.parked_cars.append(ParkedCar(x=x, y=row_y + 2.85, angle=-math.pi / 2 + (0.03 if i % 2 else -0.03),
                                           color=colors[i % len(colors)]))
    w.obstacles.append(Obstacle(x=3.0, y=19.0, angle=0.0, length=1.0, width=1.0))
    w.obstacles.append(Obstacle(x=33.0, y=19.0, angle=0.0, length=1.0, width=1.0))
    w.obstacles.append(Obstacle(x=18.0, y=24.0, angle=0.0, length=20.0, width=0.6))
    w.spawns.append(Spawn(x=12.0, y=16.0, angle=0.0, jitter_pos=2.0, jitter_angle=0.4))
    return w
