import pygame

BG = (18, 20, 26)
PANEL = (28, 31, 40)
PANEL_LIGHT = (38, 42, 54)
BORDER = (58, 63, 78)
TEXT = (225, 228, 235)
TEXT_DIM = (140, 146, 160)
ACCENT = (86, 156, 255)
GOOD = (80, 200, 120)
BAD = (235, 90, 80)
WARN = (240, 180, 60)

ASPHALT = (54, 57, 64)
LOT_EDGE = (90, 94, 104)
SLOT_LINE = (230, 230, 230)
TARGET_SLOT = (80, 200, 120)
OBSTACLE = (150, 110, 70)
PARKED_CAR = (120, 130, 150)
AGENT_CAR = (86, 156, 255)
MOVING_CAR = (240, 150, 60)
PEDESTRIAN = (240, 220, 80)
SPAWN = (180, 120, 255)
RAY = (120, 220, 255)
RAY_MOVING = (255, 120, 120)

SERIES = [(86, 156, 255), (80, 200, 120), (240, 180, 60), (235, 90, 80), (180, 120, 255), (90, 210, 210)]

KIND_COLORS = {"wall": LOT_EDGE, "obstacle": OBSTACLE, "parked_car": PARKED_CAR,
               "moving_car": MOVING_CAR, "pedestrian": PEDESTRIAN}

_fonts = {}


def font(size=14, bold=False):
    key = (size, bold)
    if key not in _fonts:
        f = pygame.font.Font(None, int(size * 1.35))
        f.set_bold(bold)
        _fonts[key] = f
    return _fonts[key]


_text_cache = {}


def text(surface, s, pos, size=14, color=TEXT, bold=False, anchor="topleft"):
    key = (str(s), size, tuple(color), bold)
    img = _text_cache.get(key)
    if img is None:
        if len(_text_cache) > 3000:
            _text_cache.clear()
        img = _text_cache[key] = font(size, bold).render(key[0], True, color)
    r = img.get_rect(**{anchor: pos})
    surface.blit(img, r)
    return r
