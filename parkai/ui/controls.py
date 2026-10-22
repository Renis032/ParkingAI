import pygame

from . import theme as T


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Controls:
    def __init__(self):
        self.mouse = (0, 0)
        self.down = False
        self.click = None
        self.released = False
        self.wheel = 0
        self.drag_id = None
        self.tip = None

    def new_frame(self):
        self.click = None
        self.released = False
        self.wheel = 0
        self.tip = None

    def feed(self, e):
        if e.type == pygame.MOUSEMOTION:
            self.mouse = e.pos
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.mouse = e.pos
            self.click = e.pos
            self.down = True
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.mouse = e.pos
            self.down = False
            self.released = True
        elif e.type == pygame.MOUSEWHEEL:
            self.wheel += e.y

    def hit(self, rect):
        if self.click is not None and pygame.Rect(rect).collidepoint(self.click):
            self.click = None
            return True
        return False

    def hover(self, rect):
        return pygame.Rect(rect).collidepoint(self.mouse)

    def button(self, surf, rect, label, active=False, enabled=True, tip=None, color=None, size=14):
        rect = pygame.Rect(rect)
        hov = enabled and self.hover(rect)
        base = color or (T.ACCENT if active else T.PANEL_LIGHT)
        bg = mix(base, (255, 255, 255), 0.12) if hov else base
        if not enabled:
            bg = mix(T.PANEL, T.BG, 0.5)
        pygame.draw.rect(surf, bg, rect, border_radius=5)
        pygame.draw.rect(surf, T.ACCENT if active else T.BORDER, rect, 1, border_radius=5)
        fg = T.TEXT if enabled else T.TEXT_DIM
        if active and color is None:
            fg = (12, 16, 24)
        T.text(surf, label, rect.center, size, fg, anchor="center")
        if hov and tip:
            self.tip = (tip, rect)
        return enabled and self.hit(rect)

    def toggle(self, surf, rect, label, value, tip=None, enabled=True):
        rect = pygame.Rect(rect)
        hov = enabled and self.hover(rect)
        bg = mix(T.PANEL_LIGHT, (255, 255, 255), 0.1) if hov else T.PANEL_LIGHT
        pygame.draw.rect(surf, bg, rect, border_radius=5)
        pygame.draw.rect(surf, T.BORDER, rect, 1, border_radius=5)
        box = pygame.Rect(rect.x + 6, rect.centery - 6, 12, 12)
        pygame.draw.rect(surf, T.ACCENT if value else T.BG, box, border_radius=3)
        pygame.draw.rect(surf, T.BORDER, box, 1, border_radius=3)
        if value:
            pygame.draw.lines(surf, (12, 16, 24), False, [(box.x + 2, box.centery), (box.x + 5, box.bottom - 3),
                                                          (box.right - 2, box.y + 3)], 2)
        T.text(surf, label, (box.right + 6, rect.centery), 13, T.TEXT if enabled else T.TEXT_DIM, anchor="midleft")
        if hov and tip:
            self.tip = (tip, rect)
        if enabled and self.hit(rect):
            return not value
        return value

    def tabs(self, surf, rect, labels, current, size=15):
        rect = pygame.Rect(rect)
        w = rect.w // len(labels)
        out = current
        for i, lab in enumerate(labels):
            r = pygame.Rect(rect.x + i * w, rect.y, w - 4, rect.h)
            hov = self.hover(r)
            col = T.ACCENT if i == current else (T.PANEL_LIGHT if not hov else T.BORDER)
            pygame.draw.rect(surf, col, r, border_radius=6)
            T.text(surf, lab, r.center, size, (12, 16, 24) if i == current else T.TEXT, bold=True, anchor="center")
            if self.hit(r):
                out = i
        return out

    def stepper(self, surf, rect, label, value, step, lo, hi, fmt="{:.2f}"):
        rect = pygame.Rect(rect)
        T.text(surf, label, (rect.x, rect.centery), 13, T.TEXT_DIM, anchor="midleft")
        bw = rect.h
        minus = pygame.Rect(rect.right - 150, rect.y, bw, rect.h)
        plus = pygame.Rect(rect.right - bw, rect.y, bw, rect.h)
        T.text(surf, fmt.format(value), ((minus.right + plus.x) // 2, rect.centery), 14, T.TEXT, anchor="center")
        out = value
        if self.button(surf, minus, "-", size=16):
            out = max(lo, value - step)
        if self.button(surf, plus, "+", size=16):
            out = min(hi, value + step)
        return round(out, 6)

    def slider(self, surf, rect, key, value, lo, hi, color=T.ACCENT):
        rect = pygame.Rect(rect)
        track = pygame.Rect(rect.x, rect.centery - 3, rect.w, 6)
        pygame.draw.rect(surf, T.BG, track, border_radius=3)
        if self.click is not None and rect.collidepoint(self.click):
            self.drag_id = key
            self.click = None
        changed = False
        if self.drag_id == key:
            if self.down:
                t = (self.mouse[0] - rect.x) / max(1, rect.w)
                value = lo + max(0.0, min(1.0, t)) * (hi - lo)
                changed = True
            else:
                self.drag_id = None
        t = 0 if hi <= lo else (value - lo) / (hi - lo)
        fill = pygame.Rect(track.x, track.y, int(track.w * t), track.h)
        pygame.draw.rect(surf, color, fill, border_radius=3)
        pygame.draw.circle(surf, T.TEXT, (rect.x + int(rect.w * t), rect.centery), 7)
        return value, changed

    def tooltip(self, surf):
        if not self.tip:
            return
        s, r = self.tip
        img = T.font(13).render(s, True, T.TEXT)
        box = img.get_rect(midtop=(r.centerx, r.bottom + 6)).inflate(12, 8)
        box.clamp_ip(surf.get_rect())
        pygame.draw.rect(surf, T.PANEL_LIGHT, box, border_radius=4)
        pygame.draw.rect(surf, T.BORDER, box, 1, border_radius=4)
        surf.blit(img, img.get_rect(center=box.center))


class ListPopup:
    ROW = 30

    def __init__(self, title, items, empty="Nothing here yet"):
        self.title = title
        self.items = items
        self.empty = empty
        self.scroll = 0

    def draw(self, surf, ctl):
        sw, sh = surf.get_size()
        dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        surf.blit(dim, (0, 0))
        rows = min(max(len(self.items), 1), 12)
        box = pygame.Rect(0, 0, 560, 80 + rows * self.ROW)
        box.center = (sw // 2, sh // 2)
        pygame.draw.rect(surf, T.PANEL, box, border_radius=8)
        pygame.draw.rect(surf, T.BORDER, box, 1, border_radius=8)
        T.text(surf, self.title, (box.x + 16, box.y + 14), 18, T.TEXT, bold=True)
        T.text(surf, "Esc to close", (box.right - 16, box.y + 18), 13, T.TEXT_DIM, anchor="topright")
        area = pygame.Rect(box.x + 12, box.y + 48, box.w - 24, rows * self.ROW)
        result = None
        if ctl.wheel and area.collidepoint(ctl.mouse):
            self.scroll -= ctl.wheel
        self.scroll = max(0, min(self.scroll, max(0, len(self.items) - rows)))
        if not self.items:
            T.text(surf, self.empty, area.center, 15, T.TEXT_DIM, anchor="center")
        for k in range(rows):
            i = self.scroll + k
            if i >= len(self.items):
                break
            label, sub, payload = self.items[i]
            r = pygame.Rect(area.x, area.y + k * self.ROW, area.w, self.ROW - 2)
            hov = r.collidepoint(ctl.mouse)
            pygame.draw.rect(surf, T.PANEL_LIGHT if hov else T.BG, r, border_radius=4)
            T.text(surf, label, (r.x + 10, r.centery), 15, T.TEXT, anchor="midleft")
            T.text(surf, sub, (r.right - 10, r.centery), 13, T.TEXT_DIM, anchor="midright")
            if ctl.hit(r):
                result = payload
        if ctl.click is not None and not box.collidepoint(ctl.click):
            ctl.click = None
            return "close"
        ctl.click = None
        return result
