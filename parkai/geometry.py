import numpy as np

RECT_UNIT = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rect_corners(x, y, angle, length, width):
    x, y, angle = np.asarray(x, float), np.asarray(y, float), np.asarray(angle, float)
    length, width = np.asarray(length, float), np.asarray(width, float)
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    lx = RECT_UNIT[:, 0] * length[..., None]
    ly = RECT_UNIT[:, 1] * width[..., None]
    cx = x[..., None] + lx * c - ly * s
    cy = y[..., None] + lx * s + ly * c
    return np.stack([cx, cy], -1)


def poly_edges(corners):
    nxt = np.roll(corners, -1, axis=-2)
    return np.concatenate([corners, nxt], -1)


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def segments_intersect(a, b):
    px, py = a[..., 0], a[..., 1]
    rx, ry = a[..., 2] - px, a[..., 3] - py
    qx, qy = b[..., 0], b[..., 1]
    sx, sy = b[..., 2] - qx, b[..., 3] - qy
    rxs = _cross(rx, ry, sx, sy)
    qpx, qpy = qx - px, qy - py
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qpx, qpy, sx, sy) / rxs
        u = _cross(qpx, qpy, rx, ry) / rxs
    return (np.abs(rxs) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)


def points_in_rects(points, rx, ry, rangle, rlength, rwidth):
    dx = points[..., 0] - rx
    dy = points[..., 1] - ry
    c, s = np.cos(rangle), np.sin(rangle)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    return (np.abs(lx) <= rlength * 0.5) & (np.abs(ly) <= rwidth * 0.5)


def to_local(points, x, y, angle):
    dx = points[..., 0] - x
    dy = points[..., 1] - y
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([dx * c + dy * s, -dx * s + dy * c], -1)


def to_world(local, x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    lx, ly = local[..., 0], local[..., 1]
    return np.stack([x + lx * c - ly * s, y + lx * s + ly * c], -1)


def cast_rays(origins, dirs, segs, max_range):
    px = origins[..., None, None, 0]
    py = origins[..., None, None, 1]
    rx = dirs[..., :, None, 0]
    ry = dirs[..., :, None, 1]
    if segs.ndim == 2:
        segs = segs[None, None]
    else:
        segs = segs[:, None]
    qx, qy = segs[..., 0], segs[..., 1]
    sx, sy = segs[..., 2] - qx, segs[..., 3] - qy
    rxs = _cross(rx, ry, sx, sy)
    qpx, qpy = qx - px, qy - py
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qpx, qpy, sx, sy) / rxs
        u = _cross(qpx, qpy, rx, ry) / rxs
    valid = (np.abs(rxs) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1) & (t <= max_range)
    t = np.where(valid, t, np.inf)
    if t.shape[-1] == 0:
        shape = t.shape[:-1]
        return np.full(shape, np.inf), np.full(shape, -1, dtype=np.int64)
    idx = np.argmin(t, -1)
    dist = np.take_along_axis(t, idx[..., None], -1)[..., 0]
    return dist, idx


def path_pose(waypoints, cumlen, s, closed):
    total = cumlen[-1]
    if total <= 1e-9:
        n = np.shape(s)
        return np.full(n, waypoints[0, 0]), np.full(n, waypoints[0, 1]), np.zeros(n)
    if closed:
        s = np.mod(s, total)
        forward = np.ones_like(s, dtype=bool)
    else:
        period = 2 * total
        s = np.mod(s, period)
        forward = s <= total
        s = np.where(forward, s, period - s)
    i = np.clip(np.searchsorted(cumlen, s, side="right") - 1, 0, len(cumlen) - 2)
    seg_len = np.maximum(cumlen[i + 1] - cumlen[i], 1e-9)
    f = (s - cumlen[i]) / seg_len
    p0, p1 = waypoints[i], waypoints[i + 1]
    pos = p0 + (p1 - p0) * f[..., None]
    d = p1 - p0
    heading = np.arctan2(d[..., 1], d[..., 0])
    heading = np.where(forward, heading, heading + np.pi)
    return pos[..., 0], pos[..., 1], wrap_angle(heading)
