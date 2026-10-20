"""Векторизованные генераторы геометрии (numpy) для сцены.

Каждая функция возвращает (N, 6) float32-массив: [x, y, z, nx, ny, nz],
треугольники в порядке CCW снаружи. Генерация O(1) python-итераций —
весь цикл живёт внутри numpy.
"""

from __future__ import annotations

import numpy as np


def build_floor(size: float, divisions: int) -> np.ndarray:
    """Пол size×size из divisions² клеток, каждая разбита на 2 треугольника."""
    half = size * 0.5
    step = size / divisions

    # Координаты сетки: i — по Z, j — по X
    idx = np.arange(divisions, dtype=np.float32)
    x0 = (-half + idx * step).astype(np.float32)
    x1 = (x0 + step).astype(np.float32)
    z0 = x0.copy()
    z1 = x1.copy()

    # Декартово произведение клеток (divisions²) — всё одним махом numpy
    gx0, gz0 = np.meshgrid(x0, z0)   # (rows=z, cols=x)
    gx1, gz1 = np.meshgrid(x1, z1)

    a = np.stack([gx0.ravel(), np.zeros_like(gx0.ravel()), gz0.ravel()], -1)
    b = np.stack([gx1.ravel(), np.zeros_like(gx1.ravel()), gz0.ravel()], -1)
    c = np.stack([gx1.ravel(), np.zeros_like(gx1.ravel()), gz1.ravel()], -1)
    d = np.stack([gx0.ravel(), np.zeros_like(gx0.ravel()), gz1.ravel()], -1)

    # Два треугольника на квад: (a,b,c) и (a,c,d) — CCW сверху (+Y)
    tris = np.concatenate(
        (np.stack([a, b, c], 1), np.stack([a, c, d], 1)), axis=0
    ).reshape(-1, 3)  # (3*verts, 3)

    normals = np.tile(np.array([0.0, 1.0, 0.0], np.float32), (len(tris), 1))
    return np.hstack((tris, normals)).astype(np.float32, copy=False)


# Грани куба (нормаль, p00, u=p10-p00, v=p01-p00), CCW при взгляде снаружи.
# Координаты — в единичном кубе [-1, 1]³; масштабируются на size (без *0.5).
_FACE_DEFS = (
    (( 1.0, 0.0, 0.0), ( 1.0, -1.0, -1.0), (0.0,  2.0,  0.0), (0.0,  0.0,  2.0)),
    ((-1.0, 0.0, 0.0), (-1.0, -1.0,  1.0), (0.0,  2.0,  0.0), (0.0,  0.0, -2.0)),
    (( 0.0, 1.0, 0.0), (-1.0,  1.0, -1.0), (2.0,  0.0,  0.0), (0.0,  0.0, -2.0)),
    (( 0.0,-1.0, 0.0), (-1.0, -1.0,  1.0), (2.0,  0.0,  0.0), (0.0,  0.0,  2.0)),
    (( 0.0, 0.0, 1.0), (-1.0, -1.0,  1.0), (2.0,  0.0,  0.0), (0.0,  2.0,  0.0)),
    (( 0.0, 0.0,-1.0), ( 1.0, -1.0, -1.0), (-2.0, 0.0,  0.0), (0.0,  2.0,  0.0)),
)


def build_box(center, size, subdivisions: int) -> np.ndarray:
    """Куб center=(x,y,z), size=(w,h,d); каждая грань — сетка n×n (2n² тр.)."""
    cx, cy, cz = center
    # [-1,1]³ × (size/2) + center → грани на расстоянии size/2 от центра
    scale = np.asarray(size, dtype=np.float32) * 0.5
    offset = np.array([cx, cy, cz], dtype=np.float32)
    n = max(int(subdivisions), 1)

    parts = []
    t = (np.arange(n + 1, dtype=np.float32) / n).astype(np.float32)
    uu, vv = np.meshgrid(t[:-1], t[:-1])          # (n, n) нижние углы клеток
    uu1, vv1 = np.meshgrid(t[1:], t[1:])          # верхние

    for normal, p00, uvec, vvec in _FACE_DEFS:
        # corner(u,v) = (p00 + u*U + v*V) * scale + offset — без двойного масштаба
        p00 = np.array(p00, np.float32)
        U = np.array(uvec, np.float32)
        V = np.array(vvec, np.float32)

        def corner(u, v):
            pts = p00 + u[..., None] * U + v[..., None] * V
            return pts.reshape(-1, 3) * scale + offset

        a = corner(uu.ravel(), vv.ravel())
        b = corner(uu1.ravel(), vv.ravel())
        c = corner(uu1.ravel(), vv1.ravel())
        d = corner(uu.ravel(), vv1.ravel())

        tris = np.concatenate(
            (np.stack([a, b, c], 1), np.stack([a, c, d], 1)), axis=0
        ).reshape(-1, 3)
        norms = np.tile(np.array(normal, np.float32), (len(tris), 1))
        parts.append(np.hstack((tris, norms)))

    return np.concatenate(parts).astype(np.float32, copy=False)
