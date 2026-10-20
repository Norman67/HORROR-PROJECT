"""Меш: вершины+нормали в interleaved-VBO, отрисовка одним glDrawArrays.

Ключевая оптимизация движка: геометрия генерируется векторизованно numpy
(один проход при создании сцены) и живёт в GPU-буфере. Раньше пол 800×800
клеток означал ~2.5 млн вызовов glVertex3f из python-циклов за КАЖДЫЙ кадр;
теперь — один glDrawArrays на меш.
"""

from __future__ import annotations

import ctypes

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    glBindBuffer,
    glBufferData,
    glDeleteBuffers,
    glGenBuffers,
    glNormalPointer,
    glDrawArrays,
    glVertexPointer,
)

_F32 = np.dtype(np.float32).itemsize
_ZERO = ctypes.c_void_p(0)


class Mesh:
    """Interleaved-формат вершин: [x, y, z, nx, ny, nz] (6 float на вершину)."""

    STRIDE = 6 * _F32

    def __init__(self, vertices: np.ndarray):
        """vertices: (N, 6) float32-массив (xyz + нормаль на вершину)."""
        assert vertices.dtype == np.float32 and vertices.ndim == 2
        self._vbo = int(glGenBuffers(1))
        self._count = len(vertices)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self):
        """Рисует меш (клиентские массивы вершин/нормалей должны быть включены)."""
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glVertexPointer(3, GL_FLOAT, self.STRIDE, _ZERO)
        glNormalPointer(GL_FLOAT, self.STRIDE, ctypes.c_void_p(3 * _F32))
        glDrawArrays(GL_TRIANGLES, 0, self._count)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def cleanup(self):
        glDeleteBuffers(1, [self._vbo])
        self._vbo = 0
