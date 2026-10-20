"""Сцена: статическая геометрия (пол + кубы) в VBO, отрисовка за 2 вызова."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OpenGL.GL import (
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_NORMAL_ARRAY,
    GL_NORMALIZE,
    GL_VERTEX_ARRAY,
    glColor3f,
    glDisable,
    glEnableClientState,
    glFrontFace,
    glMatrixMode,
    glPushMatrix,
    glPopMatrix,
    glLoadIdentity,
    glDisableClientState,
)

from engine.geometry import build_box, build_floor
from engine.mesh import Mesh


@dataclass(frozen=True)
class BoxSpec:
    center: tuple
    size: tuple


@dataclass(frozen=True)
class SceneConfig:
    floor_size: float = 100.0
    floor_divisions: int = 800
    box_subdivisions: int = 30
    floor_color: tuple = (0.30, 0.28, 0.25)
    boxes_color: tuple = (0.32, 0.30, 0.27)
    boxes: tuple = (
        BoxSpec((-6.0, 1.0, -6.0), (2.0, 2.0, 2.0)),
        BoxSpec((4.0, 1.5, -9.0), (3.0, 3.0, 3.0)),
        BoxSpec((9.0, 0.75, 3.0), (1.5, 1.5, 1.5)),
        BoxSpec((-11.0, 2.0, 4.0), (4.0, 4.0, 4.0)),
        BoxSpec((0.0, 0.5, -14.0), (1.0, 1.0, 1.0)),
    )


class Scene:
    def __init__(self, config: SceneConfig | None = None):
        self.config = config or SceneConfig()
        cfg = self.config

        # Геометрия строится один векторизованным проходом numpy и уезжает в GPU.
        verts = [build_floor(cfg.floor_size, cfg.floor_divisions)]
        for box in cfg.boxes:
            verts.append(build_box(box.center, box.size, cfg.box_subdivisions))
        self._floor_verts = verts[0]
        self._boxes_verts = np.concatenate(verts[1:])

        self._floor_mesh: Mesh | None = None
        self._boxes_mesh: Mesh | None = None

    # ------------------------------------------------------------------ #
    def build(self):
        """Загружает меши в VBO. Требует активного GL-контекста."""
        if self._floor_mesh is None:
            self._floor_mesh = Mesh(self._floor_verts)
            self._boxes_mesh = Mesh(self._boxes_verts)

    def draw(self):
        """Два glDrawArrays вместо миллионов glVertex за кадр."""
        if self._floor_mesh is None:
            self.build()

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        try:
            glColor3f(*self.config.floor_color)
            self._floor_mesh.draw()
            glColor3f(*self.config.boxes_color)
            self._boxes_mesh.draw()
        finally:
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

    # ------------------------------------------------------------------ #
    def cleanup(self):
        if self._floor_mesh is not None:
            self._floor_mesh.cleanup()
            self._boxes_mesh.cleanup()
            self._floor_mesh = self._boxes_mesh = None
