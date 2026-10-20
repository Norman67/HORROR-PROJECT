"""Core Descent: Drilling Horror Engine — точка входа.

Ключевые улучшения относительно прошлой версии:
* геометрия (пол 800×800 клеток + subdivided-кубы) генерируется ОДИН раз
  векторизованным numpy и загружается в статические VBO; за кадр вместо
  ~2.5 млн python-вызовов glVertex3f/glNormal3f — два glDrawArrays;
* исправлен winding-порядок граней кубов (CCW снаружи), включено отсечение
  задних граней — меньше растеризуемой геометрии, корректное освещение;
* FPS-независимое сглаживание мыши — плавный обзор без «ступенек»;
* конфигурация вынесена в dataclass'ы, GL-ресурсы освобождаются корректно.

Управление: WASD — движение, Shift — бег, Мышь — обзор, F — фонарик,
            ESC — выход, F1 — освободить курсор.
Запуск из папки проекта:  python main.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame
from OpenGL.GL import (
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_BACK,
    GL_CCW,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FRONT_AND_BACK,
    GL_NORMALIZE,
    GL_SHININESS,
    GL_SPECULAR,
    glClear,
    glClearColor,
    glCullFace,
    glEnable,
    glFrontFace,
    glLoadIdentity,
    glMaterialf,
    glMaterialfv,
    glMatrixMode,
)
from OpenGL.GLU import gluPerspective

from engine.camera import Camera
from engine.lighting import LightingSystem
from engine.math3d import exponential_smoothing_factor
from engine.scene import Scene


@dataclass(frozen=True)
class AppConfig:
    width: int = 800
    height: int = 600
    title: str = "Core Descent: Drilling Horror Engine"
    fov: float = 75.0
    near: float = 0.1
    far: float = 200.0
    fps_cap: int = 60
    max_frame_time: float = 0.05      # защита от «телепорта» после сворачивания
    mouse_smooth_rate: float = 30.0   # 1/с — догон позиции мыши
    material_specular: tuple = (0.35, 0.35, 0.35, 1.0)
    material_shininess: float = 24.0
    clear_color: tuple = (0.0, 0.0, 0.0, 1.0)


class MouseLookSmoother:
    """Сглаживает дельты мыши так, чтобы итоговый угол был точным.

    get_rel() отдаёт целочисленные «ступеньки»; экспоненциально догоняя
    накопленную цель, камера движется непрерывно и не зависит от FPS.
    Сумма всех step() равна исходной дельте — чувствительность не меняется.
    """

    def __init__(self, rate: float):
        self._rate = rate
        self._target_dx = 0.0
        self._target_dy = 0.0

    def feed(self, raw_dx: int, raw_dy: int):
        self._target_dx += raw_dx
        self._target_dy += raw_dy

    def reset(self):
        self._target_dx = 0.0
        self._target_dy = 0.0

    def step(self, delta_time: float) -> tuple[float, float]:
        k = exponential_smoothing_factor(self._rate, delta_time)
        out_dx = self._target_dx * k
        out_dy = self._target_dy * k
        self._target_dx -= out_dx
        self._target_dy -= out_dy
        return out_dx, out_dy


def process_events(lighting: LightingSystem) -> bool:
    """Обрабатывает очередь событий; возвращает False при сигнале к выходу."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type != pygame.KEYDOWN:
            continue
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_F1:
            grabbed = not pygame.event.get_grab()
            pygame.event.set_grab(grabbed)
            pygame.mouse.set_visible(not grabbed)
            pygame.mouse.get_rel()  # сброс накопленной дельты
        elif event.key == pygame.K_f:
            lighting.toggle_flashlight()
    return True


def run(config: AppConfig | None = None) -> None:
    config = config or AppConfig()

    pygame.init()
    pygame.display.set_mode(
        (config.width, config.height), pygame.DOUBLEBUF | pygame.OPENGL
    )
    pygame.display.set_caption(config.title)

    camera = Camera()
    lighting = LightingSystem()
    scene = Scene()
    smoother = MouseLookSmoother(config.mouse_smooth_rate)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_CULL_FACE)
    glFrontFace(GL_CCW)
    glCullFace(GL_BACK)

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, config.material_specular)
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, config.material_shininess)

    lighting.setup_global_lighting()

    glMatrixMode(GL_PROJECTION)
    gluPerspective(
        config.fov, config.width / config.height, config.near, config.far
    )
    glMatrixMode(GL_MODELVIEW)

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()

    print(
        "Управление: WASD — движение, Shift — бег, Мышь — обзор, "
        "F — фонарик, ESC — выход, F1 — курсор"
    )

    clock = pygame.time.Clock()
    running = True

    while running:
        delta_time = min(clock.tick(config.fps_cap) / 1000.0, config.max_frame_time)

        running = process_events(lighting)
        if not running:
            break

        if pygame.event.get_grab():
            smoother.feed(*pygame.mouse.get_rel())
        else:
            pygame.mouse.get_rel()
            smoother.reset()
        mouse_dx, mouse_dy = smoother.step(delta_time)

        keys = pygame.key.get_pressed()
        camera.handle_input(keys, mouse_dx, mouse_dy, delta_time)

        glClearColor(*config.clear_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        camera.apply()
        lighting.update_flashlight(
            camera.eye_position,
            camera.get_flashlight_direction(),
            delta_time,
        )
        scene.draw()

        pygame.display.flip()

    scene.cleanup()
    pygame.quit()


if __name__ == "__main__":
    run()
