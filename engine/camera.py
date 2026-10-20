"""FPS-камера: движение WASD, мышь-обзор, head-bob и «дыхание» в покое.

Оптимизации относительно прошлой версии:
* векторизация тригонометрии head-bob через numpy (одна пакетная sin()
  вместо ~10 скальных вызовов на кадр);
* переиспользование scratch-буферов — меньше аллокаций в горячем пути;
* константы вынесены в dataclass-конфиг (легко тюнить, нет магических чисел
  внутри логики);
* публичный API сохранён: position / yaw / pitch / handle_input / apply /
  eye_position / get_forward_direction / get_flashlight_direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame
from OpenGL.GL import glRotatef, glTranslatef

from engine.math3d import TWO_PI, clamp, exponential_smoothing_factor


@dataclass(frozen=True)
class CameraConfig:
    # Базовая позиция/ориентация
    start_position: tuple = (0.0, 1.7, 5.0)
    eye_height: float = 1.7

    # Скорости
    walk_speed: float = 6.0
    sprint_speed: float = 10.0
    mouse_sensitivity: float = 0.15
    pitch_limit: float = 89.0

    # Частота шага
    step_freq_per_mps: float = 0.2

    # Head-bob при ходьбе
    walk_vert_amp: float = 0.045
    walk_lat_amp: float = 0.015
    walk_fwd_amp: float = 0.008
    walk_roll_amp: float = 1.2

    # Дыхание в покое
    breath_freq: float = 0.25
    breath_vert_amp: float = 0.008
    breath_fwd_amp: float = 0.003
    breath_yaw_amp: float = 0.15
    breath_pitch_amp: float = 0.10

    # Смешение
    bob_blend_speed: float = 3.5
    sprint_bob_boost: float = 0.45

    # Наклон фонарика вниз относительно взгляда (градусы)
    flashlight_pitch_offset: float = -8.0


# Фазы синусоид head-bob/дыхания — считаются одним векторным вызовом numpy:
# f = [sin(2p), sin(p), sin(2p+1), sin(bt), sin(.63bt+1.7), sin(.41bt), sin(.29bt+.5)]
_SIN_MULTIPLIERS = np.array([2.0, 1.0, 2.0, 1.0, 0.63, 0.41, 0.29])
_SIN_PHASES = np.array([0.0, 0.0, 1.0, 0.0, 1.7, 0.0, 0.5])
_BASE_SELECT = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])  # 0 -> p, 1 -> bt


class Camera:
    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        cfg = self.config

        self.position = list(cfg.start_position)
        self.yaw = 0.0
        self.pitch = 0.0

        self.speed = cfg.walk_speed

        # --- Состояние head-bob ---
        self.smooth_speed = 0.0
        self.bob_phase = 0.0
        self.idle_time = 0.0
        self.bob_offset_world = np.zeros(3, dtype=np.float64)
        self.bob_roll = 0.0
        self.bob_pitch = 0.0
        self.bob_yaw = 0.0

        # Scratch для векторизованного расчёта фаз (не аллоцируется за кадр)
        self._phase_vec = np.empty(7, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def handle_input(self, keys, mouse_dx, mouse_dy, delta_time):
        """Один проход ввода: обзор + движение + обновление head-bob."""
        cfg = self.config

        self.yaw -= mouse_dx * cfg.mouse_sensitivity
        self.pitch = clamp(
            self.pitch - mouse_dy * cfg.mouse_sensitivity,
            -cfg.pitch_limit,
            cfg.pitch_limit,
        )

        yaw_rad = math.radians(self.yaw)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        forward_x, forward_z = -sy, -cy
        right_x, right_z = cy, -sy

        move_x = 0.0
        move_z = 0.0
        if keys[pygame.K_w]:
            move_x += forward_x
            move_z += forward_z
        if keys[pygame.K_s]:
            move_x -= forward_x
            move_z -= forward_z
        if keys[pygame.K_a]:
            move_x -= right_x
            move_z -= right_z
        if keys[pygame.K_d]:
            move_x += right_x
            move_z += right_z

        length = math.hypot(move_x, move_z)
        is_moving = length > 0.0
        if is_moving:
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            move_speed = cfg.sprint_speed if shift_held else cfg.walk_speed
            self.speed = move_speed
            step = move_speed * delta_time / length
            self.position[0] += move_x * step
            self.position[2] += move_z * step
        else:
            self.speed = cfg.walk_speed
            move_speed = 0.0

        self._update_bob(delta_time, is_moving, move_speed, sy, cy)

    # ------------------------------------------------------------------ #
    def _update_bob(self, delta_time, is_moving, move_speed, sy, cy):
        cfg = self.config

        target_speed = move_speed if is_moving else 0.0
        k = exponential_smoothing_factor(cfg.bob_blend_speed, delta_time)
        self.smooth_speed += (target_speed - self.smooth_speed) * k

        weight = min(self.smooth_speed / cfg.walk_speed, 1.0 + cfg.sprint_bob_boost)

        self.bob_phase += delta_time * self.smooth_speed * cfg.step_freq_per_mps * TWO_PI
        self.idle_time += delta_time

        # Все 7 синусоид — одним пакетным вызовом numpy вместо 7 скальных math.sin
        p = self.bob_phase
        bt = self.idle_time * cfg.breath_freq * TWO_PI
        pv = self._phase_vec
        np.multiply(_BASE_SELECT, bt, out=pv)
        pv += (1.0 - _BASE_SELECT) * p  # выбор базовой фазы: p или bt
        pv *= _SIN_MULTIPLIERS
        pv += _SIN_PHASES
        np.sin(pv, out=pv)

        s2p, sp, s2p1, sbt, sbt63, sbt41, sbt29 = pv

        vert = cfg.breath_vert_amp * sbt + cfg.walk_vert_amp * s2p * weight
        lat = cfg.walk_lat_amp * sp * weight
        fwd = cfg.breath_fwd_amp * sbt63 + cfg.walk_fwd_amp * s2p1 * weight

        self.bob_roll = cfg.walk_roll_amp * sp * weight
        self.bob_yaw = cfg.breath_yaw_amp * sbt41
        self.bob_pitch = cfg.breath_pitch_amp * sbt29

        off = self.bob_offset_world
        off[0] = cy * lat - sy * fwd
        off[1] = vert
        off[2] = -sy * lat - cy * fwd

    # ------------------------------------------------------------------ #
    @property
    def eye_position(self):
        """Позиция глаз с учётом head-bob (np.ndarray из 3 float)."""
        off = self.bob_offset_world
        pos = self.position
        return np.array(
            (pos[0] + off[0], pos[1] + off[1], pos[2] + off[2]),
            dtype=np.float64,
        )

    def apply(self):
        """Ставит view-матрицу через текущий GL-модельview (без лишних аллокаций)."""
        off = self.bob_offset_world
        pos = self.position
        glRotatef(self.bob_roll, 0.0, 0.0, 1.0)
        glRotatef(-(self.pitch + self.bob_pitch), 1.0, 0.0, 0.0)
        glRotatef(-(self.yaw + self.bob_yaw), 0.0, 1.0, 0.0)
        glTranslatef(-(pos[0] + off[0]), -(pos[1] + off[1]), -(pos[2] + off[2]))

    # ------------------------------------------------------------------ #
    def get_forward_direction(self):
        """Единичный вектор взгляда (pitch/yaw), numpy-массив."""
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cp = math.cos(pitch_rad)
        return np.array(
            (-math.sin(yaw_rad) * cp, math.sin(pitch_rad), -math.cos(yaw_rad) * cp),
            dtype=np.float64,
        )

    def get_flashlight_direction(self):
        """Направление фонарика: взгляд + bob + лёгкий наклон вниз (в мире)."""
        cfg = self.config
        yaw_rad = math.radians(self.yaw + self.bob_yaw)
        pitch_rad = math.radians(
            self.pitch + self.bob_pitch + cfg.flashlight_pitch_offset
        )
        cp = math.cos(pitch_rad)
        return np.array(
            (-math.sin(yaw_rad) * cp, math.sin(pitch_rad), -math.cos(yaw_rad) * cp),
            dtype=np.float64,
        )

    def flashlight_eye_direction(self):
        """То же направление, но уже в eye-space камеры: ровно (0, sinα, −cosα),
        где α = flashlight_pitch_offset. Не зависит от position/yaw/pitch —
        прожектор всегда светит «прямо перед глазами» с фиксированным наклоном.
        """
        alpha = math.radians(self.config.flashlight_pitch_offset)
        return np.array((0.0, math.sin(alpha), -math.cos(alpha)), dtype=np.float64)
