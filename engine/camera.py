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

    # Коллизии / гравитация
    player_radius: float = 0.35      # «цилиндр» игрока в XZ
    player_height: float = 1.7       # рост (для проверки сверху на кубы)
    gravity: float = 20.0            # м/с² — прыжок с куба быстро приземляет


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

        # --- Коллизии / вертикальное движение ---
        # Список AABB-препятствий (min_x, min_y, min_z, max_x, max_y, max_z),
        # устанавливается из сцены через set_colliders().
        self._colliders: list[tuple] = []
        self.ground_y = cfg.eye_height   # высота «пола» под ногами (0 = земля)
        self.vy = 0.0                    # вертикальная скорость (гравитация)

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
    def set_colliders(self, aabb_list):
        """Принимает препятствия: итерируемый ((minx,miny,minz),(maxx,maxy,maxz))."""
        self._colliders = [
            (float(mn[0]), float(mn[1]), float(mn[2]),
             float(mx[0]), float(mx[1]), float(mx[2]))
            for mn, mx in aabb_list
        ]

    def _resolve_horizontal(self, x: float, z: float) -> tuple[float, float]:
        """Выталкивает круг игрока (XZ) из AABB кубов.

        Игрок НЕ проходит сквозь кубы: при пересечении он выталкивается
       вдоль минимальной оси — скольжение вдоль стены сохраняется.
        """
        r = self.config.player_radius
        for bx0, by0, bz0, bx1, by1, bz1 in self._colliders:
            if by1 <= 0.05:          # препятствие ниже уровня ног — не стена
                continue
            feet_y = self.position[1] - self.config.player_height
            if by0 >= feet_y + 1e-4:  # игрок стоит ВЫШЕ верха куба — можно идти
                continue
            # нет пересечения по XZ с учётом радиуса — пропускаем быстро
            if x < bx0 - r or x > bx1 + r or z < bz0 - r or z > bz1 + r:
                continue
            # ближайшие границы (выталкивание по минимальному проникновению)
            dx_min = x - (bx0 - r)   # вытолкнуть в -X
            dx_max = (bx1 + r) - x   # вытолкнуть в +X
            dz_min = z - (bz0 - r)
            dz_max = (bz1 + r) - z
            m = min(dx_min, dx_max, dz_min, dz_max)
            if m == dx_min:
                x = bx0 - r
            elif m == dx_max:
                x = bx1 + r
            elif m == dz_min:
                z = bz0 - r
            else:
                z = bz1 + r
        return x, z

    def _update_vertical(self, delta_time: float):
        """Гравитация + опора на верх кубов/пол. position[1] — высота глаз."""
        cfg = self.config
        feet = self.position[1] - cfg.player_height

        # Пол под ногами: максимум по всем AABB, над которыми мы находимся
        ground_top = 0.0
        r = 0.05  # узкая зона «заступа» — чтобы не спотыкаться о грани
        for bx0, by0, bz0, bx1, by1, bz1 in self._colliders:
            x, z = self.position[0], self.position[2]
            if bx0 - r <= x <= bx1 + r and bz0 - r <= z <= bz1 + r:
                if feet >= by1 - 0.35:  # мы сверху или чуть выше крышки
                    ground_top = max(ground_top, by1)

        target_eye = ground_top + cfg.player_height
        if self.position[1] > target_eye + 1e-6:
            # падаем
            self.vy -= cfg.gravity * delta_time
            self.position[1] += self.vy * delta_time
            if self.position[1] <= target_eye:
                self.position[1] = target_eye
                self.vy = 0.0
        else:
            # стоим на опоре (или плавно поднимаемся при заходе на куб)
            self.position[1] += (target_eye - self.position[1]) * min(
                1.0, 12.0 * delta_time
            )
            self.vy = 0.0
        self.ground_y = ground_top

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
            # Новое положение + разрешение коллизий со стенами кубов:
            # сквозь препятствия игрок пройти не может, но скользит вдоль них.
            nx = self.position[0] + move_x * step
            nz = self.position[2] + move_z * step
            self.position[0], self.position[2] = self._resolve_horizontal(nx, nz)
        else:
            self.speed = cfg.walk_speed
            move_speed = 0.0

        # Гравитация / опора на верх кубов (можно запрыгнуть сверху — пологий
        # подъём; при шаге с высоты — падение).
        self._update_vertical(delta_time)

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
