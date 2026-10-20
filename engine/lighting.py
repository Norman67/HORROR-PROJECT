"""Система освещения: фонарик (GL_LIGHT0) с мерцанием, «рукой» и запаздыванием.

Оптимизации относительно прошлой версии:
* все векторные операции (базис «руки», догоняющее направление, нормализации)
  — на numpy без аллокаций в горячем пути (переиспользуемые буферы);
* GL-состояние фонарика кэшируется: одинаковые значения не пересылаются в
  драйвер каждый кадр (glLightfv — дорогой вызов, ~6 штук за кадр раньше);
* константы вынесены в dataclass-конфиг;
* публичный API сохранён: setup_global_lighting / update_flashlight /
  toggle_flashlight / toggle_flicker / is_on / is_flickering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_COLOR_MATERIAL,
    GL_CONSTANT_ATTENUATION,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LIGHT_MODEL_AMBIENT,
    GL_LIGHT_MODEL_LOCAL_VIEWER,
    GL_LINEAR_ATTENUATION,
    GL_POSITION,
    GL_QUADRATIC_ATTENUATION,
    GL_SPECULAR,
    GL_SPOT_CUTOFF,
    GL_SPOT_DIRECTION,
    GL_SPOT_EXPONENT,
    glEnable,
    glColorMaterial,
    glLightf,
    glLightfv,
    glLightModelfv,
    glLightModeli,
)

from engine.math3d import TWO_PI, exponential_smoothing_factor

_BLACK = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class LightingConfig:
    # Фонарик
    flashlight_color: tuple = (1.0, 0.95, 0.85, 1.0)
    flashlight_intensity: float = 1.0
    flashlight_cutoff: float = 35.0
    flashlight_exponent: float = 14.0
    linear_attenuation: float = 0.03
    quadratic_attenuation: float = 0.004
    light_ambient: tuple = (0.06, 0.06, 0.08, 1.0)

    # Мерцание
    flicker_speed: float = 0.1

    # Оценка скорости камеры
    speed_smooth_k: float = 8.0

    # Скорости-ориентиры
    walk_speed_ref: float = 6.0
    sprint_speed_ref: float = 10.0
    sprint_amp_boost: float = 0.5

    # Походка «руки»
    sway_freq: float = 1.6
    sway_freq_speed_gain: float = 0.15
    sway_pos_lat_amp: float = 0.035
    sway_pos_vert_amp: float = 0.025
    sway_dir_amp: float = math.radians(1.5)

    # Тремор рук (всегда)
    tremor_pos_amp: float = 0.006
    tremor_dir_amp: float = math.radians(0.35)

    # Беговой джиттер (только при спринте)
    run_pos_amp: float = 0.018
    run_dir_amp: float = math.radians(2.0)

    # Запаздывание направления («фонарь догоняет взгляд»)
    dir_lag_speed: float = 10.0
    dir_lag_sprint_mult: float = 0.65


# Частоты осцилляторов для векторизованного расчёта тряски.
# Порядок фаз: [w, w*1.71+1.3, w*2+0.4, w*1.13+2.1, t*7.3, t*11.7+0.7,
#               t*9.1+2.3, t*13.4+1.1, t*14.3, t*19.7+1.1, t*16.1+0.9, t*22.3+2.2]
_OSC_BASE = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
_OSC_FREQ = np.array([1.0, 1.71, 2.0, 1.13, 7.3, 11.7, 9.1, 13.4, 14.3, 19.7, 16.1, 22.3])
_OSC_PHASE = np.array([0.0, 1.3, 0.4, 2.1, 0.0, 0.7, 2.3, 1.1, 0.0, 1.1, 0.9, 2.2])
# Веса внутри каждой группы (sway_lat, sway_vert, tremor, run):
_MIX_LAT = np.array([0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0])
_MIX_VERT = np.array([0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.6, 0.4])
_TREMOR_LAT = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_TREMOR_VERT = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])


class LightingSystem:
    def __init__(self, config: LightingConfig | None = None):
        self.config = config or LightingConfig()
        cfg = self.config

        self.is_on = True
        self.is_flickering = False

        # --- Состояние ---
        self._prev_pos = None
        self._smooth_speed = 0.0
        self._walk_phase = 0.0
        self._tremor_time = 0.0
        self._flicker_time = 0.0
        self._smooth_dir = None

        # Scratch-буферы (без аллокаций за кадр)
        self._osc = np.empty(12, dtype=np.float64)   # sin-фазы осцилляторов
        self._lagged = np.empty(3, dtype=np.float64)
        self._dir_buf = np.empty(3, dtype=np.float64)
        self._up_buf = np.empty(3, dtype=np.float64)
        self._pos_buf = np.zeros(4, dtype=np.float32)  # w=0, eye-point
        self._diffuse_buf = np.empty(4, dtype=np.float32)
        self._diffuse_buf[3] = 1.0
        # Направление прожектора: мировое -> eye-space
        self._spot_world = np.empty(3, dtype=np.float64)
        self._spot_eye = np.empty(3, dtype=np.float64)
        self._basis_mat = np.empty(9, dtype=np.float64)

        # Кэш последнего отправленного в GL состояния фонарика:
        # {(pname): значение} — пропускаем redundant glLightfv
        self._gl_cache = {}
        # Кэш GL_LIGHT_MODEL_AMBIENT: не пересылаем одно и то же в драйвер.
        self._last_model_ambient = None
        self._setup_done = False

    # ------------------------------------------------------------------ #
    def setup_global_lighting(self):
        if self._setup_done:
            return
        cfg = self.config
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1)
        self._set_model_ambient(cfg.light_ambient)
        glLightfv(GL_LIGHT0, GL_AMBIENT, cfg.light_ambient)
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, cfg.linear_attenuation)
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, cfg.quadratic_attenuation)
        glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, cfg.flashlight_cutoff)
        glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, cfg.flashlight_exponent)
        self._gl_cache.clear()
        self._setup_done = True

    # ------------------------------------------------------------------ #
    def _set_model_ambient(self, value):
        """GL_LIGHT_MODEL_AMBIENT с дедупликацией пересылок."""
        if self._last_model_ambient == tuple(value):
            return
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, value)
        self._last_model_ambient = tuple(value)

    def _set_light_vector(self, pname, value):
        """glLightfv с дедупликацией: не шлём в драйвер неизменное значение.

        ВНИМАНИЕ: glLightfv(GL_POSITION/GL_SPOT_DIRECTION) трансформируется
        ТЕКУЩЕЙ modelview. Все значения фонарика задаются в eye-space, а
        cache сбрасывается каждый кадр (см. update_flashlight), поэтому
        пересылка всегда происходит при актуальной матрице камеры.
        """
        cached = self._gl_cache.get(pname)
        if cached is not None and np.array_equal(cached, value):
            return
        glLightfv(GL_LIGHT0, pname, value)
        self._gl_cache[pname] = value.copy()

    @staticmethod
    def _world_basis_to_eye(yaw_rad, pitch_rad, out):
        """Строки view-базиса камеры (3×3) в плоский массив `out` (in-place).

        Соответствует glRotatef(-pitch,X)·glRotatef(-yaw,Y) из camera.apply():
        out = [ex_x, ey_x, ez_x,  ex_y, ey_y, ez_y,  ex_z, ey_z, ez_z],
        где eye = Rᵀ · (world - eye_pos), R — повороты камеры.
        """
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        # forward (eye -Z axis expressed in world) = (-cp*sy? ) — вывод:
        # view row0 (screen X) = (-sy, 0, cy); row1 (screen Y) = (-sp*cy? ...)
        out[0] = -sy;   out[1] = 0.0;  out[2] = cy
        out[3] = -sp * cy; out[4] = cp; out[5] = sp * sy
        out[6] = -cp * cy; out[7] = -sp; out[8] = cp * sy

    @staticmethod
    def _perp_basis(direction, right_out, up_out):
        """Ортонормированный базис вокруг направления (in-place, без аллокаций)."""
        dx, dy, dz = direction
        rx, rz = -dz, dx
        rl = math.hypot(rx, rz)
        if rl < 1e-6:
            rx, rz, rl = 1.0, 0.0, 1.0
        inv = 1.0 / rl
        rx *= inv
        rz *= inv
        right_out[0] = rx
        right_out[1] = 0.0
        right_out[2] = rz
        up_out[0] = -rz * dy
        up_out[1] = rz * dx - rx * dz
        up_out[2] = rx * dy

    # ------------------------------------------------------------------ #
    def _update_hand_shake(self, camera_position, delta_time):
        cfg = self.config

        # --- Скорость камеры (для амплитуд тряски) ---
        pos = np.asarray(camera_position, dtype=np.float64)
        if self._prev_pos is None:
            self._prev_pos = pos.copy()
            inst_speed = 0.0
        else:
            inst_speed = float(np.linalg.norm(pos - self._prev_pos)) / max(
                delta_time, 1e-6
            )
            self._prev_pos = pos

        k = exponential_smoothing_factor(cfg.speed_smooth_k, delta_time)
        self._smooth_speed += (inst_speed - self._smooth_speed) * k

        weight = min(self._smooth_speed / cfg.walk_speed_ref,
                     1.0 + cfg.sprint_amp_boost)
        denom = max(cfg.sprint_speed_ref - cfg.walk_speed_ref, 1e-6)
        run_factor = min(max((self._smooth_speed - cfg.walk_speed_ref) / denom, 0.0), 1.0)

        # --- Все 12 осцилляторов одним пакетным sin() ---
        freq = cfg.sway_freq * (1.0 + cfg.sway_freq_speed_gain * self._smooth_speed)
        self._walk_phase += delta_time * TWO_PI * freq
        self._tremor_time += delta_time

        osc = self._osc
        base = np.where(_OSC_BASE > 0.5, self._tremor_time, self._walk_phase)
        np.multiply(base, _OSC_FREQ, out=osc)
        osc += _OSC_PHASE
        np.sin(osc, out=osc)

        sway_lat = float(osc @ _MIX_LAT) * weight
        sway_vert = float(osc @ _MIX_VERT) * weight
        tr_lat = float(osc @ _TREMOR_LAT)
        tr_vert = float(osc @ _TREMOR_VERT)
        # run-компоненты: последние 4 осциллятора, веса те же что MIX_* но *run_factor
        run_lat = (osc[8] * 0.6 + osc[9] * 0.4) * run_factor
        run_vert = (osc[10] * 0.6 + osc[11] * 0.4) * run_factor

        pos_lat = (sway_lat * cfg.sway_pos_lat_amp
                   + tr_lat * cfg.tremor_pos_amp
                   + run_lat * cfg.run_pos_amp)
        pos_vert = (sway_vert * cfg.sway_pos_vert_amp
                    + tr_vert * cfg.tremor_pos_amp
                    + run_vert * cfg.run_pos_amp)
        dir_lat = (sway_lat * cfg.sway_dir_amp
                   + tr_lat * cfg.tremor_dir_amp
                   + run_lat * cfg.run_dir_amp)
        dir_vert = (sway_vert * cfg.sway_dir_amp * 0.7
                    + tr_vert * cfg.tremor_dir_amp
                    + run_vert * cfg.run_dir_amp)

        return pos_lat, pos_vert, dir_lat, dir_vert

    # ------------------------------------------------------------------ #
    def update_flashlight(
        self, camera_position, camera_target_direction, delta_time,
        camera_yaw=None, camera_pitch=None, bob_roll=0.0,
    ):
        """Обновляет GL_LIGHT0 (прожектор «в руке» игрока).

        camera_yaw/camera_pitch/bob_roll — актуальная ориентация камеры
        (углы в градусах). Они нужны, чтобы задать направление прожектора
        в eye-space: glLightfv(GL_SPOT_DIRECTION) трансформируется текущей
        modelview, поэтому мировое «догоняющее» направление пересчитывается
        в глаза камеры напрямую (без зависимости от того, вызвана ли уже
        camera.apply()).
        """
        cfg = self.config

        # Кэш глади невалиден между кадрами: пересылать векторы нужно при
        # актуальной матрице модели-вида каждого кадра.
        self._gl_cache.clear()

        # Фонарик выключен → гасим свет, состояние тряски не трогаем
        if not self.is_on:
            self._set_light_vector(GL_AMBIENT, _BLACK)
            self._set_light_vector(GL_DIFFUSE, _BLACK)
            self._set_light_vector(GL_SPECULAR, _BLACK)
            return

        intensity = cfg.flashlight_intensity
        if self.is_flickering:
            self._flicker_time += delta_time * cfg.flicker_speed
            intensity *= 0.4 + 0.5 * (1.0 + math.sin(self._flicker_time * 10.0))

        # 1) Тряска «руки» (заодно обновляет _smooth_speed)
        pos_lat, pos_vert, dir_lat, dir_vert = self._update_hand_shake(
            camera_position, delta_time
        )

        # 2) Запаздывание направления: фонарь «догоняет» взгляд (в мире)
        target = np.asarray(camera_target_direction, dtype=np.float64)
        if self._smooth_dir is None:
            self._smooth_dir = target.copy()
        k_dir = cfg.dir_lag_speed
        if self._smooth_speed > cfg.walk_speed_ref:
            k_dir *= cfg.dir_lag_sprint_mult
        k = exponential_smoothing_factor(k_dir, delta_time)
        lagged = self._lagged
        np.add(self._smooth_dir, (target - self._smooth_dir) * k, out=lagged)
        norm = float(np.linalg.norm(lagged))
        if norm > 1e-12:
            lagged /= norm
        self._smooth_dir = lagged.copy()

        # 3) Мировой базис «руки» вокруг запаздывающего направления
        right = self._dir_buf
        up = self._up_buf
        self._perp_basis(lagged, right, up)

        # Смещённое мировое направление луча с джиттером
        dd = self._spot_world
        np.add(lagged + right * dir_lat, up * dir_vert, out=dd)
        dn = float(np.linalg.norm(dd))
        if dn > 1e-12:
            dd /= dn

        # 4) Перевод мирового направления в eye-space камеры.
        #    Если углы не переданы — луч точно по направлению взгляда.
        if camera_yaw is None or camera_pitch is None:
            se = self._spot_eye
            np.copyto(se, dd)  # вызывающий уже передал направление в eye-space
            spot_eye = se
        else:
            yaw_rad = math.radians(camera_yaw)
            pitch_rad = math.radians(camera_pitch)
            m = self._basis_mat
            self._world_basis_to_eye(yaw_rad, pitch_rad, m)
            ex = m[0] * dd[0] + m[1] * dd[1] + m[2] * dd[2]
            ey = m[3] * dd[0] + m[4] * dd[1] + m[5] * dd[2]
            ez = m[6] * dd[0] + m[7] * dd[1] + m[8] * dd[2]
            ln = math.sqrt(ex * ex + ey * ey + ez * ez)
            if ln < 1e-12:
                spot_eye = np.array((0.0, 0.0, -1.0), dtype=np.float64)
            else:
                se = self._spot_eye
                se[0], se[1], se[2] = ex / ln, ey / ln, ez / ln
                spot_eye = se

        # Позиция света — eye-point (w=0): источник всегда «у глаз»,
        # направление задаётся уже в eye-space и не зависит от modelview.
        gp = self._pos_buf  # [0, 0, 0, 0]
        self._set_light_vector(GL_POSITION, gp)
        self._set_light_vector(GL_SPOT_DIRECTION, spot_eye)

        # Диффузный/спекулярный цвет (одинаковые) — один буфер на оба
        color = cfg.flashlight_color
        df = self._diffuse_buf
        df[0] = color[0] * intensity
        df[1] = color[1] * intensity
        df[2] = color[2] * intensity
        self._set_light_vector(GL_DIFFUSE, df)
        self._set_light_vector(GL_SPECULAR, df)

    # ------------------------------------------------------------------ #
    def toggle_flicker(self):
        self.is_flickering = not self.is_flickering

    def toggle_flashlight(self):
        self.is_on = not self.is_on
        self._gl_cache.clear()  # следующее включение должно переслать цвета

    def ensure_visible_without_flashlight(self):
        """Мягкий дежурный ambient, чтобы при выключенном фонарике экран не был чёрным.

        Без этого при выключении света сцена остаётся ровно с нулевым
        освещением и становится полностью чёрной («пол пропал»). При
        включённом фонарике ambient возвращается к обычному значению —
        так сохраняется хоррор-атмосфера со световым пятном.
        """
        if self.is_on:
            self._set_model_ambient(self.config.light_ambient)
        else:
            self._set_model_ambient((0.35, 0.33, 0.30, 1.0))
