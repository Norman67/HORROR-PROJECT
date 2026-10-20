import math
from OpenGL.GL import *


class LightingSystem:
    def __init__(self):
        # --- Базовые параметры фонарика ---
        self.flashlight_color = [1.0, 0.95, 0.85, 1.0]
        self.flashlight_intensity = 1.0
        self.flashlight_cutoff = 35.0
        self.flashlight_exponent = 14.0
        self.linear_attenuation = 0.03
        self.quadratic_attenuation = 0.004
        self.light_ambient = [0.06, 0.06, 0.08, 1.0]
        self.is_on = True

        # Мерцание
        self.is_flickering = False
        self.flicker_speed = 0.1
        self.flicker_time = 0.0

        # --- «Фонарик в руке» ---
        self._prev_pos = None
        self._smooth_speed = 0.0
        self.speed_smooth_k = 8.0

        # Скорости-ориентиры
        self.walk_speed_ref   = 6.0
        self.sprint_speed_ref = 10.0
        self.sprint_amp_boost = 0.5

        # Фазы
        self._walk_phase = 0.0
        self._tremor_phase = 0.0

        # Походка
        self.sway_freq = 1.6
        self.sway_freq_speed_gain = 0.15
        self.sway_pos_lat_amp  = 0.035
        self.sway_pos_vert_amp = 0.025
        self.sway_dir_amp      = math.radians(1.5)

        # Тремор рук (всегда)
        self.tremor_pos_amp = 0.006
        self.tremor_dir_amp = math.radians(0.35)

        # Беговой джиттер (только при спринте)
        self.run_pos_amp = 0.018
        self.run_dir_amp = math.radians(2.0)

        # --- Запаздывание при осмотре ---
        self._smooth_dir = None                 # сглаженное направление
        self.dir_lag_speed = 10.0               # 1/с, скорость догона
        self.dir_lag_sprint_mult = 0.65         # на бегу догон медленнее

    # ------------------------------------------------------------------ #
    def setup_global_lighting(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.light_ambient)
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION,  1.0)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION,    self.linear_attenuation)
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, self.quadratic_attenuation)
        glLightf(GL_LIGHT0, GL_SPOT_CUTOFF,           self.flashlight_cutoff)
        glLightf(GL_LIGHT0, GL_SPOT_EXPONENT,         self.flashlight_exponent)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _perp_basis(direction):
        dx, dy, dz = direction
        rx, rz = -dz, dx
        rl = math.hypot(rx, rz)
        if rl < 1e-6:
            rx, rz, rl = 1.0, 0.0, 1.0
        rx /= rl
        rz /= rl
        ux = -rz * dy
        uy = rz * dx - rx * dz
        uz = rx * dy
        return (rx, 0.0, rz), (ux, uy, uz)

    # ------------------------------------------------------------------ #
    def _update_hand_shake(self, camera_position, delta_time):
        # --- Скорость камеры ---
        if self._prev_pos is None:
            self._prev_pos = list(camera_position)
            inst_speed = 0.0
        else:
            dx = camera_position[0] - self._prev_pos[0]
            dy = camera_position[1] - self._prev_pos[1]
            dz = camera_position[2] - self._prev_pos[2]
            inst_speed = math.hypot(dx, dy, dz) / max(delta_time, 1e-6)
            self._prev_pos = list(camera_position)

        k = 1.0 - math.exp(-self.speed_smooth_k * delta_time)
        self._smooth_speed += (inst_speed - self._smooth_speed) * k

        weight = min(self._smooth_speed / self.walk_speed_ref,
                     1.0 + self.sprint_amp_boost)

        denom = max(self.sprint_speed_ref - self.walk_speed_ref, 1e-6)
        run_factor = max(0.0, min(
            (self._smooth_speed - self.walk_speed_ref) / denom,
            1.0,
        ))

        # --- Фазы ---
        freq = self.sway_freq * (1.0 + self.sway_freq_speed_gain * self._smooth_speed)
        self._walk_phase  += delta_time * 2.0 * math.pi * freq
        self._tremor_phase += delta_time

        w = self._walk_phase
        t = self._tremor_phase

        sway_lat  = (math.sin(w)              * 0.7 +
                     math.sin(w * 1.71 + 1.3) * 0.3) * weight
        sway_vert = (math.sin(w * 2.0 + 0.4)  * 0.6 +
                     math.sin(w * 1.13 + 2.1) * 0.4) * weight

        tr_lat  = 0.5 * math.sin(t * 7.3)       + 0.5 * math.sin(t * 11.7 + 0.7)
        tr_vert = 0.5 * math.sin(t * 9.1 + 2.3) + 0.5 * math.sin(t * 13.4 + 1.1)

        run_lat  = (0.6 * math.sin(t * 14.3)       +
                    0.4 * math.sin(t * 19.7 + 1.1)) * run_factor
        run_vert = (0.6 * math.sin(t * 16.1 + 0.9) +
                    0.4 * math.sin(t * 22.3 + 2.2)) * run_factor

        pos_lat  = sway_lat  * self.sway_pos_lat_amp  + tr_lat  * self.tremor_pos_amp + run_lat  * self.run_pos_amp
        pos_vert = sway_vert * self.sway_pos_vert_amp + tr_vert * self.tremor_pos_amp + run_vert * self.run_pos_amp

        dir_lat  = sway_lat  * self.sway_dir_amp + tr_lat  * self.tremor_dir_amp + run_lat  * self.run_dir_amp
        dir_vert = sway_vert * self.sway_dir_amp * 0.7 + tr_vert * self.tremor_dir_amp + run_vert * self.run_dir_amp

        return pos_lat, pos_vert, dir_lat, dir_vert

    # ------------------------------------------------------------------ #
    def update_flashlight(self, camera_position, camera_target_direction, delta_time):
        # Фонарик выключен → гасим всё, не трогая состояние
        if not self.is_on:
            glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.0, 0.0, 0.0, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.0, 0.0, 0.0, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
            return

        current_intensity = self.flashlight_intensity

        if self.is_flickering:
            self.flicker_time += delta_time * self.flicker_speed
            current_intensity *= 0.4 + 0.6 * (0.5 + 0.5 * math.sin(self.flicker_time * 10.0))

        # 1) hand-shake (он же обновляет _smooth_speed)
        pos_lat, pos_vert, dir_lat, dir_vert = self._update_hand_shake(
            camera_position, delta_time
        )

        # 2) запаздывание направления (фонарь «догоняет» взгляд)
        if self._smooth_dir is None:
            self._smooth_dir = list(camera_target_direction)

        k_dir = self.dir_lag_speed
        if self._smooth_speed > self.walk_speed_ref:
            k_dir *= self.dir_lag_sprint_mult
        k = 1.0 - math.exp(-k_dir * delta_time)
        for i in range(3):
            self._smooth_dir[i] += (camera_target_direction[i] - self._smooth_dir[i]) * k

        # нормализация (направление должно быть единичным)
        sl = math.sqrt(
            self._smooth_dir[0] ** 2 +
            self._smooth_dir[1] ** 2 +
            self._smooth_dir[2] ** 2
        ) or 1.0
        lagged_dir = [
            self._smooth_dir[0] / sl,
            self._smooth_dir[1] / sl,
            self._smooth_dir[2] / sl,
        ]

        # 3) базис — от уже-запаздывающего направления
        right, up = self._perp_basis(lagged_dir)

        light_position = [
            camera_position[0] + right[0] * pos_lat + up[0] * pos_vert,
            camera_position[1] + right[1] * pos_lat + up[1] * pos_vert,
            camera_position[2] + right[2] * pos_lat + up[2] * pos_vert,
            1.0,
        ]
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        ddx = lagged_dir[0] + right[0] * dir_lat + up[0] * dir_vert
        ddy = lagged_dir[1] + right[1] * dir_lat + up[1] * dir_vert
        ddz = lagged_dir[2] + right[2] * dir_lat + up[2] * dir_vert
        dl = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) or 1.0
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, [ddx / dl, ddy / dl, ddz / dl])

        diffuse = [
            self.flashlight_color[0] * current_intensity,
            self.flashlight_color[1] * current_intensity,
            self.flashlight_color[2] * current_intensity,
            1.0,
        ]
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, diffuse)

    # ------------------------------------------------------------------ #
    def toggle_flicker(self):
        self.is_flickering = not self.is_flickering

    def toggle_flashlight(self):
        self.is_on = not self.is_on
