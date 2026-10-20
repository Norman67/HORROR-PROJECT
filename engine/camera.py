import pygame
import math
from OpenGL.GL import *


class Camera:
    def __init__(self):
        self.position = [0.0, 1.7, 5.0]
        self.yaw = 0.0
        self.pitch = 0.0

        # --- Скорости ---
        self.walk_speed = 6.0
        self.sprint_speed = 10.0
        self.speed = self.walk_speed          # текущая (обновляется в handle_input)
        self.mouse_sensitivity = 0.15

        # --- Head-bob state ---
        self.smooth_speed = 0.0
        self.bob_phase = 0.0
        self.idle_time = 0.0
        self.bob_offset_world = [0.0, 0.0, 0.0]
        self.bob_roll = 0.0
        self.bob_pitch = 0.0
        self.bob_yaw = 0.0

        # --- Параметры шага ---
        self.step_freq_per_mps = 0.2
        self.walk_vert_amp = 0.045
        self.walk_lat_amp = 0.015
        self.walk_fwd_amp = 0.008
        self.walk_roll_amp = 1.2

        # --- Дыхание (idle) ---
        self.breath_freq = 0.25
        self.breath_vert_amp = 0.008
        self.breath_fwd_amp = 0.003
        self.breath_yaw_amp = 0.15
        self.breath_pitch_amp = 0.10

        # --- Сглаживание ---
        self.bob_blend_speed = 3.5
        self.sprint_bob_boost = 0.45          # боб на бегу до 1.45×

    # ------------------------------------------------------------------ #
    def handle_input(self, keys, mouse_dx, mouse_dy, delta_time):
        self.yaw -= mouse_dx * self.mouse_sensitivity
        self.pitch -= mouse_dy * self.mouse_sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

        yaw_rad = math.radians(self.yaw)
        forward_x = -math.sin(yaw_rad)
        forward_z = -math.cos(yaw_rad)
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)

        move_x = 0.0
        move_z = 0.0
        if keys[pygame.K_w]:
            move_x += forward_x; move_z += forward_z
        if keys[pygame.K_s]:
            move_x -= forward_x; move_z -= forward_z
        if keys[pygame.K_a]:
            move_x -= right_x;   move_z -= right_z
        if keys[pygame.K_d]:
            move_x += right_x;   move_z += right_z

        length = math.hypot(move_x, move_z)
        if length > 0.0:
            move_x /= length
            move_z /= length

        # --- Спринт на Shift ---
        is_sprinting = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        move_speed = self.sprint_speed if is_sprinting else self.walk_speed
        self.speed = move_speed

        self.position[0] += move_x * move_speed * delta_time
        self.position[2] += move_z * move_speed * delta_time

        is_moving = length > 0.0
        self._update_bob(delta_time, is_moving, move_speed)

    # ------------------------------------------------------------------ #
    def _update_bob(self, delta_time, is_moving, move_speed):
        target_speed = move_speed if is_moving else 0.0
        k = 1.0 - math.exp(-self.bob_blend_speed * delta_time)
        self.smooth_speed += (target_speed - self.smooth_speed) * k

        # Вес относительно ХОДЬБЫ: на ходу ~1, на бегу до 1 + boost
        weight = (self.smooth_speed / self.walk_speed
                  if self.walk_speed > 0.0 else 0.0)
        weight = min(weight, 1.0 + self.sprint_bob_boost)

        # Частота растёт со сглаженной скоростью → на бегу чаще
        freq = self.smooth_speed * self.step_freq_per_mps
        self.bob_phase += delta_time * freq * 2.0 * math.pi

        self.idle_time += delta_time

        p = self.bob_phase
        walk_vert = math.sin(p * 2.0)       * self.walk_vert_amp
        walk_lat  = math.sin(p)             * self.walk_lat_amp
        walk_fwd  = math.sin(p * 2.0 + 1.0) * self.walk_fwd_amp
        walk_roll = math.sin(p)             * self.walk_roll_amp

        bt = self.idle_time * self.breath_freq * 2.0 * math.pi
        breath_vert  = math.sin(bt)              * self.breath_vert_amp
        breath_fwd   = math.sin(bt * 0.63 + 1.7) * self.breath_fwd_amp
        breath_yaw   = math.sin(bt * 0.41)       * self.breath_yaw_amp
        breath_pitch = math.sin(bt * 0.29 + 0.5) * self.breath_pitch_amp

        vert = breath_vert + walk_vert * weight
        lat  = walk_lat * weight
        fwd  = breath_fwd + walk_fwd * weight

        self.bob_roll  = walk_roll * weight
        self.bob_yaw   = breath_yaw
        self.bob_pitch = breath_pitch

        yaw_rad = math.radians(self.yaw)
        sy, cy = math.sin(yaw_rad), math.cos(yaw_rad)
        self.bob_offset_world[0] = cy * lat + (-sy) * fwd
        self.bob_offset_world[1] = vert
        self.bob_offset_world[2] = (-sy) * lat + (-cy) * fwd

    # ------------------------------------------------------------------ #
    @property
    def eye_position(self):
        return [
            self.position[0] + self.bob_offset_world[0],
            self.position[1] + self.bob_offset_world[1],
            self.position[2] + self.bob_offset_world[2],
        ]

    def apply(self):
        eye = self.eye_position
        glRotatef(self.bob_roll, 0.0, 0.0, 1.0)
        glRotatef(-(self.pitch + self.bob_pitch), 1.0, 0.0, 0.0)
        glRotatef(-(self.yaw + self.bob_yaw), 0.0, 1.0, 0.0)
        glTranslatef(-eye[0], -eye[1], -eye[2])

    # ------------------------------------------------------------------ #
    def get_forward_direction(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        y = math.sin(pitch_rad)
        z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        return [x, y, z]

    def get_flashlight_direction(self):
        yaw_rad = math.radians(self.yaw + self.bob_yaw)
        pitch_rad = math.radians(self.pitch + self.bob_pitch - 8.0)
        x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        y = math.sin(pitch_rad)
        z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        return [x, y, z]