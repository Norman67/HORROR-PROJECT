"""Утилиты линейной алгебры для движка.

Все тяжёлые вычисления (head-bob, тремор фонарика, нормализации) переведены
на numpy-векторизацию: вместо тысяч скальных python-операций за кадр —
несколько пакетных операций над маленькими float64-массивами.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi

# Временные буферы переиспользуются между кадрами (однопоточный рендер-цикл),
# что убирает аллокации из горячего пути.
_BUF3_A = np.empty(3, dtype=np.float64)
_BUF3_B = np.empty(3, dtype=np.float64)


def normalize(vec, out=None):
    """Единичный вектор; нулевой вектор возвращается как есть."""
    v = np.asarray(vec, dtype=np.float64)
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        return v.copy() if out is None else out
    scaled = v / length
    if out is None:
        return scaled
    out[:] = scaled
    return out


def exponential_smoothing_factor(blend_rate, delta_time):
    """Коэффициент экспоненциального сглаживания, не зависящий от FPS.

    k = 1 - exp(-rate * dt); при rate*dt -> 0 даёт ~rate*dt, но корректно
    работает и на больших шагах времени (k всегда в [0, 1)).
    """
    return 1.0 - math.exp(-blend_rate * delta_time)


def clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def temp_buffer(name):
    """Переиспользуемый scratch-буфер ('a' или 'b')."""
    return _BUF3_A if name == "a" else _BUF3_B
