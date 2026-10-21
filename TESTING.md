# Как тестировать проект

## 1. Запуск на своём компьютере (основной способ)
Из папки проекта:
```bash
pip install pygame PyOpenGL numpy
python main.py
```
Управление: WASD — движение, Shift — бег, Мышь — обзор, F — фонарик,
F1 — освободить курсор, ESC — выход. В консоли печатается FPS-счётчик.

Если `import OpenGL.GLU` падает с `NullFunctionError: gluPerspective` —
не установлена нативная библиотека GLU:
- Debian/Ubuntu: `sudo apt install libglu1-mesa`
- Fedora: `sudo dnf install mesa-libGLU`
- macOS обычно из коробки; Windows — PyOpenGL pip-пакет включает GLU.

## 2. Быстрые проверки без запуска окна
```bash
python -m py_compile main.py engine/*.py          # синтаксис
python -c "import engine.scene, engine.mesh"      # импорты ядра
```

## 3. Автотест в headless-среде (CI / сервер без GPU)
```bash
Xvfb :99 -screen 0 1024x768x24 &
DISPLAY=:99 SDL_VIDEODRIVER=x11 python main.py    # windowless-прогон
```
На software-рендерере (llvmpipe) движок стабильно выдаёт ~900 FPS при
ограничении cap'ом — узкое место больше не Python.

## 4. На что смотреть при тесте
* Старая версия рисовала пол 800×800 циклами `glVertex3f` (~2.5 млн вызовов
  на кадр). Теперь геометрия один раз грузится в VBO и рисуется двумя
  `glDrawArrays` — проверьте, что FPS не проседает при движении мыши.
* Кубы должны быть «цельными» (исправлен winding + отсечение задних граней).
* Обзор должен быть плавным независимо от FPS (сглаживание привязано к dt).
