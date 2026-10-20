import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from engine.camera import Camera
from engine.lighting import LightingSystem


class Scene:
    def __init__(self, floor_size=100.0, floor_divisions=800, box_subdivisions=30):
        self.floor_size = floor_size
        self.floor_divisions = floor_divisions
        self.box_subdivisions = box_subdivisions
        self._floor_list = None
        self._boxes_list = None

    def draw(self):
        self._draw_floor()
        self._draw_boxes()

    # ---------- пол ----------
    def _draw_floor(self):
        if self._floor_list is None:
            self._build_floor_list()
        glColor3f(0.30, 0.28, 0.25)
        glCallList(self._floor_list)

    def _build_floor_list(self):
        half = self.floor_size * 0.5
        step = self.floor_size / self.floor_divisions

        self._floor_list = glGenLists(1)
        glNewList(self._floor_list, GL_COMPILE)
        glNormal3f(0.0, 1.0, 0.0)
        glBegin(GL_QUADS)
        for i in range(self.floor_divisions):
            z0 = -half + i * step
            z1 = z0 + step
            for j in range(self.floor_divisions):
                x0 = -half + j * step
                x1 = x0 + step
                glVertex3f(x0, 0.0, z0)
                glVertex3f(x1, 0.0, z0)
                glVertex3f(x1, 0.0, z1)
                glVertex3f(x0, 0.0, z1)
        glEnd()
        glEndList()

    # ---------- кубы ----------
    def _draw_boxes(self):
        if self._boxes_list is None:
            self._build_boxes_list()
        glColor3f(0.32, 0.30, 0.27)
        glCallList(self._boxes_list)

    def _build_boxes_list(self):
        self._boxes_list = glGenLists(1)
        glNewList(self._boxes_list, GL_COMPILE)
        n = self.box_subdivisions
        self._draw_box(-6.0, 1.0, -6.0, 2.0, 2.0, 2.0, n)
        self._draw_box( 4.0, 1.5, -9.0, 3.0, 3.0, 3.0, n)
        self._draw_box( 9.0, 0.75, 3.0, 1.5, 1.5, 1.5, n)
        self._draw_box(-11.0, 2.0, 4.0, 4.0, 4.0, 4.0, n)
        self._draw_box( 0.0, 0.5, -14.0, 1.0, 1.0, 1.0, n)
        glEndList()

    def _draw_box(self, x, y, z, w, h, d, n):
        """Куб, каждая грань — сетка n×n подквадов.

        n=1 → 6 квадов на куб (как было)
        n=3 → 54 квада на куб (~9x, ближайшая чистая сетка к 10x)
        n=4 → 96 квадов на куб (16x)
        """
        hw, hh, hd = w * 0.5, h * 0.5, d * 0.5

        glPushMatrix()
        glTranslatef(x, y, z)

        # Каждая грань: (нормаль, 4 угла в CCW-порядке при взгляде снаружи)
        # углы: p00 → p10 → p11 → p01 (это квадрат в порядке обхода)
        faces = [
            # +X
            (( 1, 0, 0),
             ( hw, -hh, -hd), ( hw,  hh, -hd), ( hw,  hh,  hd), ( hw, -hh,  hd)),
            # -X
            ((-1, 0, 0),
             (-hw, -hh,  hd), (-hw,  hh,  hd), (-hw,  hh, -hd), (-hw, -hh, -hd)),
            # +Y
            (( 0, 1, 0),
             (-hw,  hh, -hd), ( hw,  hh, -hd), ( hw,  hh,  hd), (-hw,  hh,  hd)),
            # -Y
            (( 0,-1, 0),
             (-hw, -hh,  hd), ( hw, -hh,  hd), ( hw, -hh, -hd), (-hw, -hh, -hd)),
            # +Z
            (( 0, 0, 1),
             (-hw, -hh,  hd), ( hw, -hh,  hd), ( hw,  hh,  hd), (-hw,  hh,  hd)),
            # -Z
            (( 0, 0,-1),
             ( hw, -hh, -hd), (-hw, -hh, -hd), (-hw,  hh, -hd), ( hw,  hh, -hd)),
        ]

        for normal, p00, p10, p11, p01 in faces:
            glNormal3f(*normal)
            glBegin(GL_QUADS)
            for i in range(n):
                u0 = i / n
                u1 = (i + 1) / n
                for j in range(n):
                    v0 = j / n
                    v1 = (j + 1) / n
                    glVertex3f(*self._bilerp(p00, p10, p11, p01, u0, v0))
                    glVertex3f(*self._bilerp(p00, p10, p11, p01, u1, v0))
                    glVertex3f(*self._bilerp(p00, p10, p11, p01, u1, v1))
                    glVertex3f(*self._bilerp(p00, p10, p11, p01, u0, v1))
            glEnd()

        glPopMatrix()

    @staticmethod
    def _bilerp(p00, p10, p11, p01, u, v):
        w00 = (1.0 - u) * (1.0 - v)
        w10 = u * (1.0 - v)
        w11 = u * v
        w01 = (1.0 - u) * v
        return (
            w00 * p00[0] + w10 * p10[0] + w11 * p11[0] + w01 * p01[0],
            w00 * p00[1] + w10 * p10[1] + w11 * p11[1] + w01 * p01[1],
            w00 * p00[2] + w10 * p10[2] + w11 * p11[2] + w01 * p01[2],
        )

    def cleanup(self):
        if self._floor_list is not None:
            glDeleteLists(self._floor_list, 1)
            self._floor_list = None
        if self._boxes_list is not None:
            glDeleteLists(self._boxes_list, 1)
            self._boxes_list = None


def main():
    pygame.init()
    display = (800, 600)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Core Descent: Drilling Horror Engine")

    camera = Camera()
    lighting = LightingSystem()
    scene = Scene()

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  [0.35, 0.35, 0.35, 1.0])
    glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 24.0)

    lighting.setup_global_lighting()

    glMatrixMode(GL_PROJECTION)
    gluPerspective(75.0, display[0] / display[1], 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()

    clock = pygame.time.Clock()
    running = True

    print("Управление: WASD — движение, Мышь — обзор, F — вкл/выкл фонарик, ESC — выход, F1 — курсор")

    while running:
        delta_time = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    is_grabbed = not pygame.event.get_grab()
                    pygame.event.set_grab(is_grabbed)
                    pygame.mouse.set_visible(not is_grabbed)
                    pygame.mouse.get_rel()
                elif event.key == pygame.K_f:
                    lighting.toggle_flashlight()

        if pygame.event.get_grab():
            mouse_dx, mouse_dy = pygame.mouse.get_rel()
        else:
            pygame.mouse.get_rel()
            mouse_dx, mouse_dy = 0, 0

        keys = pygame.key.get_pressed()
        camera.handle_input(keys, mouse_dx, mouse_dy, delta_time)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        camera.apply()

        lighting.update_flashlight(
            camera.position,
            camera.get_flashlight_direction(),
            delta_time,
        )
        scene.draw()

        pygame.display.flip()

    scene.cleanup()
    pygame.quit()


if __name__ == "__main__":
    main()