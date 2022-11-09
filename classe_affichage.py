import cv2
import numpy as np
from random import random, randint
from time import time, sleep
from math import sin, cos, sqrt, log, comb
from colorsys import hsv_to_rgb


class Affichage:
    def __init__(self, shape=(1024, 1024, 4), tick_rate=64, max_points=16, step: float = (2**-4),
                 radius_start=8, radius_end=8, hue_offset=0, window_name='aj', show_control=False):
        self.shape = shape      # shape of the render, third value is colour depth
        self.tick_rate = tick_rate
        self.max_points = max_points
        self.step = step
        self.radius_start = radius_start
        self.radius_end = radius_end
        self.hue_offset = hue_offset
        self.window_name = window_name
        self.show_control = show_control

        self.points = []
        self.img: np.ndarray = np.zeros(shape, dtype=np.uint8)

        self.tick_delay = (1.0 / self.tick_rate)
        self.tick = time()
        self.next_tick_ready: bool = True

        self.render_one_frame()
        cv2.imshow(self.window_name, self.img)

        def callback_function(event, x, y, flags, params):
            self.callback_method(event, x, y, flags, params)

        cv2.setMouseCallback(self.window_name, callback_function)
        cv2.waitKey(0)

    def callback_method(self, event, x, y, flags, params):
        if (time() - self.tick) > self.tick_delay:      # if enough time has passed since last tick
            self.tick = time()
            self.next_tick_ready = True
            self.tick_and_render(event, x, y, flags, params)
        else:
            if self.next_tick_ready:                    # if the next tick in queue is ready
                self.next_tick_ready = False            # stop waiting for the next tick in queue
                while (time() - self.tick) < self.tick_delay:
                    pass
                self.tick_and_render(event, x, y, flags, params)

    def run_one_tick(self, event, x, y, flags, params):
        if event == 0:      # 0 = mouse movement
            self.points.append((x, y))

    def render_one_frame(self):
        self.img.fill(0)        # rempli l'image (np.ndarray) avec des 0
        #self.img = np.zeros_like(self.img)

        n = len(self.points)
        if n > 0:
            while n > self.max_points >= 3:
                # just making sure there are not too many points in the list, which is almost never the case
                n = len(self.points)
                self.points.pop(0)

            h, w, c = self.shape
            area = h * w                        # area of self.img
            diagonal = sqrt(h ** 2 + w ** 2)    # length of the self.img 's diagonals (Pythagoras' theorem)
            x_prev, y_prev = self.points[0]     # initial value of "previous value" for x and y
            hue = 0         # hue for the HSV to RGB color conversion, may change later, may be unused
            t = 0
            while t <= 1:
                v = bezier(t, self.points)      # un point sur la courbe obtenu avec (self.points)
                r = max(0, int(self.radius_start * log(1 + t) + self.radius_end * log(2 - t)))
                # r est le rayon du cercle qui sera mis sur l'image
                if (len(v) >= 2) and (r > 0):
                    x = int(v[0])
                    y = int(v[1])
                    hue = (t + 0.25) % 1
                    # hue = (x + y * w) / area
                    # hue = (hue + sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2) / 512) % 1
                    self.hue_offset += self.step / 64
                    c = float_to_int(hsv_to_rgb(hue + self.hue_offset, 0.8, 0.8)) + (128,)
                    cv2.circle(self.img, (x, y), r, c, thickness=-1)
                    # thickness=-1 pour un cercle plein, pas un contour

                    x_prev, y_prev = x, y
                t += (self.step / sqrt(n))
            if self.show_control:
                for p in self.points:
                    cv2.circle(self.img, p, 4, (255, 255, 255, 255), thickness=2)

    def tick_and_render(self, event, x, y, flags, params):
        self.run_one_tick(event, x, y, flags, params)
        self.render_one_frame()
        cv2.imshow(self.window_name, self.img)


def float_to_int(c):
    """converts from the [0, 1] float color system to the [0, 255] int color system"""
    return tuple(int(a * 255) for a in c)


def vector_scale(v, n):
    """simply multiplies the vector v by the number n"""
    return tuple(a * n for a in v)


def vector_add(v0, v1):
    """adds vectors v0 and v1"""
    return tuple((v0[i] + v1[i]) for i in range(min(len(v0), len(v1))))


def bezier(t, _points):
    n = len(_points)
    if n <= 0 or t > 1 or t < 0:
        return tuple()

    result = vector_scale(_points[0], 0)
    for i in range(n):
        p = _points[i]
        v = vector_scale(p, comb(n - 1, i) *
                         (t ** i) *
                         ((1 - t) ** ((n - 1) - i))
                         )
        result = vector_add(result, v)

    return result


if __name__ == '__main__':
    a = Affichage(window_name='omagus', max_points=16, radius_start=16, show_control=True)


# TODO (je ne le ferais jamais) trouver un meilleur nom pour le fichier et la classe,
#  en fait il en faut aussi un pour le projet

