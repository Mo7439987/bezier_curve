import cv2
import numpy as np
from random import random, randint
from time import time
from math import sin, cos, sqrt, log, factorial
from colorsys import hsv_to_rgb


def float_to_int(c):
    return tuple(int(a * 255) for a in c)


def vector_scale(v, n):
    return tuple(a * n for a in v)


def vector_add(v0, v1):
    return tuple((v0[i] + v1[i]) for i in range(min(len(v0), len(v1))))


def binomial(n, k):
    p = 1
    for i in range(k):
        p *= (n - i) / (i + 1)
    return p


def bezier(t, _points):
    n = len(_points)
    if n <= 0 or t > 1 or t < 0:
        return tuple()

    result = vector_scale(_points[0], 0)
    for i in range(n):
        p = _points[i]
        v = vector_scale(p, binomial(n - 1, i) *
                         (t ** i) *
                         ((1 - t) ** ((n - 1) - i))
                         )
        result = vector_add(result, v)

    return result


def draw_bezier(_img, _points: list, step=(2 ** -8), radius_start=1, radius_end=1, color=(255, 255, 255), max_points=16):
    t = 0
    h, w, c = _img.shape
    area = h * w

    n = len(_points)
    while n > max_points >= 3:
        n = len(_points)
        _points.pop(0)

    while t <= 1:
        v = bezier(t, _points)
        if len(v) >= 2:
            x = int(v[0])
            y = int(v[1])

            c = float_to_int(hsv_to_rgb((t + 0.25) % 1, 0.8, 0.8)) + (128,)
            #c = float_to_int(hsv_to_rgb((x + y * w) / area, 0.8, 0.8)) + (128,)
            r = int(radius_start * log(1 + t) + radius_end * log(2 - t))
            cv2.circle(_img, (x, y), r, c, thickness=-1)
        t += (step / sqrt(n))


points = []
shape = (1024, 1024, 4)
tick = time()
do_tick = True
tick_delay = 0.01


def run_tick(event, x, y, flags, params):
    if event == 0:      # 0 = mouse movement
        do_tick = True
        tick = time()
        points.append((x, y))
        img = np.zeros(shape, dtype=np.uint8)
        draw_bezier(img, points, step=(2 ** -4), radius_start=8, radius_end=4, max_points=32)

        cv2.imshow('aj', img)


def click_event(event, x, y, flags, params):
    global do_tick
    global tick
    if (time() - tick) > tick_delay:
        do_tick = True
        run_tick(event, x, y, flags, params)
    else:
        while (time() - tick) < tick_delay:
            pass
        if do_tick:
            do_tick = False
            run_tick(event, x, y, flags, params)


if __name__ == '__main__':
    img = np.zeros(shape, dtype=np.uint8)
    cv2.imshow('aj', img)
    cv2.setMouseCallback('aj', click_event)
    cv2.waitKey(0)
