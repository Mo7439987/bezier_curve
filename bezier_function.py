from math import factorial
import cv2
import numpy as np
from random import random, randint
from math import sin, cos, sqrt
from colorsys import hsv_to_rgb


def float_to_int(c):
    return tuple(int(a * 255) for a in c)


def vector_scale(v, n):
    return tuple(a * n for a in v)


def vector_add(v0, v1):
    return tuple((v0[i] + v1[i]) for i in range(min(len(v0), len(v1))))


def binomial(n, k):
    f = factorial
    return f(n) / (f(k) * f(n - k))


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


def draw_bezier(_img, _points: list, step=(2 ** -8), radius=1, color=(255, 255, 255), max_points=16):
    t = 0
    h, w, c = _img.shape

    n = len(_points)
    while n > max_points >= 3:
        n = len(_points)
        #print(n, end='')
        _points.pop(0)

    while t <= 1:
        v = bezier(t, _points)
        if len(v) >= 2:
            x = int(v[0])
            y = int(v[1])
            c = float_to_int(hsv_to_rgb(t, 0.8, 0.8))
            cv2.circle(_img, (x, y), radius, c, thickness=-1)
            #_img[y, x] = c
        t += (step / sqrt(n))


points = []
print(id(points))
iterations = 0


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN or True:
        global iterations
        iterations += 1
        print(f'click {(x, y, len(points)), iterations}')
        #print(id(points))
        points.append((x, y))
        shape = (1024, 1024, 3)
        img = np.zeros(shape, dtype=np.uint8)
        draw_bezier(img, points, step=(2**-3), radius=8, max_points=32)

        cv2.imshow('aj', img)
        #cv2.setMouseCallback('aj', click_event)
        #cv2.waitKey(0)
    else:
        print(event)


if __name__ == '__main__':
    l = [(random() * 1024, random() * 1024) for _ in range(1024)]
    for _ in range(1023):
        l.pop(0)
    shape = (1024, 1024, 3)
    img = np.zeros(shape, dtype=np.uint8)
    cv2.imshow('aj', img)
    cv2.setMouseCallback('aj', click_event)
    cv2.waitKey(0)
