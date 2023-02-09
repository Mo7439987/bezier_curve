import time

from classe_affichage import Affichage
from math import cos, sin
import cv2

if __name__ == '__main__':
    a = Affichage(window_name='omagus', show_control=False, tick_rate=256,
                  max_points=24, radius_start=16, radius_end=16,
                  color_s=1, color_v=1, color_a=255)

    x: float
    y: float

    def callback_function(event, x, y, flags, params):
        a.callback_method(event, x, y, flags, params)


    cv2.setMouseCallback(a.window_name, callback_function)
    cv2.waitKey(1)

    n = 0
    while True:
        h, w, d = a.img.shape
        n += 0.01
        x = cos(n) * 128 + w / 2
        y = sin(n) * 128 + h / 2
        #a.callback_method(None, x, y, None, None)
        cv2.waitKey(1)

