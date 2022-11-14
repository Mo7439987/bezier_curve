from classe_affichage import *
from matplotlib import pyplot as plt
from time import time, sleep
from random import random, randint

size = 1024
sample = 128
l = [0] * size
t0 = time()


def progress_bar(done, total, length=128, comment=""):
    bar_char = '\u2501'
    half_char = '\u257A'
    half_char = '\u2578'
    #half_char = '╺'
    border_char = '\u254D'
    ratio = done / total
    bar_length = int(length * ratio)
    full_part = bar_char * bar_length
    if (int(length * 2 * ratio) % 2) == 1:
        full_part += half_char
        bar_length += 1
    empty_part = ' ' * (length - bar_length)
    percents = str(round(ratio * 100, 2)).ljust(6) + '%'

    col_green = '\033[92m'
    col_reset = '\033[0m'
    bar = f'{border_char}{full_part + empty_part}{border_char}'
    bar = f'{bar} [{percents}] [{done}/{total}] {comment}'
    bar = f'{col_green}{bar}{col_reset}'
    return bar


def print_progress(done, total, length=128, comment=""):
    bar = progress_bar(done, total, length=length, comment=comment)
    if done <= total:
        print(f'\r{bar}', end='')
    else:
        print()


radius_x, radius_y = 512, 512


def random_point(rx, ry):
    return tuple((randint(-rx, rx) + 512, randint(-ry, ry) + 512))


t_total = 0
for i in range(1, size):
    n = 0
    #a = Affichage(max_points=i)
    #cv2.setMouseCallback(a.window_name, lambda event, x, y, flapgs, params: None)
    #cv2.waitKey(int(1000 * a.tick_delay))
    #cv2.destroyAllWindows()
    for _ in range(sample):
        points_temp = [random_point(radius_x, radius_y) for _ in range(i)]
        #a.points = points_temp
        x, y = random_point(radius_x, radius_y)
        t0 = time()
        #a.run_one_tick(0, x, y, 0, None)
        #a.render_one_frame()
        #cv2.imshow('', a.img)
        #a.tick_and_render(0, x, y, None, 0)
        #cv2.waitKey(int(1000 * a.tick_delay))
        img = bezier(random(), points_temp)
        #cv2.imshow('jejaj', img)
        t1 = time()
        n += (t1 - t0)
    t_total += n
    l[i] = (n / sample)
    #print(i, l[i])
    temp_comment = str(f"[{round(t_total, 2)}s]")
    print_progress(i + 1, size, comment=temp_comment)

fig, ax = plt.subplots()
out = ax.plot(list(range(size)), l)
plt.show()
