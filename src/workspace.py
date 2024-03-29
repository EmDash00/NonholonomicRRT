from collections import deque
from itertools import islice
from threading import Thread
from time import sleep

import gr  # type: ignore
import numpy as np
from numpy import cos, pi, sin
from numpy.random import rand
from time import perf_counter

from geom_prim import L, w
from rrtutil import Rect, rotate_arc, linear_interp

N_obst = 30
obst_side = 0.05

DEBUG = False
DEBUG_COLL = True

NODE_MARKER_SIZE = 0.12

LIGHT_GRAY = 86
GREEN = 60
RED = 30
BLUE = 77
LIVE = True

obstacles = deque(
    (Rect(rand(2) * (0.8 - obst_side) + 0.1, bounds=(obst_side, obst_side)))
    for i in range(N_obst))

obstacles.append(Rect([0.0, 0.4], bounds=(0.1, 0.2)))
obstacles.append(Rect([0.4, 0.0], bounds=(0.2, 0.1)))

base_conf = Rect((0, 0), (L, w))
inter_conf = Rect((0, 0), (L, w))  # intermediate configuration

goal_p = np.array([0.9, 0.9, 0.5])
start_p = np.array([0.1, 0.1, 0])

tol = 0.02

def is_cobst(conf):
    """
    Check if a point is in C_obst.
    """

    for obst in obstacles:
        if obst.intersects(conf):
            return True

    return False


def setup_graphics():
    gr.setviewport(xmin=0, xmax=1, ymin=0, ymax=1)
    gr.setwindow(xmin=-0.1, xmax=1.1, ymin=-0.1, ymax=1.1)

    gr.setmarkertype(gr.MARKERTYPE_SOLID_CIRCLE)
    gr.setmarkercolorind(LIGHT_GRAY)  # Light grey
    gr.setarrowstyle(6)
    gr.setarrowsize(0.5)
    gr.setfillcolorind(LIGHT_GRAY)
    gr.setfillintstyle(1)

    gr.updatews()


def draw_root():
    gr.setmarkersize(1)
    gr.setmarkercolorind(GREEN)
    gr.polymarker([start_p[0]], [start_p[1]])
    gr.setmarkercolorind(LIGHT_GRAY)

    gr.setmarkersize(NODE_MARKER_SIZE)


def draw_goal():
    gr.setmarkersize(1)
    gr.setmarkercolorind(RED)
    gr.polymarker([goal_p[0]], [goal_p[1]])

    x = tol * cos(2 * pi * np.linspace(0, 1, num=1000)) + goal_p[0]
    y = tol * sin(2 * pi * np.linspace(0, 1, num=1000)) + goal_p[1]

    gr.setlinetype(gr.LINETYPE_DOTTED)
    gr.polyline(x, y)

    gr.setlinetype(gr.LINETYPE_SOLID)
    gr.setmarkercolorind(LIGHT_GRAY)

    gr.setmarkersize(NODE_MARKER_SIZE)


def draw_soln(soln):

    gr.setlinecolorind(BLUE)

    for n in soln[1:]:
        gr.polyline(n.path[:, 0], n.path[:, 1])
        gr.polymarker([n[0]], [n[1]])


def animate_soln(soln):
    for n in soln[1:]:
        for i in range(len(n.path) - 1):
            for p in linear_interp(n.path[i], n.path[i + 1], 20):
                gr.clearws()
                gr.setfillcolorind(LIGHT_GRAY)
                draw_root()
                draw_goal()
                draw_obstacles()
                gr.setfillcolorind(GREEN)

                rotate_arc(base_conf.data, p[2] * pi, out=inter_conf.data)

                inter_conf.data += p[:2]
                inter_conf.draw()

                draw_soln(soln)

                t0 = perf_counter()

                gr.updatews()
                while perf_counter() - t0 < (1 / 120):
                    pass


def draw_obstacles():
    for obstacle in obstacles:
        obstacle.draw()


def updatews():
    """
    This function is the target of a daemon thread that updates the GR
    workspace. I assume there's some buffer that all the draw events get
    loaded into. This flushes it onto the screen.

    We also sleep for the majority of this thread when not updating the
    workspace. The use of sleeps allows the OS thread scheduler to run
    other threads.

    I chose an update rate of 10 FPS for performance reasons.
    """

    while LIVE:
        sleep(0.1)
        gr.updatews()
