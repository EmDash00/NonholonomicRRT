from ctypes import CFUNCTYPE
from threading import Thread
from time import sleep

import gr
import numpy as np
from numpy import cos, pi, sin

NODE_MARKER_SIZE = 0.12

LIGHT_GRAY = 86
GREEN = 60
RED = 30
BLUE = 77


def setup_graphics():
    gr.setviewport(xmin=0, xmax=1, ymin=0, ymax=1)
    gr.setwindow(xmin=-0.1, xmax=1.1, ymin=-0.1, ymax=1.1)

    gr.setmarkertype(gr.MARKERTYPE_SOLID_CIRCLE)
    gr.setmarkercolorind(LIGHT_GRAY)  # Light grey
    gr.setarrowstyle(6)
    gr.setarrowsize(0.5)

    gr.updatews()

    thread = Thread(target=updatews, daemon=True)
    thread.start()


def draw_root(n):
    gr.setmarkersize(1)
    gr.setmarkercolorind(GREEN)
    gr.polymarker([n[0]], [n[1]])
    gr.setmarkercolorind(LIGHT_GRAY)

    gr.setmarkersize(NODE_MARKER_SIZE)
    gr.updatews()


def draw_goal(n, tol):
    gr.setmarkersize(1)
    gr.setmarkercolorind(RED)
    gr.polymarker([n[0]], [n[1]])

    x = tol * cos(2 * pi * np.linspace(0, 1, num=1000)) + n[0]
    y = tol * sin(2 * pi * np.linspace(0, 1, num=1000)) + n[1]

    gr.setlinetype(gr.LINETYPE_DOTTED)
    gr.polyline(x, y)

    gr.setlinetype(gr.LINETYPE_SOLID)
    gr.setmarkercolorind(LIGHT_GRAY)

    gr.setmarkersize(NODE_MARKER_SIZE)

    gr.updatews()


def draw_soln(n):

    gr.setlinecolorind(BLUE)

    chain_length = 0

    while n.parent is not None:
        chain_length += 1
        gr.polyline(n.path[:, 0], n.path[:, 1])

        gr.polymarker([n[0]], [n[1]])
        n = n.parent

    return (chain_length)


def updatews():
    """
    This function is the target of a daemon thread that updates the GR
    workspace. I assume there's some buffer that all the draw events get
    loaded into. This flushes it onto the screen.

    There's a couple caveats about how this should be done. I use the
    @CFUNCTYPE decorator from ctypes to make this a native C function. This
    means that it won't hold the GIL.

    https://realpython.com/python-gil/

    We also sleep for the majority of this thread when not updating the
    workspace. The use of sleeps allows the OS thread scheduler to run
    other threads.

    I chose an update rate of 10 FPS for performance reasons.
    """

    while True:
        sleep(0.1)
        gr.updatews()
