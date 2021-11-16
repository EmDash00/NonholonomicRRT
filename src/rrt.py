from collections import deque
from typing import List

import gr  # type: ignore
import numpy as np
from math import modf, sqrt
from numpy import asarray, cos, ndarray, pi, sin
from numpy.linalg import norm
from numpy.typing import ArrayLike

from rrtutil import dist  # type: ignore


class RRTNode(ndarray):
    def diff(self, n):
        x = self - n
        x[2] = (x[2] + 0.5) % 1 - 0.5

        return (x)

    def dist(self, n):
        a = self.diff(n)
        return np.sqrt(np.einsum('i,i', a, a))

    def dist2(self, n):
        return norm(self[:2] - n[:2])

    def __new__(cls, arr: ArrayLike):
        n = asarray(arr).view(cls)
        return (n)

    def __array_finalize__(self, obj):
        if obj is not None:
            self.parent = getattr(obj, 'parent', None)
            self.children = getattr(obj, 'children', deque())


def setup_graphics():
    gr.setviewport(xmin=0, xmax=1, ymin=0, ymax=1)
    gr.setwindow(xmin=0, xmax=1, ymin=0, ymax=1)

    gr.setmarkertype(gr.MARKERTYPE_SOLID_CIRCLE)
    gr.setmarkercolorind(86)  # Light grey

    gr.updatews()


def draw_root(n):
    gr.setmarkersize(1)
    gr.setmarkercolorind(60)
    gr.polymarker([n[0]], [n[1]])
    gr.setmarkercolorind(86)  # Light grey

    gr.setmarkersize(0.12)
    gr.updatews()


def draw_goal(n, tol):
    gr.setmarkersize(1)
    gr.setmarkercolorind(30)
    gr.polymarker([n[0]], [n[1]])

    x = tol * cos(2 * pi * np.linspace(0, 1, num=1000)) + n[0]
    y = tol * sin(2 * pi * np.linspace(0, 1, num=1000)) + n[1]

    gr.setlinetype(gr.LINETYPE_DOTTED)
    gr.polyline(x, y)

    gr.setlinetype(gr.LINETYPE_SOLID)
    gr.setmarkercolorind(86)  # Light grey

    gr.setmarkersize(0.12)

    gr.updatews()


def connect_node(n_kdtree, n: RRTNode):

    # O(log(n)) search when its balanced.
    kd_parent = n_kdtree.search_nn(n, dist)[0]
    n.parent = kd_parent.data

    diff = n.diff(n.parent)
    n[:] = n.parent + 0.03 * diff

    n.parent.children.append(n)
    kd_parent.add(n)

    gr.polyline([n.parent[0], n[0]], [n.parent[1], n[1]])
    gr.polymarker([n[0]], [n[1]])

    return (n)
