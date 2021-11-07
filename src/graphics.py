from typing import List

import gr  # type: ignore
import numpy as np
from numpy import pi, cos, sin
from numpy.typing import ArrayLike

from rrt import RRTNode  # type: ignore


def setup():
    gr.setviewport(xmin=0, xmax=1, ymin=0, ymax=1)
    gr.setwindow(xmin=-1, xmax=1, ymin=-1, ymax=1)

    gr.setmarkertype(gr.MARKERTYPE_SOLID_CIRCLE)
    gr.setmarkercolorind(86)  # Light grey
    gr.setmarkersize(1)

    gr.updatews()


def draw_root(n):
    gr.setmarkercolorind(60)  # Light grey
    gr.polymarker([n[0]], [n[1]])
    gr.setmarkercolorind(86)  # Light grey

    gr.updatews()


def draw_goal(n, tol):
    gr.setmarkercolorind(30)  # Light grey
    gr.polymarker([n[0]], [n[1]])

    x = tol * cos(2 * pi * np.linspace(0, 1, num=1000)) + n[0]
    y = tol * sin(2 * pi * np.linspace(0, 1, num=1000)) + n[1]

    gr.setlinetype(gr.LINETYPE_DOTTED)
    gr.polyline(x, y)

    gr.setlinetype(gr.LINETYPE_SOLID)
    gr.setmarkercolorind(86)  # Light grey

    gr.updatews()


def connect_node(n_kdtree, n: RRTNode):

    # n.parent = min(nodes, key=lambda x: n.dist2(x))  # type: ignore

    n.parent = n_kdtree.search_nn(n)[0].data

    diff = n.diff(n.parent)
    n[:] = n.parent + 0.03 * diff

    n.parent.children.append(n)

    n_kdtree.add(n)

    gr.setmarkersize(0.12)

    gr.polyline([n.parent[0], n[0]], [n.parent[1], n[1]])
    gr.polymarker([n[0]], [n[1]])

    gr.updatews()
