from collections import deque

import gr  # type: ignore
import numpy as np
from numpy import asarray, cos, ndarray, pi, sin
from numpy.typing import ArrayLike

from geom_prim import primative_tree
from mtree import MTree  # type: ignore
from rrtutil import dist, rotate

diff = np.empty(3)


class RRTNode(ndarray):
    def __new__(cls, arr: ArrayLike):
        n = asarray(arr).view(cls)
        return (n)

    # This is how you add properties to ndarray subclasses evidently.
    def __array_finalize__(self, obj):
        if obj is not None:
            # Deques have O(1) insertion at the end, no reallactions necessary!
            self.u = getattr(obj, 'u', np.empty(2))
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


def best_primative(nn, diff):
    return (
        primative_tree.search(rotate(diff, -nn[2] * 2 * pi))[0].obj
    )


def connect_node(mtree: MTree, n: RRTNode):
    global diff

    # n.parent = mtree.search(n)[0].obj

    # we no longer directly connect the node.
    # nn -> nearest neighbor/node from the RRT Tree
    nn = mtree.search(n)[0].obj

    # Calculate the displacement

    diff = n - nn
    n[2] = (n[2] + 0.5) % 1 - 0.5  # Angle subtraction

    # Trick: Imagine the nn is at (x = y = theta = 0) with a rotated axes

    # This is the frame of reference of an observer at nn with angle theta
    # From the perspective of the world frame,
    # The x axis will be rotated counterclockwise by theta

    # Implementation: Calculate the displacement by subtracting
    # Find the best geometric primative given the diff rotated -theta
    # Rotate the primative back and add it.

    n[:] = nn + rotate(best_primative(nn, diff), nn[2])
    n.parent = nn
    n.parent.children.append(n)

    mtree.add(n)

    gr.polyline([n.parent[0], n[0]], [n.parent[1], n[1]])

    return (n)
