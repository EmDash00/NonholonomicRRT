import gr  # type: ignore
import numpy as np
from numpy import cos, pi, sin
from numpy import floor

from geom_prim import primative_tree, RRTNode, prims
from mtree import MTree  # type: ignore
from rrtutil import rotate, rotate_arc

diff = np.empty(3)


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
    diff[2] += 0.5
    diff[2] -= (floor(diff[2]) + 0.5)

    # Trick: Imagine the nn is at (x = y = theta = 0) with a rotated axes

    # This is the frame of reference of an observer at nn with angle theta
    # From the perspective of the world frame,
    # The x axis will be rotated counterclockwise by theta

    # Implementation: Calculate the displacement by subtracting
    # Find the best geometric primative given the diff rotated -theta
    # Rotate the primative back and add it.

    best = rotate(best_primative(nn, diff), nn[2] * 2 * pi)

    n[:] = nn + best
    n.u = best.u

    n.parent = nn
    n.parent.children.append(n)

    mtree.add(n)

    gr.polymarker([n[0]], [n[1]])
    curve = rotate_arc(
        prims[n.u[2][0]:n.u[2][1], n.u[3]][:, :2],
        -nn[2] * 2 * pi
    )

    gr.polyline(curve[:, 0] + nn[0], curve[:, 1] + nn[1])
    gr.updatews()
    input()

    return (n)
