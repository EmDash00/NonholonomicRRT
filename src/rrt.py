import gr  # type: ignore
import numpy as np
from numpy import cos, pi, sin
from numpy import floor

from geom_prim import primative_tree, N_v
from mtree import MTree  # type: ignore
from rrtutil import RRTNode, rotate, rotate_arc, norm

DEBUG = False
CURVE_RES = 3
diff = np.empty(3)

def setup_graphics():
    gr.setviewport(xmin=0, xmax=1, ymin=0, ymax=1)
    gr.setwindow(xmin=0, xmax=1, ymin=0, ymax=1)

    gr.setmarkertype(gr.MARKERTYPE_SOLID_CIRCLE)
    gr.setmarkercolorind(86)  # Light grey
    gr.setarrowstyle(6)
    gr.setarrowsize(0.5)

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
    dist = norm(diff)
    i = min(int(floor(400 / 0.6 * dist**2)), N_v - 1)
    return (
        primative_tree[i].search(rotate(diff, -nn[2] * 2 * pi))[0].obj
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

    best_prim = best_primative(nn, diff)

    # The u property encodes some metadata about the node
    # u[0] and u[1] are the angle and velocity necessary to reach the node
    # u[2] is the geometric primative that should be added to the parent
    # to reach this node
    # u[3] is the index of the velocity

    # Rotate the geometric primative so that tangents line up
    path = rotate_arc(
        best_prim.primative[:, :2].copy(),
        nn[2] * 2 * pi
    ) + nn[:2]

    n[:2] = path[-1]
    n[2] = nn[2] + best_prim.primative[-1, 2]
    n[2] -= floor(n[2]) # normalize angles to [0, 1] 1.1 -> 1

    n.u = best_prim.u

    n.path = path

    n.parent = nn
    n.parent.children.append(n)

    mtree.add(n)

    if DEBUG:
        dx = np.cos(n[2] * 2 * pi) / 30
        dy = np.sin(n[2] * 2 * pi) / 30

        gr.setlinecolorind(20)
        gr.drawarrow(n[0], n[1], n[0] + dx, n[1] + dy)
        gr.setlinecolorind(1296)

        gr.polymarker([n[0]], [n[1]])

        input()
        gr.updatews()

    gr.polyline(n.path[::3, 0], n.path[::3, 1])

    return (n)
