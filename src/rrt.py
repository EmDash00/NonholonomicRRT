import gr  # type: ignore
import numpy as np
from mtree import MTree  # type: ignore
from numpy import cos, floor, pi, sin

from geom_prim import N_v, primative_tree
from rrtutil import RRTNode, map_index, rotate, rotate_arc
from workspace import DEBUG

CURVE_RES = 3
diff = np.empty(3)


def best_primative(nn, diff):
    i = map_index(diff, N_v)
    return (primative_tree[i].search(rotate(diff, -nn[2] * pi))[0].obj)


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
    # n.path is the path necessary to get from n.parent to n
    # n.primative is the primative that encodes the path

    # Rotate the geometric primative so that tangents line up
    path = rotate_arc(best_prim.primative[:, :2].copy(),
                      nn[2] * pi) + nn[:2]

    n[:2] = path[-1]
    n[2] = nn[2] + best_prim.primative[-1, 2]

    n.u = best_prim.u

    n.path = path

    n.parent = nn
    n.parent.children.append(n)

    mtree.add(n)

    gr.polyline(n.path[:, 0], n.path[:, 1])

    if DEBUG:
        dx = np.cos(n[2] * pi) / 30
        dy = np.sin(n[2] * pi) / 30

        gr.setlinecolorind(20)
        gr.drawarrow(n[0], n[1], n[0] + dx, n[1] + dy)
        gr.setlinecolorind(1296)

        gr.polymarker([n[0]], [n[1]])

        input()
        gr.updatews()

    return (n)
