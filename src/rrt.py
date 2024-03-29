import gr  # type: ignore
import numpy as np
from mtree import MTree  # type: ignore
from numpy import empty, floor, pi
from numpy.random import rand

import workspace
from geom_prim import L, N_v, primative_tree, w
from rrtutil import (
    Rect, RRTNode, dist, map_index, rotate, rotate_arc, linear_interp
)
from workspace import DEBUG, DEBUG_COLL, base_conf, inter_conf

CURVE_RES = 3
diff = np.empty(3)


def sample(goal_p, min_dist, tol):
    """
    Use Goal-Region Biased Sampling. This is a form of rejection sampling
    where we sometimes sample in a region around the goal. The probability
    of sampling in the goal region vs. the entire workspace is a function
    of the minimum distance to the goal. Intuitively this is a crude way
    to implement exploration vs. exploitation.
    """

    p = tol / (min_dist)
    r = min_dist + 0.1

    if rand() <= p:
        x = rand(3)

        while dist(x, goal_p) > r:
            x = rand(3)

        return (x)
    else:
        return (rand(3))


def valid_path(path):
    for i in range(len(path) - 1):
        for p in linear_interp(path[i], path[i + 1], 5):
            rotate_arc(base_conf.data, p[2] * pi, out=inter_conf.data)
            inter_conf.data += p[:2]

            if DEBUG:
                gr.setlinecolorind(workspace.RED)
                inter_conf.draw()

            if workspace.is_cobst(inter_conf.data):
                return False

    if DEBUG:
        input()
        gr.setlinecolorind(1296)

    return True


def best_primative(nn, diff):
    i = map_index(diff, N_v)
    return (primative_tree[i].search(rotate(diff, -nn[2] * pi), 10))


def connect_node(mtree: MTree, n: RRTNode):
    global diff

    # n.parent = mtree.search(n)[0].obj

    # we no longer directly connect the node.
    # nn -> nearest neighbor/node from the RRT Tree
    nn = mtree.search(n)[0].obj

    # Calculate the displacement

    diff = n - nn

    # Angular difference in revolutions.
    diff[2] += 0.5
    diff[2] -= (floor(diff[2]) + 0.5)

    # Trick: Imagine the nn is at (x = y = theta = 0) with a rotated axes

    # This is the frame of reference of an observer at nn with angle theta
    # From the perspective of the world frame,
    # The x axis will be rotated counterclockwise by theta

    # Implementation: Calculate the displacement by subtracting
    # Find the best geometric primative given the diff rotated -theta
    # Rotate the primative back and add it.

    results = best_primative(nn, diff)

    for res in map(lambda x: x.obj, results):

        if res is None:
            break

        # Rotate the geometric primative so that tangents line up
        # n.primative is the primative that encodes the path
        path = rotate_arc(res.primative.copy(), nn[2] * pi) + nn

        # Check if the path generate is a valid one
        if valid_path(path):
            n[:] = path[-1]

            # n.u are the inputs necessary to reach the node
            # u[0] and u[1] are the angle and velocity respectively
            n.u = res.u

            # n.path is the path necessary to get from n.parent to n
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

    return (None)
