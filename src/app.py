from threading import Thread

import gr  # type: ignore
import numpy as np
from mtree import MTree  # type: ignore
from numpy.random import rand

import rrt
import workspace
from geom_prim import RRTNode  # type: ignore
from geom_prim import primative_tree
from rrtutil import dist, dist2, norm2_squared  # type: ignore

goal_p = np.array([0.8, 0.8, 0.1])

perf = np.inf


def goal(n, tol):
    global perf

    d = dist(n, goal_p)
    perf = min(perf, d)

    return (d < tol)


def sample(min_dist, tol):
    """
    Use Goal-Region Biased Sampling. This is a form of rejection sampling
    where we sometimes sample in a region around the goal. The probability
    of sampling in the goal region vs. the entire workspace is a function
    of the minimum distance to the goal. Intuitively this is a crude way
    to implement exploration vs. exploitation.

    Note: I couldn't get this to work properly. While it does sample
    more often around the goal, this doesn't seem to improve performance due
    to kinematic constraints.
    """
    """
    p = tol / min_dist
    r = min_dist

    if rand() <= p:
        x = rand(3)

        while norm2_squared(x) > r:
            x = rand(3)

        x[:2] += goal_p[:2]

        return(x)
    else:
        return (rand(3))
    """

    return (rand(3))


def main():
    tol = 0.04
    mtree = MTree(dist, max_node_size=100)
    nodes = 1

    gr.updatews()
    thread = Thread(target=workspace.updatews, daemon=True)
    thread.start()

    try:
        root = RRTNode(rand(3))
        mtree.add(root)
        goal(root, tol)

        workspace.setup_graphics()
        workspace.draw_root(root)
        workspace.draw_goal([0.8, 0.8], tol)

        candidate = rrt.connect_node(mtree, RRTNode(sample(perf, tol)))

        while not goal(candidate, tol):
            nodes += 1
            print("Min Dist|Nodes: {:.3f}|{}".format(perf, nodes), end='\r')

            candidate = rrt.connect_node(mtree, RRTNode(sample(perf, tol)))

        # Candidate is the goal.
        print("Found solution:", candidate)
        print("Solution distance to goal:", dist(candidate, goal_p))

        print("Identified Solution in {} Nodes. Visualizing...".format(nodes))

        chain_length = workspace.draw_soln(candidate)

        print("Solution visualized. Chain length:", chain_length)
        input()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
