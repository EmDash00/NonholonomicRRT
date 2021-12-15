import gr  # type: ignore
import numpy as np
from numpy.random import rand
from threading import Thread
from ctypes import CFUNCTYPE
from time import perf_counter, sleep

import rrt
from mtree import MTree  # type: ignore
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


@CFUNCTYPE(None)
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

    return(rand(3))

def main():
    tol = 0.04
    mtree = MTree(dist, max_node_size=100)
    nodes = 1

    gr.updatews()
    thread = Thread(target=updatews, daemon=True)
    thread.start()

    try:
        root = RRTNode(rand(3))
        mtree.add(root)
        goal(root, tol)

        rrt.setup_graphics()
        rrt.draw_root(root)
        rrt.draw_goal([0.8, 0.8], tol)

        candidate = rrt.connect_node(mtree, RRTNode(sample(perf, tol)))

        while not goal(candidate, tol):
            nodes += 1
            print("Min Dist|Nodes: {:.3f}|{}".format(perf, nodes), end='\r')

            candidate = rrt.connect_node(mtree, RRTNode(sample(perf, tol)))

        # Candidate is the goal.
        print(candidate)
        print(dist(candidate, goal_p))

        print("Identified Solution in {} Nodes. Visualizing...".format(nodes))
        gr.setlinecolorind(77)

        chain_length = 1
        while candidate.parent is not None:
            chain_length += 1
            gr.polyline(candidate.path[:, 0], candidate.path[:, 1])

            gr.polymarker([candidate[0]], [candidate[1]])
            candidate = candidate.parent

        print("Solution visualized. Chain length:", chain_length)
        input()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
