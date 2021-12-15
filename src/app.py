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
from rrtutil import dist, dist2  # type: ignore

goal_p = np.array([0.8, 0.8, 0.1])

perf = np.inf


def goal(n, tol):
    global perf

    d = dist(n, goal_p)
    perf = min(perf, d)

    return (d < tol)


@CFUNCTYPE(None)
def updatews():
    t0 = 0

    while True:
        t0 = perf_counter()
        sleep(0.09)

        while perf_counter() - t0 < 0.1:
            pass

        gr.updatews()

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

        rrt.setup_graphics()
        rrt.draw_root(root)
        rrt.draw_goal([0.8, 0.8], tol)

        candidate = rrt.connect_node(mtree, RRTNode(rand(3)))

        while not goal(candidate, tol):
            nodes += 1
            print("Min Dist|Nodes: {:.3f}|{}".format(perf, nodes), end='\r')

            candidate = rrt.connect_node(mtree, RRTNode(rand(3)))

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
