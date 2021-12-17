from collections import deque
from time import perf_counter

import gr
import numpy as np
from mtree import MTree  # type: ignore

import rrt
import workspace
from geom_prim import RRTNode  # type: ignore
from workspace import goal_p, start_p, tol
from rrtutil import dist  # type: ignore
from workspace import LIVE

perf = np.inf


def goal(n, tol):
    global perf

    if n is None:
        return False

    d = dist(n, goal_p)
    perf = min(perf, d)

    return (d < tol)


def main():
    mtree = MTree(dist, max_node_size=100)
    nodes = 1

    try:
        gr.beginprint("soln.mp4")
        root = RRTNode(start_p)
        mtree.add(root)
        goal(root, tol)

        workspace.setup_graphics()
        workspace.draw_root()
        workspace.draw_goal()
        workspace.draw_obstacles()

        candidate = rrt.connect_node(mtree,
                                     RRTNode(rrt.sample(goal_p, perf, tol)))

        t0 = perf_counter()

        while not goal(candidate, tol):
            print("Min Dist|Nodes: {:.3f}|{}".format(perf, nodes), end='\r')

            candidate = rrt.connect_node(
                mtree, RRTNode(rrt.sample(goal_p, perf, tol)))

            if candidate is not None:
                nodes += 1

            if perf_counter() - t0 > 0.1:
                gr.updatews()
                t0 = perf_counter()

        # Candidate is the goal.
        print("Found solution:", candidate)
        print("Solution distance to goal:", dist(candidate, goal_p))

        print("Identified Solution in {} Nodes. Visualizing...".format(nodes))

        soln = [x for x in candidate.backtrack()][::-1]

        workspace.draw_soln(soln)
        gr.updatews()
        input()
        workspace.animate_soln(soln)
        gr.endprint()

        print("Solution visualized. Chain length:", len(soln))
        input()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
