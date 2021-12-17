import numpy as np
from mtree import MTree  # type: ignore

import rrt
from rrt import goal_p, start_p
import workspace
from geom_prim import RRTNode  # type: ignore
from rrtutil import dist  # type: ignore


perf = np.inf


def goal(n, tol):
    global perf

    if n is None:
        return False

    d = dist(n, goal_p)
    perf = min(perf, d)

    return (d < tol)


def main():
    tol = 0.02
    mtree = MTree(dist, max_node_size=100)
    nodes = 1

    try:
        root = RRTNode(start_p)
        mtree.add(root)
        goal(root, tol)

        workspace.setup_graphics()
        workspace.draw_root(root)
        workspace.draw_goal(goal_p[:2], tol)
        workspace.draw_obstacles()

        candidate = rrt.connect_node(mtree, RRTNode(
            rrt.sample(goal_p, perf, tol))
        )

        while not goal(candidate, tol):
            nodes += 1
            print("Min Dist|Nodes: {:.3f}|{}".format(perf, nodes), end='\r')

            candidate = rrt.connect_node(
                mtree, RRTNode(rrt.sample(goal_p, perf, tol))
            )

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
