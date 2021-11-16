import gr  # type: ignore
import numpy as np
from numpy.random import rand
from silkworm.silktime import IntervalTimer  # type: ignore

import rrt
from mtree.mtree import MTree  # type: ignore
from rrt import RRTNode  # type: ignore
from rrtutil import dist, dist2  # type: ignore

goal_p = np.array([0.8, 0.8, 0.1])

perf = np.inf


def goal(n, tol):
    global perf

    d = dist(n, goal_p)
    perf = min(perf, d)

    return (d < tol)


def main():
    tol = 0.045
    mtree = MTree(dist, max_node_size=100)
    nodes = 1
    t = IntervalTimer(1 / 15, start=True)

    try:
        root = RRTNode(rand(3))
        mtree.add(root)

        rrt.setup_graphics()
        rrt.draw_root(root)
        rrt.draw_goal([0.8, 0.8], tol)

        candidate = rrt.connect_node(mtree, RRTNode(rand(3)))

        while not goal(candidate, tol):
            nodes += 1
            print(f"Min Dist|Nodes: {perf:.3f}|{nodes}", end='\r')

            candidate = rrt.connect_node(mtree, RRTNode(rand(3)))

            if t.tick():
                gr.updatews()

        # Candidate is the goal.
        print(candidate)
        print(dist(candidate, goal_p))

        print(f"Identified Solution in {nodes} Nodes. Visualizing...")
        gr.setlinecolorind(77)

        chain_length = 1
        while candidate.parent is not None:
            chain_length += 1
            gr.polyline([candidate[0], candidate.parent[0]],
                        [candidate[1], candidate.parent[1]])

            gr.polymarker([candidate[0]], [candidate[1]])
            candidate = candidate.parent
            gr.updatews()

        print("Solution visualized. Chain length:", chain_length)
        input()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
