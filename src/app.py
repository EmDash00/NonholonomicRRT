import gr  # type: ignore
import kdtree  # type: ignore
import numpy as np
from numpy.random import rand
from silkworm.silktime import IntervalTimer  # type: ignore

import rrt
from rrt import RRTNode  # type: ignore
from rrtutil import dist, dist2  # type: ignore

goal_p = np.array([0.8, 0.8, 0.1])


def goal(n, tol):
    return (dist2(n, goal_p) < tol)


def main():
    tol = 0.01
    n_kdtree = kdtree.create(dimensions=3)
    nodes = 1
    t = IntervalTimer(1 / 120, start=True)

    try:
        root = RRTNode(rand(3))
        n_kdtree.add(root)

        rrt.setup_graphics()
        rrt.draw_root(root)
        rrt.draw_goal([0.8, 0.8], tol)

        candidate = rrt.connect_node(n_kdtree, RRTNode(rand(3)))

        while not goal(candidate, tol):
            nodes += 1

            if nodes % 5000 == 0:
                print(nodes)
                n_kdtree = n_kdtree.rebalance()

            candidate = rrt.connect_node(n_kdtree, RRTNode(rand(3)))

            if t.tick():
                gr.updatews()

        # Candidate is the goal.
        print(candidate)
        print(candidate.dist([0.8, 0.8, 0.1]))

        print("Identified Solution. Visualizing...")
        gr.setlinecolorind(77)

        while candidate.parent is not None:
            gr.polyline([candidate[0], candidate.parent[0]],
                        [candidate[1], candidate.parent[1]])

            candidate = candidate.parent
            gr.updatews()

        print("Solution visualized.")
        input()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
