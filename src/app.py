from typing import List

from numpy.random import rand

import graphics
from rrt import RRTNode  # type: ignore

import kdtree  # type: ignore


def goal(n, tol):
    return (n.dist([0.8, 0.8, 0.1]) < tol)


def main():
    tol = 0.05
    n_kdtree = kdtree.create(dimensions=3)

    try:
        root = RRTNode(rand(3))
        n_kdtree.add(root)

        graphics.setup()
        graphics.draw_root(root)
        graphics.draw_goal([0.8, 0.8], tol)

        candidate = RRTNode(rand(3))
        graphics.connect_node(n_kdtree, candidate)

        while not goal(candidate, tol):
            candidate = RRTNode(rand(3))
            graphics.connect_node(n_kdtree, candidate)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
