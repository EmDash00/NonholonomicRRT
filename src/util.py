import numpy as np
from numba.pycc import CC  # type: ignore
from math import sqrt
from numpy.linalg import norm

cc = CC("rrtutil")


@cc.export('diff', 'f8[:](f8[:], f8[:])')
def diff(n1, n2):
    x = n1 - n2
    x[2] = (x[2] + 0.5) % 1 - 0.5

    return (x)


@cc.export('dist', 'f8(f8[:], f8[:])')
def dist(n1, n2):
    return sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2 +
                (((n1[2] - n2[2]) + 0.5) % 1 - 0.5)**2)


@cc.export('dist2', 'f8(f8[:], f8[:])')
def dist2(n1, n2):
    return sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)


if __name__ == '__main__':
    cc.compile()
