from numba import njit  # type: ignore
from numpy import sqrt, floor


@njit(fastmath=True, cache=True)
def diff(n1, n2):
    x = n1 - n2
    x[2] += 0.5
    x[2] -= (floor(x[2]) + 0.5)

    return (x)


@njit(fastmath=True, cache=True)
def dist(n1, n2):
    angdiff = n1[2] - n2[2] + 0.5
    angdiff -= (floor(angdiff) + 0.5)

    return sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2 +
                (angdiff**2))


@njit(fastmath=True, cache=True)
def dist2(n1, n2):
    return sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)
