from numba import njit  # type: ignore
from numpy import sqrt, floor, cos, sin, array, min
from collections import deque
from numpy import ndarray, asarray
from numpy.typing import ArrayLike


class RRTNode(ndarray):
    def __new__(cls, arr: ArrayLike):
        n = asarray(arr).view(cls)
        return (n)

    # This is how you add properties to ndarray subclasses evidently.
    def __array_finalize__(self, obj):
        if obj is not None:
            # [phi, v, (start_t_idx, stop_t_idx), phi_idx]
            self.u = getattr(obj, 'u', [None] * 2)
            self.primative = getattr(obj, 'primative', None)
            self.path = getattr(obj, 'path', None)
            self.parent = getattr(obj, 'parent', None)

            # Deques have O(1) insertion at the end, no reallactions necessary!
            self.children = getattr(obj, 'children', deque())


@njit(fastmath=True, cache=True)
def norm(n):
    return sqrt(n[0]**2 + n[1]**2 + n[2]**2)


@njit(fastmath=True, cache=True)
def norm_squared(n):
    return n[0]**2 + n[1]**2 + n[2]**2


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

    return sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2 + (angdiff**2))


@njit(fastmath=True, cache=True)
def dist2(n1, n2):
    return sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)


@njit(fastmath=True, cache=True)
def rotate(n, theta):
    """
    Equivalent to (but faster than)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    n[:2] = R @ n[:2]

    return(n)
    """

    c = cos(theta)
    s = sin(theta)

    n0 = n[0]
    n1 = n[1]

    n[0] = n0 * c + n1 * s
    n[1] = -n0 * s + n1 * c

    return(n)

@njit(fastmath=True, cache=True)
def map_index(diff, N_v):
    return (min(int(floor(400 / 0.6 * norm_squared(diff))), N_v - 1))


@njit(fastmath=True, cache=True)
def rotate_arc(n, theta):

    R = array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])

    for i in range(n.shape[0]):
        n[i][:] = R @ n[i]

    return(n)
