import numpy as np
from numpy import cos, sin, pi, ndarray, asarray
from numpy.typing import ArrayLike
from collections import deque
from rrtutil import dist, dist2
from mtree import MTree  # type: ignore
from os import path


class RRTNode(ndarray):
    def __new__(cls, arr: ArrayLike):
        n = asarray(arr).view(cls)
        return (n)

    # This is how you add properties to ndarray subclasses evidently.
    def __array_finalize__(self, obj):
        if obj is not None:
            # [phi, v, (start_t_idx, stop_t_idx), phi_idx]
            self.u = getattr(obj, 'u', [None] * 4)
            self.parent = getattr(obj, 'parent', None)

            # Deques have O(1) insertion at the end, no reallactions necessary!
            self.children = getattr(obj, 'children', deque())


L = 0.1
v = 0.5
T = 0.1  # Amount of time to simulate into the future

N_phi = 21  # Number of turning angles to generate
N_t = 100  # Resolution of Euler integration
dt = T / N_t
phi = np.linspace(-0.4 * np.pi, 0.4 * np.pi, N_phi)


def generate_primatives():
    primatives = np.zeros((N_t, N_phi, 3))

    for i in range(1, N_t):
        primatives[i, :, 0] = (
            primatives[i - 1, :, 0] +
            v * cos(phi) * cos(primatives[i - 1, :, 2] * 2 * pi) * dt
        )

        primatives[i, :, 1] = (
            primatives[i - 1, :, 1] +
            v * cos(phi) * sin(primatives[i - 1, :, 2] * 2 * pi) * dt
        )

        primatives[i, :, 2] = (
            primatives[i - 1, :, 2] +
            v / L * sin(phi) / (2 * pi) * dt
        )

    # Negative v is just a sign change.
    # Don't account for
    # np.save("primatives.npy", np.concatenate((primatives, -primatives)))
    np.save("primatives.npy", np.concatenate((primatives, -primatives)))


if __name__ == "__main__":
    generate_primatives()

if not path.exists("./primatives.npy"):
    generate_primatives()

# (timestep, phi, x/y/theta)
prims = np.load("primatives.npy")
primative_tree = MTree(dist)

for t in range(N_t):
    for i, row in enumerate(prims[t]):
        n = RRTNode(row)
        n.u[0] = phi[i] / (2 * np.pi)
        n.u[1] = v

        n.u[2] = (0, t)
        n.u[3] = i

        primative_tree.add(n)

for t in range(N_t, 2 * N_t):
    for i, row in enumerate(prims[t]):
        n = RRTNode(row)
        n.u[0] = phi[i] / (2 * np.pi)
        n.u[1] = -v

        n.u[2] = (N_t, t)
        n.u[3] = i

        primative_tree.add(n)
