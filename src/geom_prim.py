import numpy as np
from numpy import cos, sin, pi, floor
from rrtutil import RRTNode, dist, dist2
from mtree import MTree  # type: ignore
from os import path


L = 0.1
v_max = 1
T = 0.1  # Amount of time to simulate into the future

N_phi = 21  # Number of turning angles to generate
N_t = 100  # Resolution of Euler integration
N_v = 100  # Number of velocities to integrate over
dt = T / N_t
phi = np.linspace(-0.4 * np.pi, 0.4 * np.pi, N_phi)


def generate_primatives():
    primatives = np.zeros((N_t, N_v, N_phi, 3))
    neg_primatives = np.zeros_like(primatives)

    for i in range(1, N_t):
        for j in range(N_v):
            v = i * v_max / N_v

            primatives[i, j, :, 0] = (
                primatives[i - 1, j, :, 0] +
                v * cos(phi) * cos(primatives[i, j - 1, :, 2] * 2 * pi) * dt
            )

            primatives[i, j, :, 1] = (
                primatives[i - 1, j, :, 1] +
                v * cos(phi) * sin(primatives[i, j - 1, :, 2] * 2 * pi) * dt
            )

            primatives[i, j, :, 2] = (
                primatives[i - 1, j, :, 2] +
                v / L * sin(phi) / (2 * pi) * dt
            )

    # Caveat, primatives moving in the opposite direction have
    # Orientation rotated by pi/2 (0.5 revs)
    neg_primatives[..., :2] = -primatives[..., :2]
    neg_primatives[..., 2] = primatives[..., 2] + 0.5
    neg_primatives[..., 2] -= floor(neg_primatives[..., 2])

    # Negative v is just a sign change.
    np.save("primatives.npy", np.concatenate((primatives, neg_primatives)))


if __name__ == "__main__":
    generate_primatives()

if not path.exists("./primatives.npy"):
    generate_primatives()

# (timestep, phi, x/y/theta)
prims = np.load("primatives.npy")
prims_pos = prims[0:N_t]
prims_neg = prims[N_t:2 * N_t]

primative_tree = [MTree(dist) for i in range(500)]

for i in range(N_v):
    v = i * v_max / N_v
    for j in range(N_phi):
        n = RRTNode(prims_pos[-1, i, j])
        n.u[0] = phi[j]
        n.u[1] = v

        n.u[2] = prims_pos[:, i, j]
        n.u[3] = i

        primative_tree[i].add(n)

        n_neg = RRTNode(prims_neg[-1, i, j])
        n_neg.u[0] = phi[j]
        n_neg.u[1] = v

        n_neg.u[2] = prims_neg[:, i, j]
        n_neg.u[3] = i

        primative_tree[i].add(n_neg)
