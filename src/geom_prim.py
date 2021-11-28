import numpy as np
from numpy import cos, sin, pi
from rrtutil import dist
from mtree import MTree  # type: ignore
from os import path

L = 0.1
v = 1
T = 0.1  # Amount of time to simulate into the future

N_phi = 21  # Number of turning angles to generate
N_t = 10  # Resolution of Euler integration
dt = T / N_t


def generate_primatives():
    phi = np.linspace(-0.4 * np.pi, 0.4 * np.pi, N_phi)
    primatives = np.zeros((N_phi, 3))

    for i in range(N_t):
        primatives[:, 0] += (
            v * cos(phi) * cos(primatives[:, 2] * 2 * pi) * dt
        )

        primatives[:, 1] += (
            v * cos(phi) * sin(primatives[:, 2] * 2 * pi) * dt
        )

        primatives[:, 2] += v / L * sin(phi) / (2 * pi) * dt

    np.save("prim_ends.npy", np.concatenate((primatives, -primatives)))


if not path.exists("./prim_ends.npy"):
    generate_primatives()

prim_ends = np.load("prim_ends.npy")

primative_tree = MTree(dist)
for end in prim_ends:
    primative_tree.add(end)
