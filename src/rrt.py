from numpy import asarray, ndarray, pi
from numpy.linalg import norm
from numpy.typing import ArrayLike

tau = 2 * pi


class RRTNode(ndarray):
    def diff(self, n):
        x = self - n
        x[2] = (x[2] + pi) % (tau) - pi

        return (x)

    def dist(self, n):
        return norm(self.diff(n))

    def dist2(self, n):
        return norm(self[:2] - n[:2])

    def __new__(cls, arr: ArrayLike):
        n = asarray(arr).view(cls)
        return(n)

    def __array_finalize__(self, obj):
        if obj is not None:
            self.parent = getattr(obj, 'parent', None)
            self.children = getattr(obj, 'children', [])
