import math
import numbers

import numpy as np


def get_rotation_matrix(axis: str, angle: float) -> np.array:
    """
    Returns the correct pre-populated rotation matrix based on the `axis`
    and `angle` (in degrees) provided.

    See:
    - https://www.cs.columbia.edu/~allen/F19/NOTES/homogeneous_matrices.pdf
    """

    angle = math.radians(angle)

    sin = math.sin(angle)
    cos = math.cos(angle)

    matrix = None

    match axis:
        # fmt: off
        case "x":
            matrix = [
                [1, 0, 0, 0],
                [0, cos, -sin, 0],
                [0, sin, cos, 0],
                [0, 0, 0, 1]
            ]
        case "y":
            matrix = [
                [cos, 0, sin, 0],
                [0, 1, 0, 0],
                [-sin, 0, cos, 0],
                [0, 0, 0, 1]
            ]
        case "z":
            matrix = [
                [cos, -sin, 0, 0],
                [sin, cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        # fmt: off

    return np.array(matrix)


class Vector(object):
    """A vector with useful vector / matrix operations."""

    DEFAULT_W = 1

    def __init__(self, *args):
        if len(args) == 0:
            self.components = np.array([0, 0, 0, self.DEFAULT_W])
        elif len(args) == 3:
            self.components = np.array([*args, self.DEFAULT_W])
        else:
            self.components = np.array(args)

    @property
    def x(self):
        assert len(self) >= 1
        return self.components[0]

    @x.setter
    def x(self, val):
        self.components[0] = val

    @property
    def y(self):
        assert len(self) >= 2
        return self.components[1]

    @y.setter
    def y(self, val):
        self.components[1] = val

    @property
    def z(self):
        assert len(self) >= 3
        return self.components[2]

    @z.setter
    def z(self, val):
        self.components[2] = val

    @property
    def w(self):
        assert len(self) >= 4
        return self.components[3]

    @w.setter
    def w(self, val):
        self.components[3] = val

    @property
    def xyz(self):
        assert len(self) >= 3
        return self.components[:3]

    def norm(self):
        """Return the norm (magnitude) of this vector."""
        return np.linalg.norm(self.xyz)

    def normalize(self) -> "Vector":
        """Return a normalized unit vector from this vector."""
        magnitude = self.norm()
        return Vector(*(self.xyz / magnitude), self.w)

    def dot(self, other: "Vector"):
        """Return the dot product of this and another vector."""
        return np.dot(self.xyz, other.xyz)

    def cross(self, other: "Vector") -> "Vector":
        """Return the cross product of this and another vector."""
        assert len(self) == len(other), "Vectors must be the same size."
        assert len(self) == 4, "Cross product only implemented for 3D vectors."
        return Vector(*np.cross(self.xyz, other.xyz), self.DEFAULT_W)

    def translate(self, dx: int, dy: int, dz: int) -> "Vector":
        """
        -- Problem 1 Question 3 --

        Returns a new `Vector` equal to this one translated in the x, y and
        z axes by `dx`, `dy`, `dz` respectively.
        """

        matrix = np.array(
            [
                [1, 0, 0, dx],
                [0, 1, 0, dy],
                [0, 0, 1, dz],
                [0, 0, 0, 1],
            ]
        )

        new = np.matmul(matrix, self.components)

        return Vector(*new)

    def rotate(self, **kwargs) -> "Vector":
        """
        -- Problem 1 Question 3 --

        Return a new vector equal to this one rotated either:
        1) around `axis` by `angle`
        2) by a `matrix`
        """

        angle = kwargs.get("angle")
        axis = kwargs.get("axis")
        matrix = kwargs.get("matrix")

        # Anonymous function to check if param was provided (i.e. key-value
        # pair was found)
        provided = lambda x: x is not None

        # The two use-cases of this function
        case1 = provided(angle) and provided(axis)
        case2 = provided(matrix)
        assert case1 or case2

        r_matrix = get_rotation_matrix(axis, angle) if case1 else matrix
        vector = self.xyz if (case2 and r_matrix.shape == (3, 3)) else self.components

        rotated = np.matmul(r_matrix, vector)

        return Vector(*rotated)

    def scale(self, sx, sy, sz):
        """
        -- Problem 1 Question 3 --

        Returns a new `Vector` equal to this one scaled in the x, y and
        z axes by `sx`, `sy`, `sz` respectively.
        """

        matrix = np.array(
            [
                [sx, 0, 0, 0],
                [0, sy, 0, 0],
                [0, 0, sz, 0],
                [0, 0, 0, 1],
            ]
        )

        scaled = np.matmul(matrix, self.components)
        vector = Vector(*scaled)

        return vector

    # Overrides
    def __mul__(self, other):
        """If multiplied by another vector, return the dot product.
        If multiplied by a number, multiply each component by other.
        """
        if type(other) == type(self):
            return self.dot(other)
        elif isinstance(other, numbers.Real):
            return Vector(*(self.components * other))

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return Vector(*(self.components / other))

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(*(self.components + other.components))

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(*(self.components - other.components))

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return self.components.__iter__()
