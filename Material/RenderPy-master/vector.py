
import math
import numbers


class Vector(object):
    """A vector with useful vector / matrix operations, adapted for homogeneous coordinates."""

    def __init__(self, *args):
        # Adjust the constructor to handle an optional fourth component for homogeneous coordinates
        if len(args) == 0:
            self.components = (0, 0, 0, 1)  # Default to a point at the origin in homogeneous coordinates
        elif len(args) == 3:
            self.components = args + (1,)  # Treat as a point if three arguments are given
        else:
            self.components = args

    @property
    def x(self):
        return self.components[0]

    @x.setter
    def x(self, val):
        self.components = (val,) + self.components[1:]

    @property
    def y(self):
        return self.components[1]

    @y.setter
    def y(self, val):
        self.components = (self.components[0], val) + self.components[2:]

    @property
    def z(self):
        return self.components[2]

    @z.setter
    def z(self, val):
        self.components = self.components[:2] + (val,) + self.components[3:]

    @property
    def w(self):
        return self.components[3]

    @w.setter
    def w(self, val):
        self.components = self.components[:3] + (val,)

    def to_cartesian(self):
        """Converts the vector to Cartesian coordinates if it's in homogeneous coordinates."""
        if self.w != 0:
            return Vector(self.x / self.w, self.y / self.w, self.z / self.w)
        return Vector(self.x, self.y, self.z)

    def norm(self):
        """Return the norm (magnitude) of this vector, ignoring the homogeneous component."""
        squared_components = sum(math.pow(comp, 2) for comp in self.components[:3])
        return math.sqrt(squared_components)

    def normalize(self):
        """Return a normalized unit vector from this vector, maintaining the homogeneous component."""
        magnitude = self.norm()
        return Vector(*(comp / magnitude for comp in self.components[:3]) + (self.components[3],))

    def dot(self, other):
        """Return the dot product of this and another vector, ignoring the homogeneous component."""
        return sum(a * b for a, b in zip(self.components[:3], other.components[:3]))

    def cross(self, other):
        """Return the cross product of this and another vector, assuming 3D Cartesian coordinates."""
        assert len(self) == 4 and len(other) == 4, "Vectors must be in homogeneous coordinates."
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
            1  # The result is a point in space
        )

    # Overrides for arithmetic operations, adapted for homogeneous coordinates
    def __mul__(self, other):
        """If multiplied by another vector, return the dot product. If multiplied by a number, multiply each component by other."""
        if isinstance(other, Vector):
            return self.dot(other)
        elif isinstance(other, numbers.Real):
            return Vector(*(comp * other for comp in self.components))

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return Vector(*(comp / other for comp in self.components))

    def __add__(self, other):
        # Addition should consider the vectors as points if they are in homogeneous coordinates
        return Vector(*(a + b for a, b in zip(self.components, other.components)))

    def __sub__(self, other):
        # Subtraction should consider the vectors as points if they are in homogeneous coordinates
        return Vector(*(a - b for a, b in zip(self.components, other.components)))

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return iter(self.components)
