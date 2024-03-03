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
        length = len(args)
        if length == 0:
            self.components = (0, 0)
        elif len(args) == 3:
            self.components = (*args, self.DEFAULT_W)
        else:
            self.components = args

    @property
    def x(self):
        assert len(self) >= 1
        return self.components[0]

    @x.setter
    def x(self, val):
        self.components = (val, self.components[1], self.components[2])

    @property
    def y(self):
        assert len(self) >= 2
        return self.components[1]

    @y.setter
    def y(self, val):
        self.components = (self.components[0], val, self.components[2])

    @property
    def z(self):
        assert len(self) >= 3
        return self.components[2]

    @z.setter
    def z(self, val):
        self.components = (self.components[0], self.components[1], val)

    @property
    def w(self):
        assert len(self) >= 4
        return self.components[3]

    @w.setter
    def w(self, val):
        self.components = (
            self.components[0],
            self.components[1],
            self.components[2],
            val,
        )

    @property
    def xyz(self):
        assert len(self) >= 3
        return self.components[:3]

    @property
    def np_array(self):
        return np.array(self.components)

    def norm(self):
        """Return the norm (magnitude) of this vector."""
        squaredComponents = sum(math.pow(comp, 2) for comp in self.xyz)
        return math.sqrt(squaredComponents)

    def normalize(self):
        """Return a normalized unit vector from this vector."""
        magnitude = self.norm()
        return Vector(*[comp / magnitude for comp in self.xyz], self.w)

    def dot(self, other):
        """Return the dot product of this and another vector."""
        return sum(a * b for a, b in zip(self.xyz, other.xyz))

    def cross(self, other):
        """Return the cross product of this and another vector."""
        assert len(self) == len(other), "Vectors must be the same size."
        assert len(self) == 4, "Cross product only implemented for 3D vectors."
        return Vector(
            (self.y * other.z - self.z * other.y),
            (self.z * other.x - self.x * other.z),
            (self.x * other.y - self.y * other.x),
            self.DEFAULT_W,
        )

    def translate(self, dx: int, dy: int, dz: int):
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

        new = np.matmul(matrix, self.np_array)

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
        vector = (
            np.array(self.xyz)
            if (case2 and r_matrix.shape == (3, 3))
            else self.np_array
        )

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

        scaled = np.matmul(matrix, self.np_array)
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
            product = tuple(comp * other for comp in self)
            return Vector(*product)

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            value = tuple(comp / other for comp in self)
            return Vector(*value)

    def __add__(self, other):
        value = tuple(a + b for a, b in zip(self, other))
        return Vector(*value)

    def __sub__(self, other):
        value = tuple(a - b for a, b in zip(self, other))
        return Vector(*value)

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return self.components.__iter__()


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def identity(cls):
        return cls(1, 0, 0, 0)

    def get_conjugate(self):
        """
        --- Problem 2 Question 2 Part 3 ---

        Returns the conjugate (i.e. inverse rotation) of this `Quaternion`
        """
        w, x, y, z = self
        return Quaternion(w, -x, -y, -z)

    def get_magnitude(self):
        # fmt: off
        return math.sqrt(
            sum((
                math.pow(self.w, 2),
                math.pow(self.x, 2),
                math.pow(self.y, 2),
                math.pow(self.z, 2),
            ))
        )
        # fmt: on

    def normalise(self) -> None:
        d = self.get_magnitude()

        self.w /= d
        self.x /= d
        self.y /= d
        self.z /= d

    def __repr__(self):
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    def __iter__(self):
        return iter((self.w, self.x, self.y, self.z))

    def __add__(self, other) -> "Quaternion":
        w1, x1, y1, z1 = self

        if type(other) == type(self):
            w2, x2, y2, z2 = other
            return Quaternion(w1 + w2, x1 + x2, y1 + y2, z1 + z2)

        elif isinstance(other, numbers.Real):
            return Quaternion(w1 + other, x1 + other, y1 + other, z1 + other)

    def __mul__(self, other):
        """
        --- Problem 2 Question 2 Part 4 ---

        Calculates the product of this `Quaternion` and `other`
        """

        w1, x1, y1, z1 = self

        # If both are Quaternions:
        if type(other) == type(self):
            w2, x2, y2, z2 = other

            # fmt: off
            return Quaternion(
                w = (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),
                x = (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),
                y = (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),
                z = (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2),
            )
            # fmt: on

        # If 'other' is a scalar:
        elif isinstance(other, numbers.Real):
            return Quaternion(
                w1 * other,
                x1 * other,
                y1 * other,
                z1 * other,
            )

    def __imul__(self, other: "Quaternion"):
        """
        Implement behaviour of the *= operator
        """
        self.w, self.x, self.y, self.z = self * other
        return self

    def to_euler_angles(self) -> "EulerAngles":
        """
        --- Problem 2 Question 2 Part 2 ---

        Given a `Quaternion`, this function returns the corresponding Euler
        angles object
        """
        w, x, y, z = self

        wx = w * x
        yz = y * z
        x2 = math.pow(x, 2)
        y2 = math.pow(y, 2)
        roll = math.atan2(2 * (wx + yz), 1 - (2 * (x2 + y2)))

        minus_pi_div_2 = -math.pi / 2
        wy = w * y
        xz = x * z
        two_wy_minus_xz = 2 * (wy - xz)
        sqrt1 = math.sqrt(1 + two_wy_minus_xz)
        sqrt2 = math.sqrt(1 - two_wy_minus_xz)
        pitch = minus_pi_div_2 + (2 * math.atan2(sqrt1, sqrt2))

        wz = w * z
        xy = x * y
        z2 = math.pow(z, 2)
        yaw = math.atan2(2 * (wz + xy), 1 - (2 * (y2 + z2)))

        return EulerAngles(roll, pitch, yaw)

    def to_rotation_matrix(self) -> np.ndarray:
        """
        Converts this `Quaternion` into a rotation matrix and returns it
        """
        w, x, y, z = self

        w2 = w * w
        x2 = x * x
        xy = x * y
        wz = w * z
        xz = x * z
        wy = w * y
        y2 = y * y
        yz = y * z
        wx = w * x
        z2 = z * z

        # fmt: off
        r00 = (2 * (w2 + x2)) - 1
        r01 =  2 * (xy - wz)
        r02 =  2 * (xz + wy)

        r10 =  2 * (xy + wz)
        r11 = (2 * (w2 + y2)) - 1
        r12 =  2 * (yz - wx)

        r20 =  2 * (xz - wy)
        r21 =  2 * (yz + wx)
        r22 = (2 * (w2 + z2)) - 1

        return np.array([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ])
        # fmt: on


class EulerAngles:
    """
    - Roll  (φ) -- rotation around the X-axis
    - Pitch (θ) -- rotation around the Y-axis
    - Yaw   (ψ) -- rotation around the Z-axis
    """

    def __init__(self, roll: float, pitch: float, yaw: float) -> None:
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __repr__(self):
        return f"EulerAngles(roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})"

    def __iter__(self):
        return iter((self.roll, self.pitch, self.yaw))

    def to_quaternion(self) -> Quaternion:
        """
        --- Problem 2 Question 2 Part 1 ---

        Converts the Euler angles object to a `Quaternion`.
        """

        roll, pitch, yaw = self

        # fmt: off
        sin_roll  = math.sin(roll  * 0.5)
        sin_pitch = math.sin(pitch * 0.5)
        sin_yaw   = math.sin(yaw   * 0.5)

        cos_roll  = math.cos(roll  * 0.5)
        cos_pitch = math.cos(pitch * 0.5)
        cos_yaw   = math.cos(yaw   * 0.5)

        return Quaternion(
            w = (cos_roll * cos_pitch * cos_yaw) + (sin_roll * sin_pitch * sin_yaw),
            x = (sin_roll * cos_pitch * cos_yaw) - (cos_roll * sin_pitch * sin_yaw),
            y = (cos_roll * sin_pitch * cos_yaw) + (sin_roll * cos_pitch * sin_yaw),
            z = (cos_roll * cos_pitch * sin_yaw) - (sin_roll * sin_pitch * cos_yaw),
        )
        # fmt: on
