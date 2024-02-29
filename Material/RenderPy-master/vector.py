import math
import numbers
import os

import pandas as pd


class Vector(object):
    """A vector with useful vector / matrix operations."""

    def __init__(self, *args):
        if len(args) == 0:
            self.components = (0, 0)
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

    def norm(self):
        """Return the norm (magnitude) of this vector."""
        squaredComponents = sum(math.pow(comp, 2) for comp in self)
        return math.sqrt(squaredComponents)

    def normalize(self):
        """Return a normalized unit vector from this vector."""
        magnitude = self.norm()
        return Vector(*[comp / magnitude for comp in self])

    def dot(self, other):
        """Return the dot product of this and another vector."""
        return sum(a * b for a, b in zip(self, other))

    def cross(self, other):
        """Return the cross product of this and another vector."""
        assert len(self) == len(other), "Vectors must be the same size."
        assert len(self) == 3, "Cross product only implemented for 3D vectors."
        return Vector(
            (self.y * other.z - self.z * other.y),
            (self.z * other.x - self.x * other.z),
            (self.x * other.y - self.y * other.x),
        )

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


def import_dataset() -> pd.core.frame.DataFrame:
    """
    --- Problem 2 Question 1 ---

    This function loads the `IMUData` CSV file and returns it as a pandas
    `DataFrame`
    """
    ds_name = "IMUData.csv"
    ds_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, ds_name)
    )

    ds = pd.read_csv(ds_path)

    # For some reason, there is rogue whitespace on one or both sides of
    # each column name. Let's fix that:
    ds.columns = ds.columns.map(lambda col_name: col_name.strip())

    return ds


def convert_rotational_rate(dataset: pd.core.frame.DataFrame) -> None:
    """
    --- Problem 2 Question 1 ---

    Given the dataset obtained by `import_dataset`, this function converts
    the rotational rate (tri-axial velocity in deg/s) to radians/sec
    in-place
    """

    axes = ("X", "Y", "Z")
    for axis in axes:
        # Get the name of the column
        col = "gyroscope." + axis  # e.g. gyroscope.X

        # Convert each value in the column from degrees to radians
        dataset[col] = dataset[col].apply(lambda value: math.radians(value))


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def get_conjugate(self):
        """
        --- Problem 2 Question 2 Part 3 ---

        Returns the conjugate (i.e. inverse rotation) of this `Quaternion`
        """
        w, x, y, z = self
        return Quaternion(w, -x, -y, -z)

    def __repr__(self):
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    def __iter__(self):
        return iter((self.w, self.x, self.y, self.z))


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


def euler_to_quaternion(euler_angles: EulerAngles) -> Quaternion:
    """
    --- Problem 2 Question 2 Part 1 ---

    Converts the Euler angles object to a `Quaternion`.
    """

    roll, pitch, yaw = euler_angles

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


def quaternion_to_euler(quaternion: Quaternion) -> EulerAngles:
    """
    --- Problem 2 Question 2 Part 2 ---

    Given a `Quaternion`, this function returns the corresponding Euler
    angles object
    """
    w, x, y, z = quaternion

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
