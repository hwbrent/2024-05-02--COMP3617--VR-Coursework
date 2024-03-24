import math
import numbers

import numpy as np


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

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> "Quaternion":
        """
        --- Problem 2 Question 2 Part 1 ---

        Converts the three Euler angles (in radians) to a `Quaternion`.
        """

        # fmt: off
        sin_roll  = math.sin(roll  * 0.5)
        sin_pitch = math.sin(pitch * 0.5)
        sin_yaw   = math.sin(yaw   * 0.5)

        cos_roll  = math.cos(roll  * 0.5)
        cos_pitch = math.cos(pitch * 0.5)
        cos_yaw   = math.cos(yaw   * 0.5)

        return cls(
            w = (cos_roll * cos_pitch * cos_yaw) + (sin_roll * sin_pitch * sin_yaw),
            x = (sin_roll * cos_pitch * cos_yaw) - (cos_roll * sin_pitch * sin_yaw),
            y = (cos_roll * sin_pitch * cos_yaw) + (sin_roll * cos_pitch * sin_yaw),
            z = (cos_roll * cos_pitch * sin_yaw) - (sin_roll * sin_pitch * cos_yaw),
        )
        # fmt: on

    def to_euler(self) -> tuple[float, float, float]:
        """
        --- Problem 2 Question 2 Part 2 ---

        Given a `Quaternion`, this function returns the corresponding roll,
        pitch and yaw angles in that order
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

        return roll, pitch, yaw

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

    @classmethod
    def slerp(cls, qa: "Quaternion", qb: "Quaternion", t: float = 0.5) -> "Quaternion":
        """
        Returns the Spherical Linear intERPolation between quaternions `qa` and `qb`

        Algorithm is from here:
        - https://euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
        """
        # Calculate angle between them.
        cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z
        # if qa=qb or qa=-qb then theta = 0 and we can return qa
        if abs(cosHalfTheta) >= 1.0:
            return Quaternion(qa.w, qa.x, qa.y, qa.z)
        # Calculate temporary values.
        halfTheta = math.acos(cosHalfTheta)
        sinHalfTheta = math.sqrt(1.0 - cosHalfTheta * cosHalfTheta)
        # if theta = 180 degrees then result is not fully defined
        # we could rotate around any axis normal to qa or qb
        if math.fabs(sinHalfTheta) < 0.001:  # fabs is floating point absolute
            return Quaternion(
                w=(qa.w * 0.5) + (qb.w * 0.5),
                x=(qa.x * 0.5) + (qb.x * 0.5),
                y=(qa.y * 0.5) + (qb.y * 0.5),
                z=(qa.z * 0.5) + (qb.z * 0.5),
            )
        ratioA = math.sin((1 - t) * halfTheta) / sinHalfTheta
        ratioB = math.sin(t * halfTheta) / sinHalfTheta
        # calculate Quaternion.
        return Quaternion(
            w=(qa.w * ratioA) + (qb.w * ratioB),
            x=(qa.x * ratioA) + (qb.x * ratioB),
            y=(qa.y * ratioA) + (qb.y * ratioB),
            z=(qa.z * ratioA) + (qb.z * ratioB),
        )