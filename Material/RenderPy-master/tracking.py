import math

from vector import Vector
from quaternion import Quaternion

# TODO: Problem 3 Question 3
# Try a few different alpha values (e.g., 0.01, 0.1, ...), investigate and
# comment on their effect on drift compensation in your report. Implement
# any other processing of the accelerometer values that you consider important
# / necessary and discuss this in the report.
TILT_ALPHA = 0.01


def apply_dead_reckoning_filter(
    gyroscope, time_delta: float, orientation: Quaternion
) -> Quaternion:
    """
    --- Problem 3 Question 1 ---

    Implements a dead reckoning filter (using only the gyroscope-measured
    rotational rate)
    """
    roll, pitch, yaw = gyroscope * time_delta
    filter = Quaternion.from_euler(roll, pitch, yaw)

    orientation *= filter

    orientation.normalise()
    return orientation


def apply_tilt_correction(
    accelerometer, orientation: Quaternion, gyroscope
) -> Quaternion:
    """
    --- Problem 3 Question 2 ---

    Gets a complementary filter, and uses it to return the `Quaternion`
    representing the fusing of the accelerometer estimation with the global
    orientation quaternion
    """

    # Transform acceleration measurements into the global frame
    a_local = Quaternion(0, *accelerometer)
    a_global = a_local * orientation * orientation.get_conjugate()

    # Calculate the tilt axis
    tilt_axis_local = Quaternion(0, *gyroscope)
    tilt_axis_global = tilt_axis_local * orientation * orientation.get_conjugate()

    # Find the angle φ between the up vector and the vector obtained
    # from the accelerometer
    up = Vector(0, 0, 1)
    acc_vector_global = Vector(a_global.x, a_global.y, a_global.z)
    phi = math.acos(
        up.normalize().dot(
            acc_vector_global.normalize(),
        ),
    )

    # Use the complementary filter to fuse the gyroscope estimation
    # and the accelerometer estimation
    pitch = phi if (acc_vector_global.z < 0) else -phi

    correction = Quaternion.from_euler(0, pitch, 0)
    fused = Quaternion.slerp(orientation, orientation * correction, TILT_ALPHA)

    fused.normalise()
    return fused


def mitigate_yaw_drift(
    orientation: Quaternion, ref_reading, current_reading
) -> Quaternion:
    """
    --- Problem 4 Question 1 ---

    Given the `magnetometer` readings from the headset IMU, this function
    constructs and applies an appropriate complementary filter to the provided
    `orientation` quaternion in order to mitigate any yaw drift
    """
    orientation.normalise()
    return orientation
