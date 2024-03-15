import math

from vector import Vector, EulerAngles, Quaternion

# TODO: Problem 3 Question 3
# Try a few different alpha values (e.g., 0.01, 0.1, ...), investigate and
# comment on their effect on drift compensation in your report. Implement
# any other processing of the accelerometer values that you consider important
# / necessary and discuss this in the report.
ALPHA = 0.01


def get_dead_reckoning_filter(gyroscope, time_delta: float) -> Quaternion:
    """
    --- Problem 3 Question 1 ---

    Implements a dead reckoning filter (using only the gyroscope-measured
    rotational rate)
    """
    angles = EulerAngles(*(gyroscope * time_delta))
    return angles.to_quaternion()


def apply_tilt_correction(accelerometer, orientation: Quaternion, gyroscope):
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

    correction = EulerAngles(0, pitch, 0).to_quaternion()
    fused = Quaternion.slerp(orientation, orientation * correction, ALPHA)

    return fused
