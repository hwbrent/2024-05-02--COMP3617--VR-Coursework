from vector import EulerAngles, Quaternion


def get_dead_reckoning_filter(gyroscope, time_delta: float) -> Quaternion:
    """
    --- Problem 3 Question 1 ---

    Implements a dead reckoning filter (using only the gyroscope-measured
    rotational rate)
    """
    angles = EulerAngles(*(gyroscope * time_delta))
    return angles.to_quaternion()
