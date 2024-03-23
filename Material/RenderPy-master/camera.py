import math

from vector import Vector
from image import Image

FOCAL_LENGTH = 1
NEAR_CLIP = 0.1

# Set the camera back slightly from the origin so that the whole headset is
# visible.
camera = Vector(0, 0, -2)

NEAR_CLIP_Z = camera.z + NEAR_CLIP


# Dict relating to the level-of-detail optimisation strategy. The key is the
# "name" of the range, and the value is the min & max values of the range
# of distances used to dictate which model is used
# fmt: off
lod_ranges = {
    "closest":  (NEAR_CLIP_Z, 10),
    "middle":   (10, 25),
    "furthest": (25, math.inf),
}
# fmt: on


def getPerspectiveProjection(vector: Vector, image: Image) -> None | tuple[int, int]:
    """
    --- Problem 1 Question 2 ---

    Apply perspective projection to a given a `Vector` object. If `vector`
    is further than the clip distance, return its `x` and `y` components.
    """

    # Get the position of the given vector relative to the camera
    x, y, z = (vector - camera).xyz

    if z < NEAR_CLIP:
        return None

    project = lambda axis, dimension: int(
        ((FOCAL_LENGTH * axis) / z + 1.0) * dimension / 2.0
    )

    x = project(x, image.width)
    y = project(y, image.height)

    return x, y


def distance_to(vector: Vector) -> float:
    """
    Returns the distance between the camera and a given `vector` as a `float`
    """
    return (camera - vector).norm()


def get_lod_swap_range(centre: Vector, prev_distance: float) -> str | None:
    """
    If the model with centre `centre` should be swapped out for one of a
    different level of detail, this function returns a string indicating
    which model should be chosen. Else, it returns `None`
    """

    # Basically we three "ranges" which correspond to the distances at which
    # the respective model versions are used. We compare the distance before
    # and after whatever transformation was just executed, and if they're not
    # in the same range, we know the LoD swap of model versions is needed.

    new_distance = distance_to(centre)

    # Gets the name of the range which distance `d` falls within
    get_range = lambda d: next(
        key for key, (lower, upper) in lod_ranges.items() if lower <= d < upper
    )

    prev_range = get_range(prev_distance)
    new_range = get_range(new_distance)

    # If the ranges aren't the same, return the name of the range that was
    # just entered
    if new_range != prev_range:
        return new_range

    return None
