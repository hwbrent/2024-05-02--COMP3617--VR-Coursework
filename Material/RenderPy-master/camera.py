from vector import Vector
from image import Image

FOCAL_LENGTH = 1
NEAR_CLIP = 0.1

# Set the camera back slightly from the origin so that the whole headset is
# visible.
camera = Vector(0, 0, -2)

NEAR_CLIP_Z = camera.z + NEAR_CLIP


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


def lod_swap_needed(centre: Vector, prev_distance: float) -> bool:
    """
    Returns a bool indicating whether the model with centre `centre` should
    be swapped out for one of a different level of detail
    """

    # Basically we three "ranges" which correspond to the distances at which
    # the respective model versions are used. We compare the distance before
    # and after whatever transformation was just executed, and if they're not
    # in the same range, we know the LoD swap of model versions is needed.

    new_distance = distance_to(centre)

    # These are the "ranges". Couldn't use the `range` object because the
    # distance is a float
    closest = lambda d: NEAR_CLIP_Z <= d < 4
    middle = lambda d: 4 <= d < 10
    furthest = lambda d: 10 <= d < 20

    # fmt: off
    return (
           (closest(prev_distance)  != closest(new_distance))
        or (middle(prev_distance)   != middle(new_distance))
        or (furthest(prev_distance) != furthest(new_distance))
    )
    # fmt: on
