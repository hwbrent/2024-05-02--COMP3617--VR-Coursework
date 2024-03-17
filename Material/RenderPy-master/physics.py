import numpy as np

from vector import Vector

# fmt: off
G = 9.80665  # See https://en.wikipedia.org/wiki/Standard_gravity

# Arbitrary
DRAG_COEFFICIENT = 0.5
AIR_DENSITY      = 1.3  # kilograms per cubic metre
REFERENCE_AREA   = 0.2  # square metres
# fmt: on


def get_drag(velocity: float) -> float:
    """
    Given the value of a `Model`'s `velocity` attribute, this function
    calculates the drag force acting on the `Model`.

    See
    https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/falling.html
    for the formula
    """
    return DRAG_COEFFICIENT * 0.5 * AIR_DENSITY * (velocity**2) * REFERENCE_AREA


def get_acceleration(mass: float, velocity) -> float:
    """
    This function uses the provided `Model`'s `MASS` and `velocity` properties
    to calculate and return its acceleration

    See
    https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/falling.html
    for the formula
    """

    D = get_drag(velocity)
    W = mass * G

    return (W - D) / mass


def get_bounding_sphere(vertices: list["Vector"]) -> tuple[Vector, float]:
    """
    --- Problem 5 Question 2 ---

    Gets the centre and radius of `model`'s bounding sphere
    """
    # The average of all the xyz values of the vertices
    centre = np.mean([v.xyz for v in vertices], axis=0)
    v_centre = Vector(*centre)

    # The radius is the distance of the furthest-away vertex from
    # v_centre
    radius = max((v_centre - v).norm() for v in vertices)

    return v_centre, radius


def check_collision(
    sphere1: tuple[Vector, float], sphere2: tuple[Vector, float]
) -> bool:
    """
    --- Problem 5 Question 2 ---

    Given two `Model` objects, this function returns a `bool` indicating
    whether they are colliding.
    """
    centre1, radius1 = sphere1
    centre2, radius2 = sphere2

    distance = (centre1 - centre2).norm()
    radius_sum = radius1 + radius2
    is_colliding = distance <= radius_sum

    return is_colliding
