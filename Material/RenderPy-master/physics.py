import numpy as np

from model import Model
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


def get_acceleration(model: Model) -> float:
    """
    This function uses the provided `Model`'s `MASS` and `velocity` properties
    to calculate and return its acceleration

    See
    https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/falling.html
    for the formula
    """
    m = model.MASS

    D = get_drag(model.velocity)
    W = m * G

    return (W - D) / m


def get_bounding_sphere(model: Model) -> tuple[Vector, float]:
    """
    --- Problem 5 Question 2 ---

    Gets the centre and radius of `model`'s bounding sphere
    """
    # The average of all the xyz values of the vertices
    centre = np.mean([v.xyz for v in model.vertices], axis=0)
    v_centre = Vector(*centre)

    # The radius is the distance of the furthest-away vertex from
    # v_centre
    radius = max((v_centre - v).norm() for v in model.vertices)

    return v_centre, radius


def check_collision(model1: "Model", model2: "Model") -> bool:
    """
    --- Problem 5 Question 2 ---

    Given two `Model` objects, this function returns a `bool` indicating
    whether they are colliding.
    """
    distance = (model1.sphere_centre - model2.sphere_centre).norm()
    radius_sum = model1.sphere_radius + model2.sphere_radius
    return distance <= radius_sum
