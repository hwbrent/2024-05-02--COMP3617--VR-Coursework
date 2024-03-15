import numpy as np

from model import Model
from vector import Vector

# All are arbitrary; obtained in assignment pdf
# fmt: off
DRAG_COEFFICIENT = 0.5
AIR_DENSITY      = 1.3  # kilograms per cubic metre
REFERENCE_AREA   = 0.2  # square metres
# fmt: on


def get_drag(velocity: float) -> float:
    # Formula is:
    #   D = Cd * (0.5 * r) * V**2 * A
    # Where:
    #   D  = drag
    #   Cd = drag coefficient
    #   r  = air density
    #   V  = velocity
    #   A  = reference area on which the drag coefficient is based
    pass


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
