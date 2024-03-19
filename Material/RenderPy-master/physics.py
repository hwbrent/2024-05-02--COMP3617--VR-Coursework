import numpy as np

from vector import Vector


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
