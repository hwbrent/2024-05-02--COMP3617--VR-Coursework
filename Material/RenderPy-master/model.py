""" Module for reading a .obj file into a stored model,
	retrieving vertices, faces, properties of that model.
	Written using only the Python standard library.
"""

from vector import Vector

import numpy as np


class Model(object):
    def __init__(self, file):
        self.vertices = np.array([])
        self.faces = []
        self.scale = [0, 0, 0]
        self.rot = [0, 0, 0]
        self.trans = [0, 0, 0]

        # Read in the file
        f = open(file, "r")
        for line in f:
            if line.startswith("#"):
                continue
            segments = line.split()
            if not segments:
                continue

            # Vertices
            if segments[0] == "v":
                vertex = Vector(*[float(i) for i in segments[1:4]])
                self.vertices = np.append(self.vertices, vertex)

            # Faces
            elif segments[0] == "f":
                # Support models that have faces with more than 3 points
                # Parse the face as a triangle fan
                for i in range(2, len(segments) - 1):
                    corner1 = int(segments[1].split("/")[0]) - 1
                    corner2 = int(segments[i].split("/")[0]) - 1
                    corner3 = int(segments[i + 1].split("/")[0]) - 1
                    self.faces.append([corner1, corner2, corner3])

    def normalizeGeometry(self):
        maxCoords = [0, 0, 0]

        for vertex in self.vertices:
            maxCoords[0] = max(abs(vertex.x), maxCoords[0])
            maxCoords[1] = max(abs(vertex.y), maxCoords[1])
            maxCoords[2] = max(abs(vertex.z), maxCoords[2])

        self.vertices *= 1 / max(maxCoords)

    def translate(self, dx: int, dy: int, dz: int) -> None:
        """
        -- Problem 1 Question 3 --

        Translates this `Model` in-place in the x, y and z axes by `dx`, `dy`, `dz`
        respectively.
        """

        self.vertices = [v.translate(dx, dy, dz) for v in self.vertices]

    def rotate(self, **kwargs) -> None:
        """
        -- Problem 1 Question 3 --

        Rotates this `Model` in-place either:
        1) around `axis` by `angle`
        2) by a `matrix`
        """
        keys = sorted(kwargs.keys())

        # The two use-cases of this function
        case1 = keys == ["angle", "axis"]
        case2 = keys == ["matrix"]
        assert case1 or case2

        self.vertices = [v.rotate(**kwargs) for v in self.vertices]

    def scale(self, sx, sy, sz) -> None:
        """
        -- Problem 1 Question 3 --

        Scales this `Model` in-place in the x, y and z axes by `sx`, `sy`
        and `sz` respectively.
        """

        self.vertices = [v.scale(sx, sy, sz) for v in self.vertices]
