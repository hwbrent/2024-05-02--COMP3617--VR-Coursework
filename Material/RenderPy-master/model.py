""" Module for reading a .obj file into a stored model,
	retrieving vertices, faces, properties of that model.
	Written using only the Python standard library.
"""

from vector import Vector

import numpy as np


class Model(object):
    def __init__(self, file: str):
        self.vertices = np.array([])
        self.faces = []
        self.file = file

        self.parse_file()

    def parse_file(self) -> None:
        """
        This function populates `self.vertices` and `self.faces` according
        to the information parsed in the file at location `self.file`.
        """

        with open(self.file, "r") as f:
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

    def get_face_normals(self) -> dict[int, list[Vector]]:
        """
        (Copied and pasted from the original `render.py`)
        """

        faceNormals = {}
        for face in self.faces:
            p0, p1, p2 = [self.vertices[i] for i in face]
            faceNormal = (p2 - p0).cross(p1 - p0).normalize()

            for i in face:
                if not i in faceNormals:
                    faceNormals[i] = []

                faceNormals[i].append(faceNormal)

        return faceNormals

    def get_vertex_normals(self, faceNormals: dict[int, list[Vector]]):
        """
        (Copied and pasted from the original `render.py`)
        """
        vertexNormals = []
        for vertIndex in range(len(self.vertices)):

            # Compute vertex normals by averaging the normals of adjacent faces
            normal = Vector(0, 0, 0)
            for adjNormal in faceNormals[vertIndex]:
                normal = normal + adjNormal
            vertNorm = normal / len(faceNormals[vertIndex])

            vertexNormals.append(vertNorm)

        return vertexNormals

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
