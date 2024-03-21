""" Module for reading a .obj file into a stored model,
	retrieving vertices, faces, properties of that model.
	Written using only the Python standard library.
"""

from vector import Vector, Quaternion
from physics import get_bounding_sphere
from camera import distance_to, get_lod_swap_range

import numpy as np

import os

this_dir = os.path.dirname(__file__)
data_dir = os.path.join(this_dir, "data")

# fmt: off
HEADSET_100 = "headset_100%.obj"
HEADSET_50  = "headset_50%.obj"
HEADSET_25  = "headset_25%.obj"
BOOKS       = "books.obj"
COLA        = "cola.obj"
COW         = "cow.obj"
# fmt: on


class Model(object):
    def __init__(self, file: str = HEADSET_100):
        self.file = file

        # Define what the attributes are
        self.vertices = np.array([])
        self.faces = []
        self.centre = Vector(0, 0, 0)
        self.radius = 0.0

        # The cumulative transformations
        self.translation = Vector(0, 0, 0)
        self.rotation = Quaternion.identity()
        self.scaling = Vector(1, 1, 1)

        # Set the attributes
        self.load(file)

    def load(self, file: str) -> None:
        self.file = file
        self.vertices, self.faces = self.parse_file()
        self.centre, self.radius = get_bounding_sphere(self.vertices)

    def parse_file(self) -> None:
        """
        This function populates `self.vertices` and `self.faces` according
        to the information parsed in the file at location `self.file`.
        """
        vertices = []
        faces = []

        file_path = os.path.join(data_dir, self.file)
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                segments = line.split()
                if not segments:
                    continue

                # Vertices
                if segments[0] == "v":
                    vertex = Vector(*[float(i) for i in segments[1:4]])
                    vertices.append(vertex)

                # Faces
                elif segments[0] == "f":
                    # Support models that have faces with more than 3 points
                    # Parse the face as a triangle fan
                    for i in range(2, len(segments) - 1):
                        corner1 = int(segments[1].split("/")[0]) - 1
                        corner2 = int(segments[i].split("/")[0]) - 1
                        corner3 = int(segments[i + 1].split("/")[0]) - 1
                        faces.append([corner1, corner2, corner3])

        vertices = np.array(vertices)

        return vertices, faces

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

        for vertex in self.vertices:
            vertex.x = vertex.x / maxCoords[0]
            vertex.y = vertex.y / maxCoords[1]
            vertex.z = vertex.z / maxCoords[2]

    def transform(self, method, **kwargs) -> None:
        call_method = lambda obj: getattr(obj, method)(**kwargs)

        prev_distance = distance_to(self.centre)

        # Apply transform
        self.vertices = [call_method(v) for v in self.vertices]
        self.centre = call_method(self.centre)

        self.handle_lod_swap(prev_distance)

    def handle_lod_swap(self, prev_distance: float) -> None:
        """
        Checks if the current model needs to be swapped with another in
        in accordance with the level-of-detail streategy, and if so, it
        carries this out
        """

        lod_range = get_lod_swap_range(self.centre, prev_distance)
        if lod_range is None:
            return

        print(lod_range)

        return

        # fmt: off
        self.load(
            HEADSET_100 if lod_range == "closest" else
            HEADSET_50 if lod_range == "middle" else
            HEADSET_25
        )
        # fmt: on

    def translate(
        self, dx: float = 0, dy: float = 0, dz: float = 0, record: bool = True
    ) -> None:
        """
        -- Problem 1 Question 3 --

        Translates this `Model` in-place in the x, y and z axes by `dx`, `dy`, `dz`
        respectively.
        """

        kwargs = {"dx": dx, "dy": dy, "dz": dz}

        if record:
            # Record what the translation was
            self.translation = self.translation.translate(**kwargs)

        self.transform("translate", **kwargs)

    def rotate(self, quaternion: Quaternion) -> None:
        """
        -- Problem 1 Question 3 --

        Rotates this `Model` in-place by the rotation matrix obtained from
        a `quaternion`
        """
        # Record the rotation
        self.rotation *= quaternion
        self.rotation.normalise()
        print(*self.rotation)

        # First, translate the model back to the origin, so that the rotation
        # occurs round the centre of the model, rather than rotating the
        # model around the centre, if that makes sense
        self.translate(*-self.translation.xyz, False)

        # Then, do the actual rotation
        self.transform("rotate", matrix=quaternion.to_rotation_matrix())

        # Then translate the model back to where it was before
        self.translate(*self.translation.xyz, False)

    def scale(self, sx: float = 1, sy: float = 1, sz: float = 1) -> None:
        """
        -- Problem 1 Question 3 --

        Scales this `Model` in-place in the x, y and z axes by `sx`, `sy`
        and `sz` respectively.
        """
        kwargs = {"sx": sx, "sy": sy, "sz": sz}

        self.scaling = self.scaling.scale(**kwargs)
        self.transform("scale", **kwargs)
