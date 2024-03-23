""" Module for reading a .obj file into a stored model,
	retrieving vertices, faces, properties of that model.
	Written using only the Python standard library.
"""

from vector import Vector, Quaternion
from physics import get_bounding_sphere
from camera import distance_to, get_lod_swap_range

import numpy as np
from pymeshlab import filter_list, MeshSet

import os

this_dir = os.path.dirname(__file__)
data_dir = os.path.join(this_dir, "data")

# fmt: off
HEADSET_100 = "headset.obj"
HEADSET_50  = "headset_50%.obj"
HEADSET_25  = "headset_25%.obj"
BOOKS       = "books.obj"
COLA        = "cola.obj"
COW         = "cow.obj"
# fmt: on

# The name of the MeshLab operation that reduces the polygon number. See:
# https://pymeshlab.readthedocs.io/en/latest/filter_list.html#meshing_decimation_quadric_edge_collapse
MESHLAB_FILTER_NAME = "meshing_decimation_quadric_edge_collapse"
assert MESHLAB_FILTER_NAME in filter_list()


def reduce_polygons(mesh_set: MeshSet, targetfacenum: int) -> None:
    """
    Given a `MeshSet`, this function reduces (in-place) the polygons of the
    currently-open mesh.
    """

    print(f'Applying "{MESHLAB_FILTER_NAME}" with targetfacenum={targetfacenum}')

    mesh_set.apply_filter(
        MESHLAB_FILTER_NAME,
        targetfacenum=targetfacenum,
    )


def simplify_and_save(targetfacenum: int, output_name: str) -> None:
    """
    Helper function for `simplify_headset`

    Given a number of faces to remove, and the desired name of the output,
    this function opens the fully-detailed model, calls `reduce_polygons`,
    and saves the result
    """
    ms = MeshSet()

    ### Load the fully-detailed model ###
    input_path = os.path.join(data_dir, HEADSET_100)
    print(f'Loading "{input_path}"')
    ms.load_new_mesh(input_path)

    ### Reduce the number of polygons (to 50% or 25%) ###
    reduce_polygons(ms, targetfacenum)

    ### Save the resulting mesh ###
    output_path = os.path.join(data_dir, output_name)
    print(f'Saving to "{output_path}"')
    ms.save_current_mesh(
        output_path,
        save_polygonal=False,  # prevents crash
    )

    ### Clean up ###
    # A .mtl file is saved along with the .obj file. We don't want this. Idk
    # if there's a way to prevent this within the MeshLab paradigm, so we
    # just delete the file manually
    mtl_file_path = output_path + ".mtl"
    os.remove(mtl_file_path)


def simplify_headset():
    """
    --- Problem 6 Question 1 ---

    Using Python bindings for MeshLab, this function simplifies the provided
    VR headset model to two additional versions having 1/2 and 1/4 of the
    polygons.
    """

    ### Get the target face numbers ###
    model = Model(HEADSET_100)
    total_faces = len(model.faces)

    ### Do the simplifying ###
    simplify_and_save(total_faces // 2, HEADSET_50)
    simplify_and_save(total_faces // 4, HEADSET_25)


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

    def transform(self, method: str, record: bool, **kwargs) -> None:
        """
        Given a `method` (i.e. "translate", "rotate" or "scale"), this
        function:
        1. Applies the transform to all the `Vector`s in `self.vertices`
        2. Applies the transform to `self.centre`
        3. Triggers the checking and execution of level-of-detail model
           version switching
        """

        # Anonymous function to dynamically call `method` with the parameters
        # from `kwargs`.
        # e.g. self.vertices[0].translate(dx=1, dy=2, dz=3)
        call_method = lambda obj: getattr(obj, method)(**kwargs)

        # The distance before the transform is carried out
        prev_distance = distance_to(self.centre)

        # Apply the transform
        self.vertices = [call_method(v) for v in self.vertices]
        self.centre = call_method(self.centre)

        # if record:
        #     self.handle_lod_swap(prev_distance)

    def handle_lod_swap(self, prev_distance: float) -> None:
        """
        Checks if the current model needs to be swapped with another in
        in accordance with the level-of-detail streategy, and if so, it
        carries this out
        """

        lod_range = get_lod_swap_range(self.centre, prev_distance)
        if lod_range is None:
            return

        # fmt: off
        self.load(
            HEADSET_100 if lod_range == "closest" else
            HEADSET_50 if lod_range == "middle" else
            HEADSET_25
        )
        # fmt: on

        # Get the cumulative transformations previously applied and reapply
        # them to align the newly-loaded model with the previous model
        self.translate(*self.translation.xyz, record=False)
        self.rotate(self.rotation, record=False)
        self.scale(*self.scaling.xyz, record=False)

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

        self.transform("translate", record, **kwargs)

    def rotate(self, quaternion: Quaternion, record: bool = True) -> None:
        """
        -- Problem 1 Question 3 --

        Rotates this `Model` in-place by the rotation matrix obtained from
        a `quaternion`
        """
        if record:
            self.rotation *= quaternion
            self.rotation.normalise()

        # First, translate the model back to the origin, so that the rotation
        # occurs round the centre of the model, rather than rotating the
        # model around the centre, if that makes sense
        self.translate(*-self.translation.xyz, False)

        # Then, do the actual rotation
        self.transform("rotate", record, matrix=quaternion.to_rotation_matrix())

        # Then translate the model back to where it was before
        self.translate(*self.translation.xyz, False)

    def scale(
        self, sx: float = 1, sy: float = 1, sz: float = 1, record: bool = True
    ) -> None:
        """
        -- Problem 1 Question 3 --

        Scales this `Model` in-place in the x, y and z axes by `sx`, `sy`
        and `sz` respectively.
        """
        kwargs = {"sx": sx, "sy": sy, "sz": sz}

        if record:
            self.scaling = self.scaling.scale(**kwargs)

        self.transform("scale", record, **kwargs)


if __name__ == "__main__":
    simplify_headset()
