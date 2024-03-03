import numpy as np

from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector, Quaternion, EulerAngles
from dataset import Dataset

FOCAL_LENGTH = 1
NEAR_CLIP = 0.1

WIDTH = 512
HEIGHT = 512


# Set the camera back slightly from the origin so that the whole headset is
# visible.
camera = Vector(0, 0, -2)


def getOrthographicProjection(x, y, z):
    # Convert vertex from world space to screen space
    # by dropping the z-coordinate (Orthographic projection)
    screenX = int((x + 1.0) * WIDTH / 2.0)
    screenY = int((y + 1.0) * HEIGHT / 2.0)

    return screenX, screenY


def getPerspectiveProjection(vector: Vector) -> None | tuple[int, int]:
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

    x = project(x, WIDTH)
    y = project(y, HEIGHT)

    return x, y


def getVertexNormal(vertIndex, faceNormalsByVertex):
    # Compute vertex normals by averaging the normals of adjacent faces
    normal = Vector(0, 0, 0)
    for adjNormal in faceNormalsByVertex[vertIndex]:
        normal = normal + adjNormal

    return normal / len(faceNormalsByVertex[vertIndex])


def render(model: Model) -> None:
    image = Image(WIDTH, HEIGHT, Color(255, 255, 255, 255))

    # Init z-buffer
    zBuffer = [-float("inf")] * WIDTH * HEIGHT

    # Calculate face normals
    faceNormals = {}
    for face in model.faces:
        p0, p1, p2 = [model.vertices[i] for i in face]
        faceNormal = (p2 - p0).cross(p1 - p0).normalize()

        for i in face:
            if not i in faceNormals:
                faceNormals[i] = []

            faceNormals[i].append(faceNormal)

    # Calculate vertex normals
    vertexNormals = []
    for vertIndex in range(len(model.vertices)):
        vertNorm = getVertexNormal(vertIndex, faceNormals)
        vertexNormals.append(vertNorm)

    # Render the image iterating through faces
    for face in model.faces:
        p0, p1, p2 = [model.vertices[i] for i in face]
        n0, n1, n2 = [vertexNormals[i] for i in face]

        # Define the light direction
        lightDir = Vector(0, 0, -1)

        # Set to true if face should be culled
        cull = False

        # Transform vertices and calculate lighting intensity per vertex
        transformedPoints = []
        for p, n in zip([p0, p1, p2], [n0, n1, n2]):
            intensity = n * lightDir

            # Intensity < 0 means light is shining through the back of the face
            # In this case, don't draw the face at all ("back-face culling")
            if intensity < 0:
                cull = True  # Back face culling is disabled in this version

            # coords = getOrthographicProjection(*p.xyz)
            coords = getPerspectiveProjection(p)
            if coords is None:
                continue

            screenX, screenY = coords

            transformedPoints.append(
                Point(
                    screenX,
                    screenY,
                    p.z,
                    Color(intensity * 255, intensity * 255, intensity * 255, 255),
                )
            )

        # Don't draw triangles whose vertices have been cut off
        if not cull and len(transformedPoints) == 3:
            Triangle(
                transformedPoints[0], transformedPoints[1], transformedPoints[2]
            ).draw_faster(image, zBuffer)

    image.show()


def main() -> None:
    model = Model("data/headset.obj")
    model.normalizeGeometry()

    dataset = Dataset()

    # The data we need in the render loop
    # fmt: off
    data = dataset[[
        "time",
        "gyroscope.X",
        "gyroscope.Y",
        "gyroscope.Z",
    ]]
    # fmt: on

    orientation = Quaternion.identity()
    prev_time = None

    for entry in data.itertuples():
        _, time, x, y, z = entry

        """
        --- Problem 3 Question 1 ---

        Implement a dead reckoning filter (using only the gyroscope-measured
        rotational rate)
        """
        if prev_time is not None:
            time_diff = time - prev_time

            # Convert angular speed to angle by multiplying by time_diff
            angles = EulerAngles(
                x * time_diff,
                y * time_diff,
                z * time_diff,
            )

            orientation *= angles.to_quaternion()
            orientation.normalise()

            # Reflect orientation in model
            model.rotate(matrix=orientation.to_rotation_matrix())

        # Show the model
        render(model)

        prev_time = time


if __name__ == "__main__":
    main()
