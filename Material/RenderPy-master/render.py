import math

import numpy as np

from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector


FOV = math.radians(90)  # Standard
FOCAL_LENGTH = 1 / math.tan(FOV / 2)


width = 512
height = 512
image = Image(width, height, Color(255, 255, 255, 255))

# Init z-buffer
zBuffer = [-float("inf")] * width * height

# Load the model
model = Model("data/headset.obj")
model.normalizeGeometry()


def getOrthographicProjection(x, y, z):
    # Convert vertex from world space to screen space
    # by dropping the z-coordinate (Orthographic projection)
    screenX = int((x + 1.0) * width / 2.0)
    screenY = int((y + 1.0) * height / 2.0)

    return screenX, screenY


def getPerspectiveProjectionMatrix() -> list[list[float]]:
    """
    --- Problem 1 Question 2 ---

    This function gets the matrix used for perspective projection
    """

    aspect_ratio = image.width / image.height
    near = 0.1
    far = 1_000

    matrix = [
        [FOCAL_LENGTH / aspect_ratio, 0, 0, 0],
        [0, FOCAL_LENGTH, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ]

    return np.array(matrix)


def doPerspectiveProjection(vector: Vector) -> tuple[int, int]:
    """
    --- Problem 1 Question 2 ---

    Apply perspective projection to a given a `Vector` object, and return
    the `x` and `y` components.
    """

    # Make sure that the w component is present (by default it should be, but
    # I found that sometimes it wasn't for some reason)
    components = (
        (*vector.components, 1) if len(vector.components) != 4 else vector.components
    )

    projected = np.dot(
        np.array(components),
        getPerspectiveProjectionMatrix(),
    )

    # Perform perspective divide (avoiding division by zero)
    if projected[3] != 0:
        projected /= projected[3]

    x, y, _ = projected[:3]

    return int(x), int(y)


def getScreenCoords(x: int, y: int) -> tuple[int, int]:
    return (
        int((x + 1.0) * width / 2.0),
        int((y + 1.0) * height / 2.0),
    )


def getVertexNormal(vertIndex, faceNormalsByVertex):
    # Compute vertex normals by averaging the normals of adjacent faces
    normal = Vector(0, 0, 0)
    for adjNormal in faceNormalsByVertex[vertIndex]:
        normal = normal + adjNormal

    return normal / len(faceNormalsByVertex[vertIndex])


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

        # Get the perspective-projected coordinates, then get the screen
        # coordinates from those
        proj_X, proj_Y = doPerspectiveProjection(p)
        screenX, screenY = getScreenCoords(proj_X, proj_Y)

        transformedPoints.append(
            Point(
                screenX,
                screenY,
                p.z,
                Color(intensity * 255, intensity * 255, intensity * 255, 255),
            )
        )

    if not cull:
        Triangle(
            transformedPoints[0], transformedPoints[1], transformedPoints[2]
        ).draw_faster(image, zBuffer)

image.show()
