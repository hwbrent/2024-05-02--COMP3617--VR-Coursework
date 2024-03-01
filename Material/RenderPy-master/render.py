from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector

FOCAL_LENGTH = 1

width = 512
height = 512
image = Image(width, height, Color(255, 255, 255, 255))

# Init z-buffer
zBuffer = [-float("inf")] * width * height

# Load the model
model = Model("data/headset.obj")
model.normalizeGeometry()


def getPerspectiveProjection(x, y, z):
    return (
        int(((FOCAL_LENGTH * x) / z + 1.0) * width / 2.0),
        int(((FOCAL_LENGTH * y) / z + 1.0) * height / 2.0),
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

        screenX, screenY = getPerspectiveProjection(*p.xyz)
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

image.saveAsPNG("image.png")
