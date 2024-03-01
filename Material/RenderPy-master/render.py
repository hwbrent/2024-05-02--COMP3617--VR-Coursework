
from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector

width = 512
height = 512
image = Image(width, height, Color(255, 255, 255, 255))

# Init z-buffer
zBuffer = [-float("inf")] * width * height

# Load the model
model = Model("data/headset.obj")
model.normalizeGeometry()

def getOrthographicProjection(x, y, z, w=1):
    # Convert vertex from world space (in homogeneous coordinates) to screen space
    # by applying an orthographic projection (ignoring z and w for the conversion)
    screenX = int((x / w + 1.0) * width / 2.0)
    screenY = int((y / w + 1.0) * height / 2.0)
    return screenX, screenY

def getVertexNormal(vertIndex, faceNormalsByVertex):
    # Compute vertex normals by averaging the normals of adjacent faces
    normal = Vector(0, 0, 0, 0)  # Use homogeneous coordinates for normals
    for adjNormal in faceNormalsByVertex[vertIndex]:
        normal = normal + adjNormal

    return normal.to_cartesian().normalize()  # Convert to Cartesian and normalize

# Calculate face normals
faceNormals = {}
for face in model.faces:
    p0, p1, p2 = [Vector(*model.vertices[i], 1) for i in face]  # Convert vertices to homogeneous coordinates
    faceNormal = (p2 - p0).cross(p1 - p0).to_cartesian().normalize()  # Calculate and normalize the face normal

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
    p0, p1, p2 = [Vector(*model.vertices[i], 1) for i in face]  # Convert vertices to homogeneous coordinates
    n0, n1, n2 = [vertexNormals[i] for i in face]

    # Define the light direction
    lightDir = Vector(0, 0, -1, 0)  # Use homogeneous coordinates for the light direction

    # Set to true if face should be culled
    cull = False

    # Transform vertices and calculate lighting intensity per vertex
    transformedPoints = []
    for p, n in zip([p0, p1, p2], [n0, n1, n2]):
        intensity = n.to_cartesian() * lightDir.to_cartesian()  # Convert normals back to Cartesian for lighting calculation

        # Intensity < 0 means light is shining through the back of the face
        if intensity < 0:
            cull = True

        screenX, screenY = getOrthographicProjection(p.x, p.y, p.z, p.w)
        transformedPoints.append(
            Point(
                screenX,
                screenY,
                p.z / p.w,  # Convert z to Cartesian for depth
                Color(intensity * 255, intensity * 255, intensity * 255, 255),
            )
        )

    if not cull:
        Triangle(
            transformedPoints[0], transformedPoints[1], transformedPoints[2]
        ).draw_faster(image, zBuffer)

image.saveAsPNG("image.png")
