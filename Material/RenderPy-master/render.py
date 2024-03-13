import math
from time import time as timer

from image import Image, Color
from model import Model, HEADSET_100, HEADSET_50, HEADSET_25
from shape import Point, Line, Triangle
from vector import Vector, Quaternion, EulerAngles
from dataset import Dataset

FOCAL_LENGTH = 1
NEAR_CLIP = 0.1

# TODO: Problem 3 Question 3
# Try a few different alpha values (e.g., 0.01, 0.1, ...), investigate and
# comment on their effect on drift compensation in your report. Implement
# any other processing of the accelerometer values that you consider important
# / necessary and discuss this in the report.
ALPHA = 0.01

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


def render(model: Model) -> Image:
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

    return image


def main() -> None:
    model = Model(HEADSET_100)
    model.normalizeGeometry()

    dataset = Dataset()
    rows = dataset.df.values
    num_rows = len(rows)

    start_time = timer()

    # Key is the timestamp from the IMU, value is the rendered image
    renders = {}

    orientation = Quaternion.identity()
    prev_time = None

    for i, row in enumerate(rows):
        time = row[0]

        # Progress stats
        renders_done = f"{i+1}/{num_rows}"
        pctg_done = round(((i + 1) / num_rows) * 100, 4)
        imu_time = time
        time_elapsed = round(timer() - start_time, 2)
        print(f"{renders_done} ({pctg_done}%) | {imu_time} | {time_elapsed}")

        if prev_time is None:
            prev_time = time

            image = render(model)
            image.show()
            continue

        time_diff = time - prev_time
        prev_time = time

        gyroscope = row[1:4]
        accelerometer = row[4:7]

        """ Problem 3 Question 1 """
        g_angles = EulerAngles(*(gyroscope * time_diff))
        g_orientation = g_angles.to_quaternion()
        orientation *= g_orientation
        orientation.normalise()

        """ Problem 3 Question 2 """
        # Transform acceleration measurements into the global frame
        a_local = Quaternion(0, *accelerometer)
        a_global = a_local * orientation * orientation.get_conjugate()

        # Calculate the tilt axis
        tilt_axis_local = Quaternion(0, *gyroscope)
        tilt_axis_global = tilt_axis_local * orientation * orientation.get_conjugate()

        # Find the angle φ between the up vector and the vector obtained
        # from the accelerometer
        up = Vector(0, 0, 1)
        acc_vector_global = Vector(a_global.x, a_global.y, a_global.z)
        phi = math.acos(
            up.normalize().dot(
                acc_vector_global.normalize(),
            ),
        )

        # Use the complementary filter to fuse the gyroscope estimation
        # and the accelerometer estimation
        pitch = phi if (acc_vector_global.z < 0) else -phi
        correction = EulerAngles(0, pitch, 0).to_quaternion()
        fused = Quaternion.slerp(orientation, orientation * correction, ALPHA)
        orientation = fused
        orientation.normalise()

        model.rotate(matrix=orientation.to_rotation_matrix())

        image = render(model)
        image.show()

    #     renders[time] = image
    # Image.create_video(renders)

    Image.clean_up()


if __name__ == "__main__":
    main()
