import math
from time import time as timer

from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector, Quaternion, EulerAngles
from dataset import Dataset
from video import Video
from benchmarking import show_progress

FOCAL_LENGTH = 1
NEAR_CLIP = 0.1

# TODO: Problem 3 Question 3
# Try a few different alpha values (e.g., 0.01, 0.1, ...), investigate and
# comment on their effect on drift compensation in your report. Implement
# any other processing of the accelerometer values that you consider important
# / necessary and discuss this in the report.
ALPHA = 0.01


# Set the camera back slightly from the origin so that the whole headset is
# visible.
camera = Vector(0, 0, -2)

# Define the light direction
lightDir = Vector(0, 0, -1)


def getPerspectiveProjection(vector: Vector, image: Image) -> None | tuple[int, int]:
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

    x = project(x, image.width)
    y = project(y, image.height)

    return x, y


def render(model: Model, image: Image, zBuffer: list[float]) -> None:
    # Calculate face normals
    faceNormals = model.get_face_normals()
    vertexNormals = model.get_vertex_normals(faceNormals)

    # Render the image iterating through faces
    for face in model.faces:
        vertices = [model.vertices[i] for i in face]
        normals = [vertexNormals[i] for i in face]

        # Set to true if face should be culled
        cull = False

        # Transform vertices and calculate lighting intensity per vertex
        transformed = []
        for vertex, normal in zip(vertices, normals):
            intensity = normal * lightDir
            screen_xy = getPerspectiveProjection(vertex, image)

            # Intensity < 0 means light is shining through the back of the face
            # In this case, don't draw the face at all ("back-face culling")
            if intensity < 0 or screen_xy is None:
                cull = True
                break

            rgb = intensity * 255  # same value for each. Makes colours grayscale
            color = Color(rgb, rgb, rgb, 255)
            point = Point(*screen_xy, vertex.z, color)
            transformed.append(point)

        if cull:
            continue

        triangle = Triangle(*transformed)
        triangle.draw_faster(image, zBuffer)


def main() -> None:
    ### Initialise "globals" ###
    model = Model("data/headset.obj")
    model.normalizeGeometry()

    dataset = Dataset()
    rows = dataset.df.values
    num_rows = len(rows)

    video = Video()

    ### Prep for the programme loop ###
    start_time = timer()

    orientation = Quaternion.identity()
    prev_time = None

    for i, row in enumerate(rows):
        time = row[0]

        show_progress(i, num_rows, time, start_time)

        image = Image.white()
        zBuffer = image.get_zBuffer()

        if prev_time is None:
            prev_time = time

            render(model, image, zBuffer)
            image.show()
            video.add_frame(time, image)
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

        render(model, image, zBuffer)
        image.show()

        video.add_frame(time, image)

    video.save()
    Image.clean_up()


if __name__ == "__main__":
    main()
