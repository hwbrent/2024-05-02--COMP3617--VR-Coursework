from time import time as timer

from image import Image, Color
from model import Model, HEADSET_100, HEADSET_50, HEADSET_25
from shape import Point, Triangle
from vector import Vector, Quaternion
from dataset import Dataset
from video import Video
from benchmarking import show_progress
from tracking import get_dead_reckoning_filter, apply_tilt_correction
from camera import getPerspectiveProjection


# Define the light direction
lightDir = Vector(0, 0, -1)


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
    model = Model()
    model.normalizeGeometry()

    dataset = Dataset()

    video = Video()

    ### Prep for the programme loop ###
    start_time = timer()

    orientation = Quaternion.identity()
    prev_time = None

    for row in dataset:
        index, time, gyroscope, accelerometer, magnetometer = row

        image = Image.white()
        zBuffer = image.get_zBuffer()

        if prev_time is not None:
            time_diff = time - prev_time

            orientation *= get_dead_reckoning_filter(gyroscope, time_diff)
            orientation.normalise()

            orientation = apply_tilt_correction(accelerometer, orientation, gyroscope)
            orientation.normalise()

            model.rotate(matrix=orientation.to_rotation_matrix())

        render(model, image, zBuffer)
        image.show()
        video.add_frame(time, image)

        prev_time = time

        show_progress(index, dataset.length, time, start_time)

    video.save()

    Image.clean_up()


if __name__ == "__main__":
    main()
