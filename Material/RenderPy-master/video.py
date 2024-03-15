import numpy as np
import cv2

from image import Image


class Video:
    def __init__(self, name: str = "video", fmt: str = "avi"):
        self.name = name
        self.fmt = fmt

        self.frames: list[Image] = []
        self.timestamps: list[float] = []

    def add_frame(self, timestamp: float, image: Image) -> None:
        """
        Adds a given `image` to `self.frames` along with a `timestamp`
        """
        self.frames.append(image)
        self.timestamps.append(timestamp)

    def save(self):
        """
        Creates a video from a collection of `Image` objects and writes it
        to the disk
        """
        if len(self.frames) == 0:
            print("No images provided.")
            return

        avg_fps = 1 / np.mean(np.diff(self.timestamps))

        width = self.frames[0].width
        height = self.frames[0].height

        name_w_extension = self.name + "." + self.fmt

        video_writer = cv2.VideoWriter(
            name_w_extension,
            cv2.VideoWriter_fourcc(*"H264"),
            avg_fps,
            (width, height),
        )

        for image in self.frames:
            video_writer.write(
                cv2.cvtColor(image.to_cv2(), cv2.COLOR_RGBA2BGR),
            )

        video_writer.release()

        print(f'Video saved as "{name_w_extension}"')
