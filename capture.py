import cv2

from config import AppConfig


class CameraCapture:
    """Simple webcam wrapper with configured properties."""

    def __init__(self, config: AppConfig):
        self._cap = cv2.VideoCapture(config.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        self._cap.set(cv2.CAP_PROP_FPS, config.camera_fps)

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self._cap.release()
