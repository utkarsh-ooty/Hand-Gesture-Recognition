import cv2

class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        if not self.is_running():
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")

    def read_frame(self):
        if self.is_running():
            ret, frame = self.cap.read()
            if ret:
                # Flip the frame horizontally for a selfie-view display.
                # This makes interactions more intuitive.
                return cv2.flip(frame, 1)
        return None

    def is_running(self):
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        if self.is_running():
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.stop()
