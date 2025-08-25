"""
Generate synthetic image sequences for testing optical flow algorithms.
Create motion on x axes
Using cuda to run lucas kanade optical flow
Using cuda pinned memory
"""
import cv2
import numpy as np
import logging
from typing import NamedTuple

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def create_test_sequence(num_frames=300,  motion_speed=3,width=640, height=480):
    # Initial positions
    circle_start = (150, 200)    # Circle starts at left
    rect_start = (120, 100)      # Rectangle starts at left

    # Object properties
    circle_radius = 30
    rect_size = (50, 50)  # width, height
    for frame_idx in range(num_frames):
        # Create black background
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate current positions (moving right)
        circle_x = circle_start[0] + (frame_idx * motion_speed)
        circle_y = circle_start[1]

        rect_x = rect_start[0] + (frame_idx * motion_speed)
        rect_y = rect_start[1]

        # Wrap around if objects go off screen (optional)
        circle_x = circle_x % (width + circle_radius * 2) - circle_radius
        rect_x = rect_x % (width + rect_size[0]) - rect_size[0]

        # Draw white circle
        if -circle_radius <= circle_x <= width + circle_radius:
            cv2.circle(img, (int(circle_x), int(circle_y)), circle_radius, (0, 0, 255), -1)

        # Draw green rectangle
        if -rect_size[0] <= rect_x <= width:
            cv2.rectangle(img, 
                        (int(rect_x), int(rect_y)), 
                        (int(rect_x + rect_size[0]), int(rect_y + rect_size[1])), 
                        (0, 255, 0), -1)

        # Draw green rectangle
        if -rect_size[0] <= rect_x <= width:
            rect_y2 = rect_y + 200 * np.sin(frame_idx / 10)  # Add vertical oscillation
            cv2.rectangle(img, 
                        (int(rect_x), int(rect_y2)), 
                        (int(rect_x + rect_size[0]), int(rect_y2 + rect_size[1])), 
                        (0, 255, 0), -1)


        yield img


class LKResult(NamedTuple):
    good_new: np.ndarray
    good_old: np.ndarray

class LK():
    def __init__(self, w, h):
        self.gpu_detector = cv2.cuda.createGoodFeaturesToTrackDetector(
            cv2.CV_8UC1,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.lk = cv2.cuda.SparsePyrLKOpticalFlow_create(
            winSize=(15, 15), 
            maxLevel=5,
            iters=10, 
            useInitialFlow=False
        )


        # create pin memeory
        PAGE_LOCKED = 1
        pinned_img = cv2.cuda.HostMem(h, w, cv2.CV_8UC3, PAGE_LOCKED)
        self.pin_image = pinned_img.createMatHeader() 
        """ pinned memory for input image """

        self.gpu_img = cv2.cuda.GpuMat() 
        """ hold the current frame as GpuMat """
        self.gpu_prev_gray = cv2.cuda.GpuMat()
        """ hold previous frame as GpuMat """
        self.p0_gpu = cv2.cuda.GpuMat()
        """ hold previous points as GpuMat """

        self.minmun_points_to_redetect = 2

    def process_frame(self, frame: np.ndarray) -> LKResult:
        """
        Process a single frame for optical flow tracking.
        upload the frame to GPU memory from pinned memory.
        run detection if no points to track. (run on gpu)
        run lk calculation. on gpu
        """
        p1_gpu: cv2.cuda.GpuMat
        status_gpu: cv2.cuda.GpuMat
        error_gpu: cv2.cuda.GpuMat

        self.pin_image[:] = frame
        self.gpu_img.upload(self.pin_image)
        gpu_gray = cv2.cuda.cvtColor(self.gpu_img, cv2.COLOR_BGR2GRAY)

        if self.p0_gpu.empty():
            self.p0_gpu = self.gpu_detector.detect(gpu_gray)
            if self.p0_gpu.empty():
                raise RuntimeError("No features detected.")

            log.debug(f"---- Initial points detected: {self.p0_gpu.size()}")
            self.gpu_prev_gray = gpu_gray.clone()

        # Calculate optical flow
        p1_gpu, status_gpu, error_gpu = self.lk.calc(
            self.gpu_prev_gray, 
            gpu_gray,
            self.p0_gpu,
            None,)

        # Download point from gpu for fautere filter and process
        p0 = self.p0_gpu.download().reshape(-1, 2)
        p1 = p1_gpu.download().reshape(-1, 2)
        status = status_gpu.download().reshape(-1).astype(bool)
        error = error_gpu.download().reshape(-1)

        good_new: np.ndarray = p1[status]
        good_old: np.ndarray = p0[status]

        # upload new point back to gpu for the next iteration
        if len(good_new) > self.minmun_points_to_redetect:
            self.p0_gpu.upload(good_new.reshape(1, -1, 2).astype(np.float32))
        else:
            log.warning("No points to track, re-detecting.")
            self.p0_gpu = cv2.cuda.GpuMat()


        self.gpu_prev_gray = gpu_gray.clone()

        return LKResult(good_new=good_new, good_old=good_old)


if __name__ == "__main__":
    COLOR_NEW = (0, 0, 255) # RED
    COLOR_OLD = (255, 0, 0) # BLUE
    new: np.ndarray 
    old: np.ndarray

    lk = LK(640, 480)
    images = create_test_sequence()
    while (frame := next(images, None)) is not None:
        good_new, good_old = lk.process_frame(frame)

        for new, old in zip(good_new, good_old):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            # cv2.arrowedLine(img, (c, d), (a, b), (0, 255, 255), 2, tipLength=0.3)
            cv2.circle(frame, (a, b), 3, COLOR_NEW, -1)
            cv2.circle(frame, (c, d), 3, COLOR_OLD, -1)
        cv2.imshow("LK Optical Flow CUDA", frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break