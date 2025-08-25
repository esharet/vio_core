from typing import List
import numpy as np
import time
# import matplotlib.pyplot as plt
import logging
from of_vio.utils import world2img_generator
# import  of_vio.vio_debugger
import math


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FX = FY = 800
CX = CAMERA_WIDTH / 2
CY = CAMERA_HEIGHT / 2

logging.basicConfig(level=logging.INFO)

def fov_to_focal_length(fov_deg: float, sensor_size: float) -> float:
    fov_rad = math.radians(fov_deg)  # convert degrees to radians
    focal_length = sensor_size / (2 * math.tan(fov_rad / 2))
    return focal_length

class VIO:
    def __init__(self):
        # f = fov_to_focal_length(50, 640)
        # logging.info(f"Focal length: {f}")
        self.K = np.array([
            [FX, 0, CX],
            [0, FY, CY],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(self.K)
        
         
        self.cache = None

    def init_vio(self, points: np.array):
        """
        save current points in cache for the next estimate
        get point from O.F shape (:, 1, 2)
        """
        self.cache = points

    def calc_velocity(self, points: np.array,
                      altitude: float, 
                      orientation: np.array):
        """
        save current points in cache for the next estimate
        get point from O.F shape (:, 1, 2)
        """
        pitch_deg = orientation[1]
        self.C_world = np.array([0, 0, altitude])
        # TODO: Add full orination support
        R = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(pitch_deg)), -np.sin(np.deg2rad(pitch_deg))],
            [0, np.sin(np.deg2rad(pitch_deg)), np.cos(np.deg2rad(pitch_deg))]

        ])
        
        R_wc = R
        # Step 1: homogeneous pixels
        points = points.reshape(-1, 2)
        points = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])  # (N, 3)

        # Step 2: backproject into camera frame
        d_c = self.K_inv @ points.T  # (3, N)

        # (Optional) Normalize each ray direction
        d_c /= np.linalg.norm(d_c, axis=0, keepdims=True)

        # Step 3: rotate to world frame
        d_w = R_wc @ d_c  # (3, N)

        # Step 4: ray-plane intersection (plane z=0)
        t_ray = -self.C_world[2] / d_w[2, :]  # (N,)

        # Step 5: compute intersection
        P_hit = self.C_world.reshape(3, 1) + d_w * t_ray  # (3, N)
        P_hit = P_hit.T  # (N, 3)

        if self.cache is None:
            self.cache = P_hit
            logging.info("First points, save in cache and exit")
            return
            

        vx, vy = self._calc_speed_xy(P_hit-self.cache)
        self.cache = P_hit
        return vx, vy
    
    def _calc_speed_xy(self, deltas: np.array):
        """
        points shape (:, 1, 3)
        """

        x = deltas[:, 0]  # X coordinates
        y = deltas[:, 1]  # Y coordinate

        # --- 1. Histogram ---
        #  why
        # counts_x, bin_edges_x = np.histogram(x, bins=50)
        # counts_y, bin_edges_y = np.histogram(y, bins=50)

        

        # --- 2. Compute mean and std ---
        mean_x, std_x = np.mean(x), np.std(x)
        mean_y, std_y = np.mean(y), np.std(y)

        return mean_x, mean_y

def gen_sector_center_points():
    width = 640   # X
    height = 480  # Y

    cols = 3
    rows = 3

    sector_w = width / cols
    sector_h = height / rows

    centers = []

    for i in range(rows):       # row index
        for j in range(cols):   # column index
            # Center coordinates
            cx = (j + 0.5) * sector_w
            cy = (i + 0.5) * sector_h
            centers.append((cx, cy))

    # Convert to numpy array if needed
    centers = np.array(centers).reshape(-1, 1, 2)
    return centers

def gen_multiple_middle_points(x, y):
    POINTS = 500
    points = np.array(
        [[x, y]]
    )
    points = np.tile(points, (POINTS, 1, 1))

    return points


if __name__ == "__main__":
    vio = VIO()
    pitch_deg = 0
    ALTITUDE = 10
    C_world_orientation = np.array([
        0, pitch_deg, 0])

    vx, vy = 0, 0
    
    # d1 = gen_multiple_middle_points(320, 242)
    # d2 = gen_multiple_middle_points(320, 240)

    # d1 = gen_sector_center_points()
    # d2 = gen_sector_center_points()

    d1 = world2img_generator.world_to_pixel(0, 0, ALTITUDE, C_world_orientation)
    d2 = world2img_generator.world_to_pixel(0, 0.5, ALTITUDE, C_world_orientation)
    d3 = world2img_generator.world_to_pixel(0, 1.0, ALTITUDE, C_world_orientation)
    d4 = world2img_generator.world_to_pixel(0, 1.5, ALTITUDE, C_world_orientation)
    d5 = world2img_generator.world_to_pixel(0, 2.0, ALTITUDE, C_world_orientation)
    
    for points_snap in [d1, d2, d3, d4, d5]:
        start = time.perf_counter()
        result = vio.calc_velocity(points_snap, ALTITUDE, orientation=C_world_orientation)
        if result is not None:
            vx, vy = result
        end = time.perf_counter()
        logging.info(f"vx: {vx}, vy: {vy}")
        logging.info(f"Runtime: {end - start} seconds")    # logging.info(f"Runtime: {end - start} seconds")
    