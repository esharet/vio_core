import numpy as np
import matplotlib.pyplot as plt

# --------------------- Camera setup (same style as yours) ---------------------
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CX = CAMERA_WIDTH / 2
CY = CAMERA_HEIGHT / 2

# Intrinsics (your choice: fx=640, fy=480, cx=320, cy=240)
K = np.array([
    [CAMERA_WIDTH, 0,            CX],
    [0,            CAMERA_HEIGHT, CY],
    [0,            0,             1]
], dtype=float)

# Camera pose in WORLD: camera center and pitch (camera tilted down by +pitch)
C_w = np.array([0.0, 0.0, 10.0])   # camera center in world coords
pitch_deg = 31.0                   # rotation about x-axis
c, s = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))

# Rotation CAMERA->WORLD (same structure as your code)
R_wc = np.array([
    [1, 0, 0],
    [0, c,-s],
    [0, s, c]
], dtype=float)

# We need WORLD->CAMERA for projection
R_cw = R_wc.T

# --------------------- World -> pixel function ---------------------
def world_to_pixel(P_w, K, R_cw, C_w):
    """
    P_w : (3,) world point [X,Y,Z] (Z may be 0 for ground).
    K   : (3,3) intrinsics.
    R_cw: (3,3) rotation world->camera.
    C_w : (3,)  camera center in world.
    Returns: (u,v), Zc (depth)
    """
    # world -> camera
    P_c = R_cw @ (P_w - C_w)
    Xc, Yc, Zc = P_c
    # if Zc <= 0:
    #     raise ValueError(f"Point is not in front of the camera (Zc={Zc:.3f}).")
    # camera -> pixel (homogeneous then dehomogenize)
    uvh = K @ P_c
    u, v = uvh[0]/uvh[2], uvh[1]/uvh[2]
    return np.array([u, v]), Zc

# --------------------- Example: a ground point (Z=0) ---------------------
P_ground = np.array([0.0, 5.77350269, 0.0])   # any ground point (x,y,0)
(uv), Zc = world_to_pixel(P_ground, K, R_cw, C_w)

print(f"Ground point Pw={P_ground} projects to pixel (u,v)={uv} with depth Zc={Zc:.3f}")

# --------------------- (Optional) visualize on a 640x480 image ---------------------
plt.figure(figsize=(6.4, 4.8))
plt.gca().add_patch(plt.Rectangle((0,0), CAMERA_WIDTH, CAMERA_HEIGHT, fill=False))
plt.scatter([uv[0]], [uv[1]], s=80)
plt.axvline(CX, ls="--"); plt.axhline(CY, ls="--")
plt.gca().invert_yaxis()
plt.xlim(0, CAMERA_WIDTH); plt.ylim(0, CAMERA_HEIGHT)
plt.title("Projection of world ground point to image")
plt.xlabel("u (px)"); plt.ylabel("v (px)")
plt.grid(True)
plt.show()