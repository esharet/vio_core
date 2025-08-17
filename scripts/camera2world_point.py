# Calc point location on ground from camera pixel
# Take Camera orentation and height (pitch_deg, C_world)
# Take camera point pixel (u, v)

import numpy as np
import matplotlib.pyplot as plt

# region Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CX = CAMERA_WIDTH / 2
CY = CAMERA_HEIGHT / 2


u, v = 320, 240  # Pixel coordinates
# u, v = 320, 248.37843123  # Pixel coordinates
pitch_deg = 0  # Pitch angle in degrees
# pitch_deg = 31  # Pitch angle in degrees
C_world = np.array([0, 0, 10])  # Camera position in world coordinates

# Camera intrinsics
K = np.array([
    [CAMERA_WIDTH, 0, CX],
    [0, CAMERA_HEIGHT, CY],
    [0, 0, 1]
])
# endregion Camera


t_wc = [0, 0, 10]  # Translation from world to camera coordinates (camera location))

# Compute camera extrinsics
R = np.array([
    [1, 0, 0],
    [0, np.cos(np.deg2rad(pitch_deg)), -np.sin(np.deg2rad(pitch_deg))],
    [0, np.sin(np.deg2rad(pitch_deg)), np.cos(np.deg2rad(pitch_deg))]
])

R_wc = R
pix_h = np.array([u, v, 1])  # Homogeneous pixel coordinates
d_c = np.linalg.inv(K) @ pix_h  # Camera coordinates (camera intrinsics)
d_c = d_c / np.linalg.norm(d_c)  # Normalize direction vector
d_w = R_wc @ d_c  # Direction in world coordinates (extransic without translation)
C = t_wc

print("Camera position in world:", C)
print("Direction in world coordinates:", d_w)
print("Direction in camera coordinates:", d_c)

t_ray = -C[2]/  d_w[2] # ration between camera z and ray z
P_hit = C + t_ray * d_w  # Intersection point with z=0 plane

print("World point at z=0:", P_hit)

# region Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*C, color='blue', s=50, label='Camera center')
ax.plot([C[0], P_hit[0]], 
        [C[1], P_hit[1]],
        [C[2], P_hit[2]], color='orange', label='Ray to z=0 plane')

ax.scatter(*P_hit, color='green', s=50, label='ground Point') 

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Ray Intersection with Z=0 Plane')
ax.legend()
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([0, 10])
plt.show()
# endregion Visualization

#30: World point at z=0: [0.         5.77350269 0.        ]
#31: World point at z=0: [0.         6.00860619 0.        ]