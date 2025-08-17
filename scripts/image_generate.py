import numpy as np
import cv2

def make_dot_pattern(h=480, w=640, num=2000, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.uint8)
    xs = rng.integers(0, w, size=num)
    ys = rng.integers(0, h, size=num)
    img[ys, xs] = 255
    return cv2.GaussianBlur(img, (0, 0), 0.8)  # soften a bit

def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def generate_sequence(
    N=60,                   # number of frames
    w=640, h=480,           # image size
    fx=600, fy=600,         # intrinsics
    cx=None, cy=None,
    altitude=10.0,          # meters (distance to ground plane)
    pitch_deg=20.0,         # camera pitched down (+) about X (right) axis
    vx=1.0, vy=0.0, vz=0.0, # m/s of camera in world (x forward, y right, z up)
    yaw_rate_deg=0.0,       # deg/s yaw about z
    fps=30,                 # frames per second
    base_img=None           # background texture (None -> random dots)
):
    """
    Returns list of frames (uint8), and dict of info.
    World frame: x forward, y right, z up. Ground plane is z=0. Camera starts at (0,0,altitude).
    Camera optical axis initially points along world -z, then pitched down by +pitch about camera X.
    """
    cx = w/2 if cx is None else cx
    cy = h/2 if cy is None else cy

    # Intrinsic matrix
    K  = np.array([[fx, 0,  cx],
                   [0,  fy, cy],
                   [0,  0,   1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)

    # Base texture (reference view content in "world"—we’ll just warp this each frame)
    if base_img is None:
        base_img = make_dot_pattern(h=h, w=w, num=int(w*h*0.01), seed=123)

    # Ground plane parameters
    n = np.array([[0.0], [0.0], [1.0]])  # world +z is up
    d = altitude

    # Camera pose at frame k (world->camera): we will construct relative motion from frame 0
    # Start pose: camera at (0,0,altitude), looking down (optical axis ~ -z) with pitch about X
    # We'll build camera orientation as: R_cam0 = Rx(pitch) * R_down
    # R_down aligns camera z with -world z: that is a 180° rotation about X or Y; simpler: use identity and interpret
    # pitch as tilt down (positive) about camera X. For planar homography we only need relative motion between frames.

    # Per-frame small motion (world): delta R and delta t
    dt = 1.0 / fps
    yaw_rate = np.deg2rad(yaw_rate_deg)
    pitch0 = np.deg2rad(pitch_deg)

    # Camera orientation wrt world at frame k:
    # Start with camera looking down: rotate around X by +pitch0 (tilt down) and then any yaw over time
    # We define a base orientation that looks down the -z axis with some pitch:
    # Use R_base = Rx(pitch0)
    R_base = Rx(pitch0)

    # Accumulate motion over frames in world frame:
    frames = []
    infos = {"K": K, "altitude": altitude, "pitch_deg": pitch_deg, "fps": fps,
             "vel_world_mps": (vx, vy, vz), "yaw_rate_deg_s": yaw_rate_deg}

    # Camera center in world at frame k
    Cw = np.array([0.0, 0.0, altitude], dtype=np.float64)
    yaw = 0.0

    # We will compute homography that maps the base image (frame 0) to each frame k.
    # For a plane z=0, mapping from frame k to the plane via H_k uses relative (R_k, t_k) from world to camera.
    # The standard plane-induced homography is: H = K (R - t n^T / d) K^{-1}
    # Here R, t are camera-from-world for the given frame.

    for k in range(N):
        # Update pose (world)
        yaw = k * yaw_rate * dt
        # Integrate translation in world frame
        Cw_k = Cw + np.array([vx*dt*k, vy*dt*k, vz*dt*k])

        # Camera rotation wrt world (world->camera): yaw about world z, then base pitch
        R_wc = Rz(yaw).T @ R_base   # world->camera (apply yaw in world, then base)

        # Camera translation wrt world in camera coords: t = -R * C
        t_wc = (-R_wc @ Cw_k.reshape(3,1))  # (3,1)

        # Homography from world plane (z=0) to image at frame k
        H = K @ (R_wc - (t_wc @ n.T) / d) @ Kinv

        warped = cv2.warpPerspective(base_img, H, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
        frames.append(warped)

    return frames, infos


# Suppose 'frames' is your list of images from the synthetic generator
# frames, _ = generate_sequence(...)

# Parameters for Shi-Tomasi corner detection (to pick good features to track)
feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))






if __name__ == "__main__":
    frames, info = generate_sequence(
        N=60, w=640, h=480,
        fx=600, fy=600,
        altitude=10.0,
        pitch_deg=20.0,
        vx=1.0, vy=0.0, vz=0.0,
        yaw_rate_deg=0.0,
        fps=30
    )

    feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners
    old_frame = frames[0]
    old_gray = old_frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing (optional)
    mask = np.zeros_like(old_frame)

    # Color for drawing tracks
    color = (0, 255, 0)  # green in BGR
    for i in range(1, len(frames)):
        frame_gray = frames[i]

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is None:
            break
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw tracks
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
            frame_bgr = cv2.circle(frame_bgr, (int(a), int(b)), 3, color, -1)

        
        # img = cv2.add(frame_bgr, mask)

        cv2.imshow('Optical Flow LK', frame_bgr)
        if cv2.waitKey(50) & 0xFF == 27:  # press Esc to exit
            break

        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
