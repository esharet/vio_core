import time
import cv2
import numpy as np
from of_vio.generator import fetch_data

class LK_CV:
    def __init__(self):
        # Parameters for Shi-Tomasi corner detection (to pick good features to track)
        self.feature_params = dict(maxCorners=200,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.old_gray = None
        self.p0 = None

    def process_frame(self, frame):
        """
        Processes a single frame to calculate optical flow.
        
        Args:
            frame (np.array): The current frame (expected to be grayscale).
            
        Returns:
            tuple: A tuple containing:
                - good_new (np.array): Newly tracked good points.
                - good_old (np.array): Old good points corresponding to good_new.
        """
        if self.old_gray is None:
            # First frame, find initial corners
            self.old_gray = frame
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            return None, None
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame, self.p0, None, **self.lk_params)

        # Select good points
        if p1 is None or st is None:
            # No points tracked
            self.old_gray = frame
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            return None, None

        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        print(f"good features amount: {len(good_new)}")

        # Update previous frame and points
        self.old_gray = frame
        self.p0 = good_new.reshape(-1, 1, 2)

        return good_new, good_old
    

if __name__ == "__main__":
    lk_cv = LK_CV()
    i = 0 
    for img, data in fetch_data(156, 300):
        i += 1
        # print(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        good_new, good_old = lk_cv.process_frame(gray)
        
        if good_new is not None and good_old is not None:
            # Draw tracks (optional, for visualization)
            for _, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw line between old and new points
                # cv2.line(img, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                # Draw new points
                cv2.circle(img, (int(a), int(b)), 5, (0, 0, 255), -1)
        
            # cv2.imshow("LK Optical Flow", img)
            # if cv2.waitKey(1000) & 0xFF == ord('q'):
            #     break
        time.sleep(1/30)
    cv2.destroyAllWindows()
