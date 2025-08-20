import time
import cv2
import numpy as np
from of_vio.generator import fetch_data
import vpi

class LK_VPI():
    def __init__(self):
        self.p0 = None
        self.optflow = None
            

    def _goodFeatureToTrack(self, frame):
        with vpi.Backend.CPU:
            
            curFeatures, scores = frame.harriscorners(strength=0.1, sensitivity=0.01)
            self.optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 3)
            return curFeatures, scores

    def process_frame(self, frame):
        with vpi.Backend.CUDA:
            frame = vpi.asimage(frame, vpi.Format.BGR8).convert(vpi.Format.U8)
            if self.p0 is None:
                # First frame, find initial corners
                self.p0, _ = self._goodFeatureToTrack(frame)
                self.p0_cpu = self.p0.cpu()
                return None, None
            

            # Calculate where keypoints are in current frame
            p1, status = self.optflow(frame)
            p1_cpu, status_cpu = p1.cpu(), status.cpu()
            
            p1_good = p1_cpu[status_cpu == 0]
            p0_good = self.p0_cpu[status_cpu == 0]
            
            print(f"good features amount: {len(p1_cpu)}")

            self.p0 = p1

        return p1_good, p0_good







if __name__ == "__main__":
    lk_cv = LK_VPI()
    i = 0 
    for img, data in fetch_data(156, 300):
        good_new, good_old = lk_cv.process_frame(img)
        print(type(good_new), type(good_old))


        
        if good_new is not None and good_old is not None:
            # Draw tracks (optional, for visualization)
            for _, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw line between old and new points
                # cv2.line(img, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                # Draw new points
                cv2.circle(img, (int(a), int(b)), 5, (0, 0, 255), -1)
        
            cv2.imshow("LK Optical Flow", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        # time.sleep(1/30)
    cv2.destroyAllWindows()