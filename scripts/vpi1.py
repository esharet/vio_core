import cv2
import numpy as np
import vpi




img_np = np.zeros((480, 640, 3), dtype=np.uint8)
img_vpi = vpi.asimage(img_np)
img_cuda = img_vpi.to(vpi.MemType.CUDA)