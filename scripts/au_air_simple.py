from of_vio.auairtools.auair import AUAIR
import cv2
from dataclasses import dataclass
from typing import List

annotFile = '/home/user/Documents/vio/auair2019annotations/annotations.json'
dataDir = '/home/user/Documents/vio/auair2019data/images/'



# Create a AUAIR object.
auairdataset = AUAIR(annotation_file=annotFile, data_folder = dataDir)

@dataclass
class ImageData:
    image_name: str
    longtitude: float
    latitude: float
    altitude: float
    linear_x: float
    linear_y: float
    linear_z: float
    angle_phi: float
    angle_theta: float
    angle_psi: float

def fetch_data(start, stop):
    for index in range(start, stop):
        name = f"frame_20190829091111_x_0000{index}.jpg"
        img, data_dict = auairdataset.get_data_by_name(name)
        # 'altitude': 19921.6, 
        # 'linear_x': 0.03130074199289083, 
        # 'linear_y': 0.028357808757573367, 
        # 'linear_z': 0.0744575835764408, 
        # 'angle_phi': -0.06713105738162994, 
        # 'angle_theta': 0.06894744634628296, 
        # 'angle_psi': 1.1161083340644837
        image_data = ImageData(
            image_name=data_dict["image_name"],
            longtitude=data_dict["longtitude"],
            latitude=data_dict["latitude"],
            altitude=data_dict["altitude"],
            linear_x=data_dict["linear_x"],
            linear_y=data_dict["linear_y"],
            linear_z=data_dict["linear_z"],
            angle_phi=data_dict["angle_phi"],
            angle_theta=data_dict["angle_theta"],
            angle_psi=data_dict["angle_psi"]
        )
        yield img, image_data


# for i in range(200,250):
#     im = f"frame_20190829091111_x_0000{i}.jpg"
#     auairdataset.display_image(im)
for img, data in fetch_data(156, 166):
    cv2.imshow("xxx", img)
    cv2.waitKey(100)


