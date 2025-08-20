from of_vio.auairtools.auair import AUAIR

annotFile = '/home/user/Documents/vio/auair2019annotations/annotations.json'
dataDir = '/home/user/Documents/vio/auair2019data/images/'

# Create a AUAIR object.
auairdataset = AUAIR(annotation_file=annotFile, data_folder = dataDir)

for i in range(200,250):
    im = f"frame_20190829091111_x_0000{i}.jpg"
    auairdataset.display_image(im)

print("Done")