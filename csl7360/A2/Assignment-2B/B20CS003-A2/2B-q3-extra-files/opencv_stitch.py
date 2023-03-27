import numpy as np
import argparse
import cv2
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--folder', required=True, help='Path to the folder')
args = argparser.parse_args()
folder_path = args.folder

image_list = os.listdir(folder_path)
image_list.sort()
images = []

for image in image_list:
    image_path = os.path.join(folder_path, image)
    image = cv2.imread(image_path)
    images.append(image)

stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

if not status:
    cv2.imwrite('stitched_opencv.jpg', stitched)
    cv2.imshow('Stitched', stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Stitching failed')
