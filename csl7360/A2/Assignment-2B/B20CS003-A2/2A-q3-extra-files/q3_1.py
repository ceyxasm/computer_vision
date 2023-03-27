import logging
import argparse
import os
import cv2
import numpy as np


argparser = argparse.ArgumentParser()
argparser.add_argument('--ref', help='path to reference image')
argparser.add_argument('--img', help='path to image to be compared')
arg = argparser.parse_args()

ref_path = arg.ref
img_path = arg.img

def sift_compare(ref_path, img_path):
    ref_img = cv2.imread(ref_path)
    img = cv2.imread(img_path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    kp2, des2 = sift.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            
    print('number of matches: ',len(good))
    if len(good) > 900:
        return True
    else:
        return False

#ref_path = os.path(ref_path)
#img_path = os.path(img_path)
if sift_compare(ref_path, img_path):
    print('The images are similar.')
else:
    print('The images are different.')
