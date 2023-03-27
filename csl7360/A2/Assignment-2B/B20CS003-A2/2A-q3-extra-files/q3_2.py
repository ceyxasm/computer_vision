import cv2
import numpy as np
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--ref', '--reference', help='Reference image')
argparser.add_argument('--img', '--probe', help='Probe image')
args = argparser.parse_args()


def crop_white_background(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = masked_img[y:y+h, x:x+w]
    return cropped_img


def resize(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 > h2:
        h, w, c = img2.shape
        img1 = cv2.resize(img1, (w, h))
    else:
        h, w, c = img1.shape
        img2 = cv2.resize(img2, (w, h))
    return img1, img2

def diff(img1, img2):
    #calculate the difference between the two images and if it is above a threshold, mark it as a change
    # if higher than threshold, plot the pixel as red
    diff = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    sum_diff = np.sum(diff)//1000
    print(sum_diff)
 #   if sum_diff > 10:
 #       return True
 #   else:
 #       return False

ref = cv2.imread(args.ref)
probe = cv2.imread(args.img)

ref_crop = crop_white_background(args.ref)
probe_crop = crop_white_background(args.img)
ref_crop, probe_crop = resize(ref_crop, probe_crop)

print('shape of ref_crop: ', ref_crop.shape)
print('shape of probe_crop: ', probe_crop.shape)

diff(ref_crop, probe_crop)
#    print('Different')
#else:
#    print('Same')

#print('Reference image aspect ratio: {}'.format(ref_crop.shape[1] / ref_crop.shape[0]))
#print('Probe image aspect ratio: {}'.format(probe_crop.shape[1] / probe_crop.shape[0]))
