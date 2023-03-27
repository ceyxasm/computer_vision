import cv2
import numpy as np
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--folder', help='Path to the logo folder')
args = argparser.parse_args()
folder = args.folder

# function that stitches two images
def stitch(img1, img2, kp1, kp2, good):
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w, d = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
    return img3

def show_matches(scene, logo, kp1, kp2, matches):
    img3 = cv2.drawMatchesKnn(scene, kp1, logo, kp2, matches[:10], None, flags=2)
    cv2.imshow('Matches', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift_detect(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    img3 = stitch(img1, img2, kp1, kp2, good)
    return img3
#    show_matches(scene, logo, kp1, kp2, good)


def orb_detect(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = []
    for m in matches:
        if m.distance < 0.5*len(matches): #0.75
            good.append(m)
    img3 = stitch(img1, img2, kp1, kp2, good)
    return img3


def brief_detect(img1, img2):
    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = brief.compute(scene, kp1)
    kp2, des2 = brief.compute(logo, kp2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = []
    for m in matches:
        if m.distance < 0.5*len(matches): #0.75
            good.append(m)
    img3 = stitch(img1, img2, kp1, kp2, good)
    return img3


folder_list = os.listdir(folder)
folder_list.sort()
base_img = cv2.imread(folder + '/' + folder_list[0])
base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

for technique in ['sift', 'orb', 'brief']:
    for img in folder_list[1:]:
        curr_img_path = cv2.imread(folder + '/' + img)
        curr_img = cv2.cvtColor(curr_img_path, cv2.COLOR_BGR2GRAY)
        if technique == 'sift':
            base_img = sift_detect(base_img, curr_img)
        elif technique == 'orb':
            base_img = orb_detect(base_img, curr_img)
        elif technique == 'brief':
            base_img = brief_detect(base_img, curr_img)
        else:
            print('Invalid technique')
            exit()  
    cv2.imwrite('stitch_' + technique + '.jpg', base_img)

