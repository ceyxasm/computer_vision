import os
import cv2
import argparse
import numpy as np



argparser = argparse.ArgumentParser()
argparser.add_argument('--scene', help='Path to the image')
argparser.add_argument('--folder', help='Path to logo folder')
args = argparser.parse_args()
technique = input("Enter the technique to be used: ")

def sift_match(scene, logo):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(scene, None)
    kp2, des2 = sift.detectAndCompute(logo, None)
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    return len(matches)


def orb_match(scene, logo):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(scene, None)
    kp2, des2 = orb.detectAndCompute(logo, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

def surf_match(scene, logo):
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(scene, None)
    kp2, des2 = surf.detectAndCompute(logo, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

logos = os.listdir(args.folder)
matches = []
for logo in logos:
    logo_path = os.path.join(args.folder, logo)
    logo_img = cv2.imread(logo_path)
    scene_img = cv2.imread(args.scene)
    #scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    #logo_img = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
    if technique == 'sift':
        matches.append(sift_match(scene_img, logo_img))
    elif technique == 'orb':
        matches.append(orb_match(scene_img, logo_img))
    elif technique == 'surf':
        matches.append(surf_match(scene_img, logo_img))
    else:
        print('Invalid technique')
        break
print('The best match is: ', logos[np.argmax(matches)])
