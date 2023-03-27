import cv2
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--scene', help='Path to the image')
argparser.add_argument('--folder', help='Path to the logo folder')
args = argparser.parse_args()
scene_path = args.scene
logo_folder = args.folder

def show_matches(scene, logo, kp1, kp2, matches):
    img3 = cv2.drawMatchesKnn(scene, kp1, logo, kp2, matches[:10], None, flags=2)
    cv2.imshow('Matches', img3)
    random2 = os.urandom(2)
    cv2.imwrite('matches' + str(random2) + '.png', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift_detect(scene, logo):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(scene, None)
    kp2, des2 = sift.detectAndCompute(logo, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    
#    show_matches(scene, logo, kp1, kp2, good)
    return len(good)


def orb_detect(scene, logo):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(scene, None)
    kp2, des2 = orb.detectAndCompute(logo, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = []
    for m in matches:
        if m.distance < 0.5*len(matches): #0.75
            good.append(m)
    return len(good)

def brief_detect(scene, logo):
    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = fast.detect(scene, None)
    kp2 = fast.detect(logo, None)
    kp1, des1 = brief.compute(scene, kp1)
    kp2, des2 = brief.compute(logo, kp2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = []
    for m in matches:
        if m.distance < 0.5*len(matches): #0.75
            good.append(m)
    return len(good)


logo_list = os.listdir(logo_folder)
logo_list.sort()
for technique in ['sift', 'orb', 'brief']:
    match_count = []
    for logo in logo_list:
        logo_path = os.path.join(logo_folder, logo)
        scene = cv2.imread(scene_path)
        logo = cv2.imread(logo_path)
        if technique == 'sift':
            matches = sift_detect(scene, logo)
        elif technique == 'orb':
            matches = orb_detect(scene, logo)
        elif technique == 'brief':
            matches = brief_detect(scene, logo)
        else:
            print('Invalid technique')
            exit()  
        match_count.append(matches)

    max_pos = match_count.index(max(match_count))
    max_match = match_count[max_pos]
    if technique == 'sift' and max_match < 30:
        print('No match found')
    elif technique == 'orb' and max_match < 30:
        print('No match found')
    elif technique == 'brief' and max_match < 250:
        print('No match found')
    else:
        print('Best match for {} is {}'.format(technique, logo_list[max_pos]))
        logo_path = os.path.join(logo_folder, logo_list[max_pos])
        scene = cv2.imread(scene_path)
        logo = cv2.imread(logo_path)
        if technique == 'sift':
            tech = cv2.xfeatures2d.SIFT_create()
        elif technique == 'orb':
            tech = cv2.ORB_create()
        elif technique == 'brief':
            tech = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1, des1 = tech.detectAndCompute(scene, None)
        kp2, des2 = tech.detectAndCompute(logo, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        show_matches(scene, logo, kp1, kp2, good)

