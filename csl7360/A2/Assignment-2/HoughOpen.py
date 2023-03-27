import numpy as np
import cv2
import math
import argparse

start = cv2.getTickCount()
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--image", required=True, help="path to input image")
args = argparser.parse_args()

# detect and draw line using openCV
def detectLineOpenCV(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    canny = cv2.Canny(binary, 50, 150, apertureSize=7)
    lines = cv2.HoughLines(canny, 1, np.pi/180, 120)
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

image = cv2.imread(args.image)
image = detectLineOpenCV(image)
stop = cv2.getTickCount()
cv2.imshow("OpenCV", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("OpenCV.jpg", image)
print("OpenCV time: {} ms".format((stop - start) / cv2.getTickFrequency() * 1000))
