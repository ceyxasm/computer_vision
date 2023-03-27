import numpy as np
import cv2
import math
import argparse

start = cv2.getTickCount()
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--image", required=True, help="path to input image")
args = argparser.parse_args()


def hough_transform(img):
    H = np.zeros((2*d, len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)

    edge_point_count = len(x_idxs)
    for i in range(edge_point_count):
        x= x_idxs[i]
        y = y_idxs[i]
        for j in range(len(thetas)):
            rho = int(x * math.cos(thetas[j]) + y * math.sin(thetas[j]) + d)
            H[rho, j] += 1
    return H


img = cv2.imread(args.image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = img.copy()

h,w = gray.shape
d = int(math.sqrt(h**2 + w**2))
thetas = np.deg2rad(np.arange(-90.0, 90.0, 1))
rhos = np.arange(-d, d)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150, apertureSize=3)
H = hough_transform(edges)
idx = np.argmax(H)
rho = rhos[idx // len(thetas)]
theta = thetas[idx % len(thetas)]
print("rho: ", rho, "theta: ", theta)

a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho
x1 = int(x0 + 1000 * (-b))
y1 = int(y0 + 1000 * (a))
x2 = int(x0 - 1000 * (-b))
y2 = int(y0 - 1000 * (a))
stop = cv2.getTickCount()

cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite("houghlines.jpg", img2)
print("Scratch time: {} ms".format((stop - start) / cv2.getTickFrequency() * 1000))

