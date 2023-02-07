import cv2
from google.colab.patches import cv2_imshow

#### 1. (Spot the diff) Given a stitched image containing two very similar scenes,
find out the differences. (a) Submit your implementation. (b) Write down
your algorithm in brief. (c) Show the image where differences are suitably
marked. (d) Write down scenarios when your implementation may not work.

img= cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-1/Spot_the_difference.png')
cv2_imshow(img)

# image split in 2
h, w, _= img.shape
w_cut= w //2

s1= img[:, :w_cut]
s2= img[:, w_cut:]
cv2_imshow(s1)
cv2_imshow(s2)

'''
both images are converted to grayscale, difference is calculated and thresholded
'''
gray1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(gray1, gray2)

thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
cv2_imshow(thresholded)

'''
contours are drawn along the edges of our thresholded image,
which spots our difference
'''
contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_with_contours = s1.copy()
cv2.drawContours(img_with_contours, contours, -1, (255,0,0), 2)

cv2_imshow( img_with_contours)
