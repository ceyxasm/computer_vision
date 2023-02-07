import cv2
from google.colab.patches import cv2_imshow

#### 5. (Fun with Landmarks) Choose three images of a world landmark from
the Google Landmark dataset (Link: https://storage.googleapis.com/gld-v2/web/index.html). The name of your chosen landmark should begin
with the first letter of your first name. For example: If your name is Adhrit,you could choose Amarnath. (a0) Resize all images to 256 × 256. Convert it
to gray. (a) Show the average of all three images. (c) Subtract Image 1 with

Image 2. (d) Add salt noise with 5% probability in one of the images. (e) Re-
move the noise. (f) Use the following 3×3 kernel: {−1, −1, −1; 0, 0, 0; 1, 1, 1}

for performing convolution in one of the images and show the output.

img1= cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-5/Aali-qapu.jpg')
img2= cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-5/abq-temple.jpg')
img3= cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-5/aksharadham.jpg')

img1= cv2.resize( img1, (256, 256))
img2= cv2.resize( img2, (256, 256))
img3= cv2.resize( img3, (256, 256))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

cv2_imshow(img1)
cv2_imshow(img2)
cv2_imshow(img3)

# average of images
imgx= img1+ img2+ img3
imgx = imgx/3
cv2_imshow( imgx)

# difference of img1 and img2
cv2_imshow(img1-img2)

# salt and pepper @5%
h,w= img1.shape[0], img1.shape[1]
total_pixels = h*w
pick_pixels = int(0.05*total_pixels)

imgy= img1.copy()

import random

for i in range(pick_pixels):
  x = random.randint(0, w-1)
  y = random.randint(0, h-1)
  r1 = random.random()
  if r1>0.5:
    imgy[y][x]=255
  else:
    imgy[y][x]=0

cv2_imshow(img1)
cv2_imshow(imgy)

  


# 3 types of de-noise v/s original
imgz = img2.copy()
bilateral = cv2.bilateralFilter(imgz, 9, 7, 7)
gaussian_blur = cv2.GaussianBlur(imgz, (3, 3), 0)
median_blur = cv2.medianBlur(imgz, 3)

cv2_imshow( imgz)
print('original\n\n')
cv2_imshow(bilateral)
print('bilateral denoising\n\n')
cv2_imshow( gaussian_blur)
print('gausian denoising\n\n')
cv2_imshow( median_blur)
print('median denoising \n\n')


# sobel filter to find horizontal edges
import numpy as np
kernel = np.array([[-1,-1,-1], [0, 0, 0], [1,1,1]])

conv = cv2.filter2D(img3, -1, kernel)
cv2_imshow( img3)
cv2_imshow( conv)
