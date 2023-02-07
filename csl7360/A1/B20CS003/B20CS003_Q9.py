import cv2
from google.colab.patches import cv2_imshow

#### 9. (Histogram Equalization) Choose one image from Problem 5. Show his-
togram of pixel values with bin size 10. Perform histogram equalization and

show the output image.

img = cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-5/Aali-qapu.jpg')
img_ = cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-5/Aali-qapu.jpg',
                  cv2.IMREAD_GRAYSCALE)

cv2_imshow( img)
cv2_imshow( img_)

import matplotlib.pyplot as plt
b, g, r = cv2.split(img)

hist_b = cv2.calcHist([b],[0],None,[10],[0,256])
hist_g = cv2.calcHist([g],[0],None,[10],[0,256])
hist_r = cv2.calcHist([r],[0],None,[10],[0,256])
hist__ = cv2.calcHist([img_],[0],None,[10],[0,256])

# Plot the histograms
plt.subplot(2,1,1)
plt.plot(hist_b, color='blue')
plt.xlabel('Pixel intensity values')
plt.ylabel('Number of pixels')
plt.title('Histogram of Blue channel')

plt.subplot(2,1,2)
plt.plot(hist_g, color='green')
plt.xlabel('Pixel intensity values')
plt.ylabel('Number of pixels')
plt.title('Histogram of Green channel')

plt.tight_layout()
plt.show()


plt.subplot(2,1,1)
plt.plot(hist_r, color='red')
plt.xlabel('Pixel intensity values')
plt.ylabel('Number of pixels')
plt.title('Histogram of Red channel')


plt.subplot(2,1,2)
plt.plot(hist__, color='black')
plt.xlabel('Pixel intensity values')
plt.ylabel('Number of pixels')
plt.title('Histogram of grayscale channel')

plt.tight_layout()
plt.show()



equalized_r = cv2.equalizeHist(r)
equalized_g = cv2.equalizeHist(g)
equalized_b = cv2.equalizeHist(b)
equalized__ = cv2.equalizeHist(img_)

img_norm= cv2.merge([ equalized_r, equalized_g, equalized_b] )
# Display the original and equalized images
cv2_imshow( img_norm)
cv2_imshow( equalized__)


import matplotlib.pyplot as plt
b, g, r = cv2.split(img_norm)

hist_b = cv2.calcHist([b],[0],None,[10],[0,256])
hist_g = cv2.calcHist([g],[0],None,[10],[0,256])
hist_r = cv2.calcHist([r],[0],None,[10],[0,256])
hist__ = cv2.calcHist([equalized__],[0],None,[10],[0,256])

# Plot the histograms
plt.subplot(2,1,1)
plt.plot(hist_b, color='blue')
plt.xlabel('Pixel intensity values')
plt.ylabel('Number of pixels')
plt.title('Histogram of Blue channel')

plt.subplot(2,1,2)
plt.plot(hist_g, color='green')
plt.xlabel('Pixel intensity values')
plt.ylabel('Number of pixels')
plt.title('Histogram of Green channel')

plt.tight_layout()
plt.show()


plt.subplot(2,1,1)
plt.plot(hist_r, color='red')
plt.xlabel('Pixel intensity values')
plt.ylabel('Number of pixels')
plt.title('Histogram of Red channel')


plt.subplot(2,1,2)
plt.plot(hist__, color='black')
plt.xlabel('Pixel intensity values')
plt.ylabel('Number of pixels')
plt.title('Histogram of grayscale channel')

plt.tight_layout()
plt.show()

