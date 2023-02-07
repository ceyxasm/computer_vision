import cv2
from google.colab.patches import cv2_imshow

#### 4 (Towards reading time) Given an image of a clock find out the angle
between the hour and minute hands. (a) Submit your implementation. (b)
Write down your approach for finding out the angle. (c) Write down the
limitations of your approach.

clk1_clr= cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-4/clock_1.png')
clk2_clr= cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-4/clock_2.jpg')

h1, w1= clk1_clr.shape[0], clk1_clr.shape[1] 
h2, w2= clk2_clr.shape[0], clk2_clr.shape[1]

clk1_gray = cv2.cvtColor(clk1_clr, cv2.COLOR_BGR2GRAY)
# clk1= cv2.GaussianBlur( clk1,  (7,7) ,1.8)
# clk1 = cv2.threshold(clk1, 170, 190, cv2.THRESH_BINARY)[1]

clk2_gray = cv2.cvtColor(clk2_clr, cv2.COLOR_BGR2GRAY)
# clk2 = cv2.threshold(clk2, 180, 190, cv2.THRESH_BINARY)[1]


cv2_imshow(clk1_gray)
cv2_imshow(clk2_gray)


def fine_line_purge( img, thickness):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  edges = cv2.Canny(blurred, 50, 150)

  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=50, maxLineGap=10)

  result = img.copy()

  
  '''code can be changed to make this a parameter'''

  # Iterate through the lines and erase all lines with the specified thickness
  for line in lines:
      x1, y1, x2, y2 = line[0]
      if abs(x2-x1) <= thickness or abs(y2-y1) <= thickness:
          continue
      cv2.line(result, (x1, y1), (x2, y2), (255, 255, 255), thickness)

  mask = np.zeros(img.shape, dtype=np.uint8)
  mask[:] = (255, 255, 255)

  # Bitwise-AND the mask and the copy of the original image to erase the lines with the specified thickness
  result = cv2.bitwise_and(result, mask)

  # cv2_imshow( img)
  cv2_imshow( result)
  return result

##### For clock 1

import numpy as np
clk1_clean= fine_line_purge( clk1_clr, thickness=2)
print('cleand clock\n')
_, binary = cv2.threshold(clk1_clean, 128, 255, cv2.THRESH_BINARY )
edge= cv2.Canny( binary, 100, 500, apertureSize=7)
cv2_imshow( edge)

import numpy as np

thetas= []
r_s=[]
lines = cv2.HoughLines(edge, 1, np.pi/180, 45)
for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    r_s.append(r)
    a = np.cos(theta)
    b = np.sin(theta)

    thetas.append( b/a)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(clk1_clr, (x1, y1), (x2, y2), (0, 0, 255), 2)
  
cv2_imshow( clk1_clr)



print(thetas)

!pip install scipy

import numpy as np
from sklearn.cluster import KMeans

thetas = np.array(thetas).reshape(-1, 1)

kmeans = KMeans(n_clusters=2, random_state=0).fit(thetas)

# find the mean of each cluster
means = [np.mean(thetas[kmeans.labels_ == i]) for i in range(kmeans.n_clusters)]

print("Means of each cluster:", means)

import math
diff= abs( means[0]- means[1])

print(f'angle between the hands is {math.degrees(math.atan(diff)) }')

##### For clock 2

canvas= np.zeros( (h2, w2))
circles = cv2.HoughCircles(clk2_gray, cv2.HOUGH_GRADIENT, 1.2, 100)
if circles is not None:
  circles = np.round(circles[0, :]).astype("int")

  small_clock= None
  cur_rad= 0
  for (x, y, r) in circles:
    if r> cur_rad:
      cur_rad= r
      clock= (x, y, r)
  small_clock= cv2.circle(canvas, (clock[0], clock[1]), clock[2]//2, (225, 255, 225), -1)

imageCopy = clk2_clr.copy()
imageCopy[canvas == 0] = (0, 0, 0)

x = clock[0] - clock[2]
y = clock[1] - clock[2]
h = 2*clock[2]
w = 2*clock[2]

croppedImg = imageCopy[y:y + h, x:x + w]

	# show the output image
cv2_imshow( croppedImg)

clk2_clean= fine_line_purge( croppedImg, thickness=5)
print('cleand clock\n')
_, binary = cv2.threshold(clk2_clean, 128, 255, cv2.THRESH_BINARY )
edge= cv2.Canny( binary, 100, 500, apertureSize=7)
cv2_imshow( edge)


thetas= []
r_s=[]
lines = cv2.HoughLines(edge, 1, np.pi/180, 65)
for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    r_s.append(r)
    a = np.cos(theta)
    b = np.sin(theta)

    thetas.append( b/a)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(croppedImg, (x1, y1), (x2, y2), (0, 0, 255), 2)
  
cv2_imshow( croppedImg)



len(lines)

import numpy as np
from sklearn.cluster import KMeans

thetas = np.array(thetas).reshape(-1, 1)

kmeans = KMeans(n_clusters=2, random_state=0).fit(thetas)

# find the mean of each cluster
means = [np.mean(thetas[kmeans.labels_ == i]) for i in range(kmeans.n_clusters)]

print("Means of each cluster:", means)

import math
diff= abs( means[0]- means[1])

print(f'angle between the hands is {math.degrees(math.atan(diff)) }')
