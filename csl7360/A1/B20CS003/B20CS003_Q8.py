import cv2
from google.colab.patches import cv2_imshow

#### 8. (Template Matching) Write your name in capital letters on a piece of
white paper and a random letter from your name. Click photographs of these.
Implement the Template Matching algorithm and discuss your observation.

template= cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-8/template.jpg')
name_org= cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-8/name.jpg')
name_org2= name_org.copy()
name_org3= name_org.copy()
cv2_imshow(template)
cv2_imshow(name_org)

'''
ref: https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/
'''

_, img = cv2.threshold(name_org, 128, 255, cv2.THRESH_BINARY)
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.35 ## Value of threshold had to be cherry picked
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    h, w = template.shape[0], template.shape[1]
    top_left = (pt[0] + 15, pt[1] + 5)
    bottom_right = (pt[0] + w -5, pt[1] + h -5)
    cv2.rectangle(name_org, top_left, bottom_right, (0, 255, 0), 2)

cv2_imshow( name_org)
print(f'at threshold: {threshold}')



threshold = 0.6 ## Value of threshold had to be cherry picked
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    h, w = template.shape[0], template.shape[1]
    top_left = (pt[0] + 15, pt[1] + 5)
    bottom_right = (pt[0] + w -5, pt[1] + h -5)
    cv2.rectangle(name_org2, top_left, bottom_right, (0, 255, 0), 2)

cv2_imshow( name_org2)
print(f'at threshold: {threshold}')



threshold = 0.9 ## Value of threshold had to be cherry picked
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    h, w = template.shape[0], template.shape[1]
    top_left = (pt[0] + 15, pt[1] + 5)
    bottom_right = (pt[0] + w -5, pt[1] + h -5)
    cv2.rectangle(name_org3, top_left, bottom_right, (0, 255, 0), 2)

cv2_imshow( name_org3)
print(f'at threshold: {threshold}')
