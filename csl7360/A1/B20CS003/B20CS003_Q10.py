import cv2
from google.colab.patches import cv2_imshow

#### 10. (Reading Mobile Number) You will be given image of a mobile number.
Use off-the-shelf OCR and find out the last three digits of the mobile number.

url1= '/content/computer_vision/csl7360/A1/Assignment-1/Problem-10/Screenshot 2023-01-08 at 5.14.24 PM.png'
url2= '/content/computer_vision/csl7360/A1/Assignment-1/Problem-10/Screenshot 2023-01-08 at 5.14.47 PM.png'

import easyocr

img1= cv2.imread( url1)
img2= cv2.imread( url2)

reader= easyocr.Reader(['en'])
result1= reader.readtext(url1)
result2= reader.readtext(url2)

cv2_imshow( img1)
print( f'last 3 digits are {result1[0][1][-3:]}')

cv2_imshow( img2)
print( f'last 3 digits are {result2[0][1][-3:]}')
