import cv2
from google.colab.patches import cv2_imshow

#### 7. (White on Black or Black on White): Given a word image find out if the
word is bright text on a dark background or dark text on bright background.

import easyocr

url1= '/content/computer_vision/csl7360/A1/Assignment-1/Problem-7/11_1.png'
url2= '/content/computer_vision/csl7360/A1/Assignment-1/Problem-7/27_2.png'

img1= cv2.imread( url1)
img2= cv2.imread( url2)

reader= easyocr.Reader(['en'])
result1= reader.readtext(url1)
result2= reader.readtext(url2)

def in_out(img,  result):
  '''basically calculates the average pixel value of text pixel and pixels that 
  are not part of text; and accordingly decides
  '''

  imgb= cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
  coors= result[0][0]
  top= coors[0][1]
  bottom = coors[2][1]
  left = coors[0][0]
  right = coors[2][0]

  h= img.shape[0]
  w= img.shape[1]

  pin_sum=0
  pout_sum=0
  pout= h*w -(right-left)*(bottom-top)
  pin= (right-left)*(bottom-top)

  for y in range(h):
    for x in range(w):
      if x>left and x<right and y>top and y<bottom:
        pin_sum += imgb[y][x]
      else: 
        pout_sum += imgb[y][x]

  out_rat= pout_sum/ pout
  in_rat= pin_sum/ pin

  cv2_imshow(img)
  if( out_rat > in_rat):
    print('Dark text on light bg')
  else:
    print('Light text on dark bg')
  
  print(f'Average pixel value of text box: {in_rat} \nAverage pixel value of remaining image: {out_rat}')
  

in_out(img1, result1)

in_out(img2, result2)
