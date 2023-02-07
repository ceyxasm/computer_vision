import cv2
from google.colab.patches import cv2_imshow

#### 2. (Distance in images-1) Given an image of the map of India, find out the
pixel distance between two states. [Hint: use off-the-shelf OCR] (a) Submit
your implementation. (b) Write down the limitations of your approach.

import cv2 
from google.colab.patches import cv2_imshow
map = cv2.imread('/content/computer_vision/csl7360/A1/Assignment-1/Problem-2/india-map.jpg')
cv2_imshow(map)

import easyocr

reader= easyocr.Reader(['en'])
result= reader.readtext('/content/india-map.jpg')

# detected text
c=0
places= []
for i in result:
  places.append(i[1])
  print(i[1], c)
  c+=1
]

# format of our result
result[1]

''' of all the text detected, all is not useful
therefore states are cherry picked and organized in proper data format

> the midpoint of the coordinates of the text is taken as its location,
> this is used to calculate the distance
'''
states_index= [3, 5, 10, 12, 16, 19, 20, 21, 23, 28, 30, 31, 32,
               36, 39, 41, 42, 48, 49, 50, 56, 60, 73, 74, 76, 84, 97]
state_names= ['LADAKH', 'JAMMU AND KASHMIR', 'HIMACHAL PRADESH', 'PUNJAB', 
              'CHANDIGARH', 'UTTARAKHAND', 'HARYANA', 'DELH', 'UTTAR PRADESH', 
              'RAJASTHAN', 'ASSAM', 'NAGALAND', 'BIHAR', 'MEGHALYA', 'MANIPUR', 
              'JHARKHAND', 'WEST BENGAL', 'MIZORAM', 'GUJARAT', 'MADHYA PRADESH',
              'ODISHA', 'MAHARASHTRA', 'GOA', 'ANDHRA PRADESH', 'KARNATAKA', 'TAMIL NADU', 'ARUNACHAL PRADESH']

state_dict ={}
state_codes= []
for i in range( len(states_index)):
  key = str( states_index[i])

  state_codes.append( (state_names[i], states_index[i])  )

  coords= result[ states_index[i]][0]
  x_coord= 0
  y_coord= 0
  for i in coords:
    x_coord= x_coord + i[0]
    y_coord= y_coord + i[1]
  
  x_coord= x_coord/4
  y_coord= y_coord/4
  state_dict[ key]= [x_coord, y_coord]

state_dict

print('State names and cooresponding number code is given:\n ')
print( state_codes)

code1= input('enter first code: ')
code2= input('enter second code: ')

coor_diff= [ state_dict[code1][0]- state_dict[ code2][0], state_dict[code1][1] - state_dict[ code2][1]  ] 

from numpy.linalg import norm
dist= norm( coor_diff)

print(f'pixel distance is {dist}')
