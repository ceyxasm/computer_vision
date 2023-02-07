import cv2
from google.colab.patches import cv2_imshow

#### 6. (Digit Recognition) You will be given 100 handwritten images of 0 and 1.
You have to compute horizontal projection profile features and use Nearest
Neighbour and SVM classifiers to recognize the digits. Report accuracy
and show some visual examples. Dataset (choose only 0 and 1): https://github.com/myleott/mnist_png.git

!git clone https://github.com/myleott/mnist_png.git


!tar -xvzf /content/mnist_png/mnist_png.tar.gz

sample0= cv2.imread('/content/mnist_png/training/0/17285.png')
sample1= cv2.imread('/content/mnist_png/training/1/10006.png')

cv2_imshow( sample0)
cv2_imshow( sample1)

'''creating train dataset'''
import os
import cv2
import numpy as np
import pandas as pd

folder0_path = "/content/mnist_png/training/0"
folder1_path = "/content/mnist_png/training/1"
image0_files = [f for f in os.listdir(folder0_path) if f.endswith(".png")]
image1_files = [f for f in os.listdir(folder1_path) if f.endswith(".png")]

data = []

for image_file in image0_files:
    img = cv2.imread(os.path.join(folder0_path, image_file), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    horizontal_projection = np.zeros(height)
    for row in range(height):
        for col in range(width):
            if img[row, col] == 0:
                horizontal_projection[row] += 1

    data.append(np.append(horizontal_projection, [0]))

for image_file in image1_files:
    img = cv2.imread(os.path.join(folder1_path, image_file), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    horizontal_projection = np.zeros(height)

    for row in range(height):
        for col in range(width):
            if img[row, col] == 0:
                horizontal_projection[row] += 1

    data.append(np.append(horizontal_projection, [1]))

data = np.array(data)
df = pd.DataFrame(data)

df.to_csv("train_dataset.csv", index=False)



'''creating test dataset'''
import os
import cv2
import numpy as np
import pandas as pd

folder0_path = "/content/mnist_png/training/0"
folder1_path = "/content/mnist_png/training/1"
image0_files = [f for f in os.listdir(folder0_path) if f.endswith(".png")]
image1_files = [f for f in os.listdir(folder1_path) if f.endswith(".png")]

data = []

for image_file in image0_files:
    img = cv2.imread(os.path.join(folder0_path, image_file), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    horizontal_projection = np.zeros(height)
    for row in range(height):
        for col in range(width):
            if img[row, col] == 0:
                horizontal_projection[row] += 1

    data.append(np.append(horizontal_projection, [0]))

for image_file in image1_files:
    img = cv2.imread(os.path.join(folder1_path, image_file), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    horizontal_projection = np.zeros(height)

    for row in range(height):
        for col in range(width):
            if img[row, col] == 0:
                horizontal_projection[row] += 1

    data.append(np.append(horizontal_projection, [1]))

data = np.array(data)
df = pd.DataFrame(data)

df.to_csv("t_dataset.csv", index=False)


train = pd.read_csv('/content/train_dataset.csv')
test = pd.read_csv('/content/t_dataset.csv')

train.head()

test.head()

X=  train.iloc[ :, :-1]
y= train.iloc[ : , -1]

##### SVM

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

clf1 = SVC(kernel='linear', C=1)
clf1.fit(X_train, y_train)

clf2 = SVC(kernel='poly', degree=3,  C=1)
clf2.fit(X_train, y_train)

y_pred1 = clf1.predict(X_val)
y_pred2 = clf2.predict(X_val)
accuracy1 = accuracy_score(y_val, y_pred1)
accuracy2 = accuracy_score(y_val, y_pred2)

print("Accuracy of linear kernel on validation data:", accuracy1)
print("Accuracy of poly kernel on validation data:", accuracy2)

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

y_test_pred1 = clf1.predict(X_test)
y_test_pred2 = clf2.predict(X_test)

test_accuracy1 = accuracy_score(y_test, y_test_pred1)
test_accuracy2 = accuracy_score(y_test, y_test_pred2)

# Print the accuracy
print("Accuracy of linear kernel on  test data:", test_accuracy1)
print("Accuracy of poly kernel on  test data:", test_accuracy2)


sample0= cv2.imread('/content/mnist_png/training/0/5608.png')
height= sample0.shape[0]

horizontal_projection = np.zeros(height)
for row in range(height):
    for col in range(width):
        if img[row, col] == 0:
            horizontal_projection[row] += 1

horizontal_projection= horizontal_projection.reshape(-1, 1)
horizontal_projection=  horizontal_projection.T
cv2_imshow( sample0)
print( clf2.predict( horizontal_projection))

height= sample1.shape[0]

horizontal_projection = np.zeros(height)
for row in range(height):
    for col in range(width):
        if img[row, col] == 0:
            horizontal_projection[row] += 1

horizontal_projection= horizontal_projection.reshape(-1, 1)
horizontal_projection=  horizontal_projection.T
cv2_imshow( sample1)
print( clf1.predict( horizontal_projection))

##### KNN

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("Accuracy on validation data:", accuracy)

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]
y_test_pred = clf.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on test data:", test_accuracy)


sample0= cv2.imread('/content/mnist_png/training/0/5608.png')
height= sample0.shape[0]

horizontal_projection = np.zeros(height)
for row in range(height):
    for col in range(width):
        if img[row, col] == 0:
            horizontal_projection[row] += 1

horizontal_projection= horizontal_projection.reshape(-1, 1)
horizontal_projection=  horizontal_projection.T
cv2_imshow( sample0)
print( clf.predict( horizontal_projection))


height= sample1.shape[0]

horizontal_projection = np.zeros(height)
for row in range(height):
    for col in range(width):
        if img[row, col] == 0:
            horizontal_projection[row] += 1

horizontal_projection= horizontal_projection.reshape(-1, 1)
horizontal_projection=  horizontal_projection.T
cv2_imshow( sample1)
print( clf.predict( horizontal_projection))
