## Computer Vision
### Assignment-1
#### Abu Shahid
#### B20CS003

____
* The implementation can be found in the `.ipynb` file attached
* Execution of P3.py (question number 3) <br>
``` python3 P3.py <img_file.png>```
* To execute the file in google colab,
	* Comment line numbers: 11, 42, 43
	* Uncomment line numbers: 2, 12
____ 
###### <strong>Question 1- Spot the difference </strong>

a. Implementation can be found in the .ipynb file attached <br>
b. Algorithm
```py
let img1, img2
gray1 = gray( img1)
gray2 =  gray( img2)
diff = absolute_difference( gray1, gray2)
threshold -> the 'diff' is thresholded at pixel value= 25. 
#This makes all the similar regions black and the difference ican be seen as white silhouette in the thresholded image.
contour -> the active regions in 'threshold' are then used to draw contours
result -> contour image is then superimposed on the original image to get the difference
```
c.
<strong> Original Image </strong><br>
![original](Assignment-1/Problem-1/Spot_the_difference.png)
<br><strong style="text-align: center;" >Threshold of difference </strong><br>
![threshold](Assignment-1/Problem-1/threshold.png)
<br><strong style='text-align : center;'>Result</strong><br>
![result](Assignment-1/Problem-1/difference.png)<br>

d. <strong> Limitations </strong>
* The algorithm works by calculating pixel-pixel difference. So if the image is shifted, then then difference will be shown in the entire image.
* For the same reason, the algorithm is very sensitive to noise. Anything that is not an exact copy will be shown as difference.
* Minor differences may get lost while thresholding.

###### <strong>Question 2- Distance in images-1</strong>
a. Implementation can be found in the .ipynb file attached <br>
* The algorithm calculates mid-point of the bounding box given by the OCR. <br>
* These coordinates are then used to calculate the distance. <br>
* Each state is given a numerical code. <br>
	![result](Assignment-1/Problem-2/distance.png)<br>
b. Limitations <br>
* The way algorithm calculates distance is by calculating the mid-points of the co-ordinates of the bounding box of state-names. Therefore, algorithm is sensitive to font of text and orientation chosen.


###### <strong>Question 3- Distance in images-2 </strong>
* Implementation can be found in `.ipynb` and `P3.py` file attached.
* Algorithm works by calculating the thickness of the perimeter and number of pixels in the perimeter. This is used to then give the radius and then the perimeter, area.
* Thickness of perimeter is calculated by counting number of black pixels along the diameter.

![result](Assignment-1/Problem-3/result.png)



###### <strong>Question 4- Towards reading time </strong>
a. Implementation can be found in the .ipynb file attached <br>
b. Algorithm
```py
let clock_img
let clean_img = fine_line_purge( clock_img, thickness)
# fine_line_purge() erases all the lines whose thickness is less 
# than 'thickness' pixels. This is do so as to remove the 
# second's needle

let clk_gray = gray(clean_img)
binary -> clk_gray image is thresholded at pixel value 128
edge -> edge detection is applied on the binary image
let lines = HoughLines( edge, votes=45)
for all thetas in lines:
	thetas are clustered in two groups using Kmeans clustering
 angle -> tan inverse of difference of thetas
 ```
 <strong>Original clock image    </strong>   <br>
 ![original](Assignment-1/Problem-4/clock_1.png) <br>
 <strong> Image after fine_line_purge() </strong> <br> 
 ![original](Assignment-1/Problem-4/clean_1.png) <br>
  <strong>Plots from HoughLines</strong> <br>
 ![original](Assignment-1/Problem-4/line_1.png) <br>
  <strong>Result</strong> <br>
 ![original](Assignment-1/Problem-4/result_1.png) <br>
 
 c. Limitations
 * Algorithm will fail if the design of the clock has lines in it.
 * It also fails if the thickness of second's hand is comparable to other hands in the clock.
 * If the color of hands and the clock background is very similar, threshold can cause problems.
 
 
 
###### <strong>Question 5- Fun with Landmarks</strong>

NOTE: Implementation in `.ipynb` file uses grayscale images
Places chosen: <br>
* Aali-qapu <br>
![aq](Assignment-1/Problem-5/ali2.png) <br>
* Alberquerque temple <br>
![abq](Assignment-1/Problem-5/abq2.png) <br>
* Aksharadham <br>
![akd](Assignment-1/Problem-5/aks.png) <br>
 
 a. Images were resized. <br>
 b. Average of the images <br>
 ![ave](Assignment-1/Problem-5/ave.png) <br>
 c. Difference of image 1 and image 2 <br>
 ![diff](Assignment-1/Problem-5/diff.png) <br>
 d. Adding salt and pepper to image 1 with 5% probability <br>
 ![salt](Assignment-1/Problem-5/salt.png) <br>
 e. Various methods of denoising was implemented. Please check the attached `.ipynb` file <br>
 f. Convolving the mentioned kernel with image 3. <br>
 ![sobel](Assignment-1/Problem-5/sobel.png) <br>

* Implementation of all the parts in `.ipynb` file

 
 
###### <strong>Question 6- Digit recognition</strong>
* Approach
```py
image0_files = list of 0 label images
image1_files = list of 1 label images
data = [] #corresponds to our dataset
for all image0_files, image1_files:
	hppf= horizontal_projection_profile_feature( img)
	data.append( [ hppf, img.label ])

train, val, test = data.split
clf1 = SVC( train, kernel= 'linear')
clf2 = SVC( train, kernel= 'poly', degree=3)
clf3 = KNN( train, neighbours= 5)
training, validation, testing accuracies are then reported

```

* HPPF is calculated as follows
```py
    height, width = img.shape

    horizontal_projection = np.zeros(height)
    for row in range(height):
        for col in range(width):
            if img[row, col] == 0:
                horizontal_projection[row] += 1
```

* SVM Model performance <br>
![svc](Assignment-1/Problem-6/svc.png) <br>
Example predictions from Poly kernel model <br>
![svc2](Assignment-1/Problem-6/svc2.png) <br>
![svc3](Assignment-1/Problem-6/svc3.png) <br>

* KNN Model performance <br>
![knn](Assignment-1/Problem-6/knn.png) <br>
Example predictions of KNN Model <br>
![knn2](Assignment-1/Problem-6/knn2.png) <br>
![knn3](Assignment-1/Problem-6/knn3.png) <br>



###### <strong>Question 7-W on B/ B on W</strong>
Approach
* Given the image it is converted to grayscale and, the bounding box coordinates of text is extracted using EasyOCR
* Average pixel value of the pixels inside and outside the box are calculated from the grayscale image
* If the inside value is lesser than outside, it is dark text of light background, else the opposite

![ltbg](Assignment-1/Problem-7/ltbg.png) <br>
![dtbg](Assignment-1/Problem-7/dtbg.png) <br>


###### <strong>Question 8- Template Matching</strong>
<br>
Reference: [gfg](https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/)

Implementation can be found in the `.ipynb` file<br>
Approach<br>
* Image is binarized and `cv2.matchTemplate` is used to perform the template matching. 
* `cv2.TM_CCOEFF_NORMED` flag is used for matching which stands for Normalized Cross-Correlation Coefficient.
* A threshold is chosen to pick locations which can be considered a match.
* cv2.matchTemplate gives an array which represents the match between the template and the binary image at each location

Results<br>
* Template <br>
![temp](Assignment-1/Problem-8/template.jpg)
* Name <br>
![name](Assignment-1/Problem-8/name.jpg)
* Results <br>
![result](Assignment-1/Problem-8/results.png)



###### <strong>Question 9- Histogram Equalization</strong>

* Images at hand <br>
![img_b](Assignment-1/Problem-9/img.png)
![img_b](Assignment-1/Problem-9/img_.png)
* Histogram of pixel values with bin size=10, using `cv2.calcHist()` <br>
![img_b](Assignment-1/Problem-9/graph.png)
* Results after histogram equalization using `cv2.equalizeHist()` <br>
![img_b](Assignment-1/Problem-9/img_b.png)
![img_b](Assignment-1/Problem-9/img_b_.png)
* Histogram of equalized images <br>
![graph_b](Assignment-1/Problem-9/graph_b.png)




###### <strong>Question 10- Reading Mobile Number</strong>
Approach:
* Use EasyOCR to read the text in the image
* Extract last three numbers

![num](Assignment-1/Problem-10/num.png)

____
##### References
1. https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/
