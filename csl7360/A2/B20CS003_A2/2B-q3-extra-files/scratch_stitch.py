import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('1a.jpeg')
image2 = cv2.imread('1b.jpeg')
print(image1.shape)
print(image2.shape)
# Initialize the feature detector
detector = cv2.SIFT_create()

# Find keypoints and descriptors in the two images
keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

# Initialize the feature matcher
matcher = cv2.BFMatcher()

# Match the descriptors in the two images
matches = matcher.match(descriptors1, descriptors2)

# Sort the matches by their distance
matches = sorted(matches, key=lambda x: x.distance)

# Keep only the top matches
num_top_matches = 10
matches = matches[:num_top_matches]

# Extract the corresponding keypoints in each image
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute the homography matrix using RANSAC
homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp the second image using the homography matrix
height, width, channels = image1.shape
stitched_image = cv2.warpPerspective(image2, homography, (width + image2.shape[1], height))

# Combine the two images
stitched_image[:height, :width, :] = image1

# Save the result
cv2.imwrite('stitched_image_36.jpg', stitched_image)

