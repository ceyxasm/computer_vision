import sys
#from google.colab.patches import cv2_imshow
import numpy as np
import cv2
import PIL.Image

def find_circle_properties(image_file):
    circle = cv2.imread(image_file)
    circle = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)
    circle = cv2.threshold(circle, 25, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Circle", circle)
    #cv2_imshow(circle)

    min_line_width = 1000000
    line_pixel_count = 0

    for row in circle:
        line_width = 0
        flag_turn_b2w = False
        flag_turn_w2b = False
        for col in row:
            if col == 0:
                line_width += 1
                line_pixel_count += 1
            if line_width > 0 and col == 255:
                flag_turn_b2w = True
            if flag_turn_b2w and col == 0:
                flag_turn_w2b = True
        if flag_turn_w2b:
            min_line_width = min(min_line_width, line_width // 2)

    circumference = line_pixel_count / min_line_width
    radius = circumference / (2 * np.pi)
    area = np.pi * radius * radius

    print(f'circumference: {circumference}')
    print(f'area: {area}')

if __name__ == "__main__":
    image_file = sys.argv[1]
    find_circle_properties(image_file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

