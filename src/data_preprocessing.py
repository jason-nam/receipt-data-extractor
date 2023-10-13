import cv2
import numpy as np
from pathlib import Path

INPUT_PATH = '../data/unprocessed_images/'
OUTPUT_PATH = '../data/processed_images/'

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

image_name = 'img1'

# test
image = cv2.imread(INPUT_PATH + image_name + '.jpg')

gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

# Display images
# cv2.imshow("Original Image", image)
# cv2.imshow("Grayscale Image", gray)
# cv2.imshow("Thresholded Image", thresh)
# cv2.imshow("Opening Image", opening)
# cv2.imshow("Canny Image", canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save images
# cv2.imwrite(processed_image_path + image_name + '_gray_image.png', gray)
cv2.imwrite(OUTPUT_PATH + image_name + '.png', thresh)
# cv2.imwrite(processed_image_path + image_name + '_opening_image.png', opening)
# cv2.imwrite(processed_image_path + image_name + '_canny_image.png', canny)