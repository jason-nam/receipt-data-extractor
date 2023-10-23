import numpy as np
import cv2
from pathlib import Path

IN_PATH = '../data/unprocessed_images/'
OUT_PATH = '../data/processed_images/'

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, 
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# adaptive thresholding
def adaptive_thresholding(image):
    return cv2.adaptiveThreshold(image, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

# dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# gaussian blur
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# flatten
def flatten(cleaned, background):
    return 255 - cv2.divide(cleaned, 255 - background, scale=256)

# increase contrast
def increase_contrast(image):
    return cv2.convertScaleAbs(image, alpha=1.5, beta=-50) # alpha (contrast) and beta (brightness)
    
# erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# reducing noise - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
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

# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

if __name__ == '__main__':
    # list all image files in the input directory
    in_files = list(Path(IN_PATH).glob('*.[jJ][pP]*[gG]'))
    
    for in_file in in_files:
        # read the image
        image = cv2.imread(str(in_file))
        
        # apply preprocessing steps
        image = get_grayscale(image)
        image = remove_noise(image)
        image = gaussian_blur(image)
        image = adaptive_thresholding(image)
        image = deskew(image)
        image = opening(image) # idk
        
        # dilate can help you get a rough estimate of the background.
        background = dilate(image)
        image = flatten(image, background)
        
        image = increase_contrast(image) # might not be necessary
        
        # save the preprocessed image to the output directory
        out_file = Path(OUT_PATH) / in_file.name
        cv2.imwrite(str(out_file), image)


if __name__ == '__main__':

    image_name = 'img2'

    # test
    image = cv2.imread(IN_PATH + image_name + '.jpg')

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
    cv2.imwrite(OUT_PATH + image_name + '.png', thresh)
    # cv2.imwrite(processed_image_path + image_name + '_opening_image.png', opening)
    # cv2.imwrite(processed_image_path + image_name + '_canny_image.png', canny)