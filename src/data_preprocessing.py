import numpy as np
import cv2
from pathlib import Path

IN_PATH = '../data/unprocessed_images/'
OUT_PATH = '../data/processed_images/'

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def median_blur(image):
    return cv2.medianBlur(image,5)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, 
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# adaptive thresholding
def adaptive_thresholding(image):
    return cv2.adaptiveThreshold(image, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 13, 2)

# dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# gaussian blur
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# increase contrast
def increase_contrast(image):
    return cv2.convertScaleAbs(image, alpha=1.5, beta=-50) # alpha (contrast) and beta (brightness)
    
# erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# reducing noise - erosion followed by dilation
def morphology(image):
    kernel = np.ones((5,5),np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle based on its value
    if 45 > angle >= 0:
        angle = -angle
    elif -45 <= angle < 0:
        angle = -angle
    elif angle >= 45:
        angle = 90 - angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def flatten(cleaned, background):
    # Calculate the mean value of the background
    mean_value = np.mean(background)
    
    # Subtract the mean value from the cleaned image and normalize to [0, 255]
    flattened = np.clip(cleaned - mean_value, 0, 255).astype(np.uint8)
    
    return flattened

# removes noise
def denoise(image):
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def sharpening(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def increase_size(image):
    return cv2.resize(image, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    # list all image files in the input directory
    in_files = list(Path(IN_PATH).glob('*.[jJ][pP]*[gG]'))
    
    for in_file in in_files:
        # read the image
        image = cv2.imread(str(in_file))

        # TODO might need to consider brightening the white spots? or finding a way to making characters
        # more defined

        # apply preprocessing steps
        image = increase_size(image)

        image = get_grayscale(image)
        # cv2.imshow("grayscale", image)
        # cv2.waitKey(0)

        # apply some denoising to get rid of some pepperiness
        image = denoise(image)

        # sharpen image
        image = sharpening(image)

        image = increase_contrast(image)

        image = deskew(image)
        # cv2.imshow("deskew", image)
        # cv2.waitKey(0)

        image = gaussian_blur(image)
        # cv2.imshow("gaussian blur", image)
        # cv2.waitKey(0)

        image = adaptive_thresholding(image)
        # cv2.imshow("adaptive thresholding", image)
        # cv2.waitKey(0)

        # save the preprocessed image to the output directory
        out_file = Path(OUT_PATH) / in_file.name
        cv2.imwrite(str(out_file), image)