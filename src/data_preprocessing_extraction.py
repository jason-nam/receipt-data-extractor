# preprocessing data

import numpy as np
import cv2
from pathlib import Path

IN_PATH = '../data/in/'
OUT_PATH = '../data/out/processed_images/'

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
                                 cv2.THRESH_BINARY_INV, 5, 2)

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

def normalize_receipt(image):
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image -= image.min()
    image /= image.max()
    image *= 255
    return image

# arrange points in the order of top left, top right, bottom right, bottom left based on sums
def order_points(points):
    rect = np.zeros((4, 2), dtype='float32')
    points = np.array(points)
    sum = points.sum(axis=1)
    # smallest sum on top left
    rect[0] = points[np.argmin(sum)]
    # largest sum on bottom right
    rect[2] = points[np.argmax(sum)]

    diff = np.diff(points, axis=1)
    # smallest difference on the top right
    rect[1] = points[np.argmin(diff)]
    # largest difference on bottom left
    rect[3] = points[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()


def find_receipt(points):
    (tl, tr, br, bl) = points
    # find max width of the polynomial using the points
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # find max height of the polynomial using the points
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]]

    return order_points(destination_corners)

def extract_receipt(image):
    # resize image to usable size
    dim_limit = 1080
    max_dim = max(image.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)
    # create copy of original image for returning
    orig_image = image.copy()

    # remove text from document to create a perfectly white page
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)

    # remove the background using grabcut (automatically tries to detect foreground and background)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (20, 20, image.shape[1] - 20, image.shape[0] - 20)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = image * mask2[:, :, np.newaxis]

    # grayscale because canny only works on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    # edge detection
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # find contours for detected edges
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # keep only largest contour
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # contour approximation for finding edges
    # loop over the contours
    if len(page) == 0:
        return orig_image
    for c in page:
        # approximate contour
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # stop if contour has 4 points
        if len(corners) == 4:
            break
    # take four corners and order them to top left, top right, bottom left, bottom right
    corners = sorted(np.concatenate(corners).tolist())
    corners = order_points(corners)

    receipt_corners = find_receipt(corners)
    receipt = cv2.getPerspectiveTransform(np.float32(corners), np.float32(receipt_corners))
    # warp image to only include the receipt and align
    return cv2.warpPerspective(orig_image, receipt, (receipt_corners[2][0], receipt_corners[2][1]),
            flags=cv2.INTER_CUBIC)




if __name__ == '__main__':
    # list all image files in the input directory
    in_files = list(Path(IN_PATH).glob('*.[jJ][pP]*[gG]'))

    # Ideal conditions (overall faster, pickier in terms of inputs, cleaner csv output,
    # can have higher accuracy when it does work)
    # - fold up receipt to only include items and minimize size/distance of camera
    # - dark background with low reflectivity
    # - sometimes works with flash depending on background
    # - try to keep receipt as straight and flat as possible

    for in_file in in_files:
        # read the image
        image = cv2.imread(str(in_file))

        # apply preprocessing steps
        image = extract_receipt(image)

        # upscale image
        image = increase_size(image)

        # normalize
        image = normalize_receipt(image)

        # grayscale
        image = get_grayscale(image)

        # increase contrast
        image = increase_contrast(image)

        # save the preprocessed image to the output directory
        out_file = Path(OUT_PATH) / in_file.name
        cv2.imwrite(str(out_file), image)