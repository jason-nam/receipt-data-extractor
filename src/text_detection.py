try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pytesseract import Output
import csv
from pathlib import Path
import cv2

INPUT_PATH = '../data/processed_images/'
OUTPUT_PATH = '../data/text_box_images/'

# function to plot boxes around texts
def plot_text_box(image_name, image_type):

    # Getting boxes around text
    img = cv2.imread(INPUT_PATH + image_name + '.png')

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d.keys())

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show image
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # save image
    cv2.imwrite(OUTPUT_PATH + image_name + '_text_box.png', img)

# Path to the input image
image_name = 'img1'
image_type = '_thresholded_image'

# img_path = Path(INPUT_IMAGE_PATH + image_name + image_type + '.png')
plot_text_box(image_name, image_type)



