import pytesseract
from pytesseract import Output
import csv
from pathlib import Path
import cv2

IN_PATH = '../data/processed_images/'
OUT_PATH = '../data/text_box_images/'

# function to plot boxes around texts
def plot_text_box(image_name):

    # Getting boxes around text
    img = cv2.imread(IN_PATH + image_name + '.png')

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d.keys())

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # save image
    cv2.imwrite(OUT_PATH + image_name + '_text_box.png', img)

if __name__ == '__main__':
    image_name = 'img2'
    plot_text_box(image_name)



