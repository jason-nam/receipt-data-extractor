# detect text and export image with text boxes

import pytesseract
from pytesseract import Output
import csv
from pathlib import Path
import cv2

IN_PATH = '../data/out/processed_images/'
OUT_PATH = '../data/out/text_box_images/'

# function to plot boxes around texts
def plot_text_box(image):

    # Getting boxes around text

    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    # print(d.keys())

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def export_text_box_image(file):
    image = cv2.imread(str(file))
    image = plot_text_box(image)

    return image

def main():
    in_files = list(Path(IN_PATH).glob('*.[jJ][pP]*[gG]'))
    
    for in_file in in_files:
        image = export_text_box_image(in_file)

        print(str(in_file))

        out_file = Path(OUT_PATH) / in_file.name
        cv2.imwrite(str(out_file), image)

if __name__ == '__main__':
    main()