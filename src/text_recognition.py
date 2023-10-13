import re
import cv2
import pytesseract
from pytesseract import Output
import csv
from pathlib import Path

IN_PATH = '../data/processed_images/'
OUT_PATH = '../data/image_to_data/'

def extract_image_data(image_name):

    img = cv2.imread(IN_PATH + image_name + '.png')
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    keys = list(d.keys())

    # path to the CSV file
    csv_file_path = Path(OUT_PATH + image_name + '.csv')

    # Writing the OCR results to a CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for i in range(len(d['text'])):
            csv_row = [d['left'][i], d['top'][i], d['width'][i], d['height'][i], d['text'][i]]
            writer.writerow(csv_row)

if __name__ == '__main__':
    image_name = 'img2'
    image_type = '_thresholded_image'
    extract_image_data(image_name)