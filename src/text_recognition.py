import re
import cv2
import pytesseract
from pytesseract import Output
import csv
from pathlib import Path

IN_PATH = '../data/processed_images/'
OUT_PATH = '../data/image_to_data/'

def extract_image_data(image, csv_file_path):

    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    keys = list(d.keys())

    # Writing the OCR results to a CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for i in range(len(d['text'])):
            csv_row = [d['left'][i], d['top'][i], d['width'][i], d['height'][i], d['text'][i]]
            writer.writerow(csv_row)

if __name__ == '__main__':

    in_files = list(Path(IN_PATH).glob('*.[jJ][pP]*[gG]'))
    
    for in_file in in_files:
        
        image = cv2.imread(str(in_file))

        # path to the CSV file
        csv_file_path = Path(OUT_PATH + str(in_file.stem) + '.csv')

        extract_image_data(image, csv_file_path)

        print(str(in_file.name))