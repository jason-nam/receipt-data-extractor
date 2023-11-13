from pathlib import Path
import cv2

from src import process_data, process_data_extraction, export_text_box_image, export_data_csv

IN_PATH = 'data/in/'
PROCESSED_IMAGE_PATH = 'data/out/processed_images/'
TEXT_BOX_IMAGE_PATH = 'data/out/text_box_images/'
CSV_DATA_PATH = 'data/out/image_to_data/'

def main():

    # process image data
    in_files = list(Path(IN_PATH).glob('*.[jJ][pP]*[gG]'))

    for in_file in in_files:
        processed_image = process_data(in_file)
        # processed_image = process_data_extraction(in_file)
        # save the preprocessed image to the output directory
        out_file = Path(PROCESSED_IMAGE_PATH) / in_file.name
        cv2.imwrite(str(out_file), processed_image)
        print(f"Successfully processed {str(out_file)}")

    # export text box detection image and text data csv file
    in_files = list(Path(PROCESSED_IMAGE_PATH).glob('*.[jJ][pP]*[gG]'))

    for in_file in in_files:
        # text box image
        text_box_image = export_text_box_image(in_file)
        out_file = Path(TEXT_BOX_IMAGE_PATH) / in_file.name
        cv2.imwrite(str(out_file), text_box_image)
        print(f"Successfully detected text from {str(out_file)}")

        # csv data file
        csv_file_path = Path(CSV_DATA_PATH + str(in_file.stem) + '.csv')
        export_data_csv(in_file, csv_file_path)
        print(f"Successfully generated {str(csv_file_path)}")

if __name__ == "__main__":
    main()
