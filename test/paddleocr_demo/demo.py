from paddleocr import PaddleOCR
import csv

ocr = PaddleOCR(use_angle_cls=True, use_space_char=True, use_gpu=True)
result = ocr.ocr('./img1.jpg', cls=True)

with open('./demo_out.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in result:
        csv_row= []
        for row0 in row[0]:
            csv_row.append(row0[0])
            csv_row.append(row0[1])
        csv_row.append(row[1][0])
        writer.writerow(csv_row)

