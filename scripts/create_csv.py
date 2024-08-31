import os
import csv
if __name__ == '__main__':
    image_folder = 'database/caps'

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    with open('image_names.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["img_names"])

        for image in image_files:
            writer.writerow([image])
