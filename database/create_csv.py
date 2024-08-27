import os
import csv

image_folder = 'database/caps'

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Create and write to a CSV file
with open('image_names.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name"])  # Writing the header

    # Write each image file name as a row in the CSV
    for image in image_files:
        writer.writerow([image])
