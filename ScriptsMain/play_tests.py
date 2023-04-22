import cv2
import numpy as np
import requests

if __name__ == '__main__':
    # Send a test image to the Flask API endpoint and receive the response
    url = 'http://localhost:5000/process_image'
    with open(r'C:\Users\cosmi\Desktop\BottleCaps\database\test-images\test-i-have\white-background.jpg', 'rb') as f:
        r = requests.post(url, files={'image': f})

    # Print the JSON response from the Flask API endpoint
    print(r.text)