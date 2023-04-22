import numpy as np
from flask import Flask, request, jsonify
import cv2

from ScriptsMain.SIFT import detect_caps

app = Flask(__name__)


@app.route('/process_image', methods=['POST'])
def process_image():
    # Parse the image from the request
    image_file = request.files['image']
    image_data = image_file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    cropped_images = detect_caps(image)
    result = [tuple(int(v) for v in rct) for (img, rct) in cropped_images]
    print(result)
    # Return the JPEG file as a response with the correct MIME type
    return jsonify(result)


if __name__ == '__main__':
    app.run()
