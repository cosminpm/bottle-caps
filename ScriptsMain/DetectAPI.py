import json

from ScriptsMain.SIFT import detect_caps
from ScriptsMain.utils import read_img_from_path
import argparse


def process_image(path: str):
    image = read_img_from_path(path)

    cropped_images = detect_caps(image)
    return [tuple(int(v) for v in rct) for (img, rct) in cropped_images]


if __name__ == '__main__':
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True,
                        help='Path to input image file')
    args = parser.parse_args()
    result = process_image(args.image_file)

    json_result = json.dumps(result)

    with open("./result_detect_caps.json", 'w') as f:
        json.dump(result, f)
