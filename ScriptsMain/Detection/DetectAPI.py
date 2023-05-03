import json
import sys

from ScriptsMain.Identification.SIFT import detect_caps
from ScriptsMain.utilsFun import read_img_from_path


def process_image(path: str):
    image = read_img_from_path(path)
    cropped_images = detect_caps(image)
    return [tuple(int(v) for v in rct) for (img, rct) in cropped_images]


if __name__ == '__main__':
    # C:\Users\cosmi\Desktop\BottleCaps\database\test-images\test-i-have\5.jpg
    inFile = sys.argv[1]
    result = process_image(inFile)

    json_result = json.dumps(result)

    with open("../result_detect_caps.json", 'w') as f:
        json.dump(result, f)
