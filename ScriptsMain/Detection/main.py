import json
import sys

from DetectCaps import detect_caps
from ScriptsMain.utilsFun import read_img_from_path

sys.path.append('../ScriptsMain')


def process_image(path: str):
    image = read_img_from_path(path)
    cropped_images = detect_caps(image)
    return [tuple(int(v) for v in rct) for (img, rct) in cropped_images]

def main():
    # C:\Users\cosmi\Desktop\BottleCaps\database\test-images\test-i-have\5.jpg
    inFile = sys.argv[1]
    result = process_image(inFile)
    json_result = json.dumps(result)
    with open("../result_detect_caps.json", 'w') as f:
        json.dump(json_result, f)

if __name__ == '__main__':
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/")
    async def root():
        return {"greeting": "Hello world"}


