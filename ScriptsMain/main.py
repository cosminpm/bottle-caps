import json
import uvicorn

from fastapi import FastAPI, UploadFile, File
from DetectCaps import detect_caps
from utilsFun import read_img_from_path

# curl -X POST -F "file=@C:\Users\cosmi\Desktop\BottleCaps\database\test-images\test-i-have\5.jpg" http://127.1.0.2:8080/upload
# C:\Users\cosmi\Desktop\BottleCaps\database\test-images\test-i-have\5.jpg

# bottlecaps-production.up.railway.app
# curl -X POST -F "file=@C:\Users\cosmi\Desktop\BottleCaps\database\test-images\test-i-have\5.jpg" bottlecaps-production.up.railway.app/upload
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Install command
# apt-get update && apt-get install -y libopencv-dev && python ScriptsMain/main.py

# Usefull
# curl -X POST -F "file=@C:\Users\cosmi\Desktop\BottleCaps\database\test-images\test-i-have\5.jpg" https://bottlecaps-production.up.railway.app/upload


app = FastAPI()

ASGFAWEfg
def process_image(path: str):
    image = read_img_from_path(path)
    cropped_images = detect_caps(image)
    return [tuple(int(v) for v in rct) for (img, rct) in cropped_images]


@app.get("/")
async def root():
    return {"message": "Welcome to the file upload API!"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    result = process_image(file.filename)
    json_result = json.dumps(result)
    return {"filename": file.filename, "result": json_result}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
