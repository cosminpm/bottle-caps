import cv2
import numpy as np
import uvicorn

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from DetectCaps import detect_caps
from utilsFun import img_to_numpy

app = FastAPI()

origins = [
    "*",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_image(file_contents: bytes):
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)
    image = img_to_numpy(image)
    cropped_images = detect_caps(image)
    return [tuple(int(v) for v in rct) for (img, rct) in cropped_images]


@app.get("/")
async def root():
    return {"message": "Welcome to the file upload API!"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    result = process_image(await file.read())
    return {"filename": file.filename, "result": result}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8080)
