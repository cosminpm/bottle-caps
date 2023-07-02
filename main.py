import json

import cv2
import keras
import numpy as np
import uvicorn

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ScriptsMain.DetectCaps import detect_caps
from ScriptsMain.Pinecone import PineconeContainer
from ScriptsMain.UtilsFun import img_to_numpy
from ScriptsMain.cnn import identify_cap, get_model



app = FastAPI()
pinecone_container = PineconeContainer()
model: keras.Sequential = get_model()

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
    caps_identified = []
    for cap in cropped_images:
        caps_identified.append(identify_cap(cap=np.array(cap[0]), model=model, pinecone_container=pinecone_container))
    positions = [tuple(int(v) for v in rct) for (img, rct) in cropped_images]
    result = {'positions': positions, 'caps_identified': caps_identified}
    print(result)
    return result


@app.get("/")
async def root():
    return {"message": "Welcome to the file upload API!"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    result = process_image(await file.read())
    return {"filename": file.filename, "result": json.dumps(result)}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8080)
