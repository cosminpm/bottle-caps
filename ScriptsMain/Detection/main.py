import json
import uvicorn
from ScriptsMain.Detection.DetectCaps import detect_caps
from ScriptsMain.utilsFun import read_img_from_path
from fastapi import FastAPI, UploadFile, File

app = FastAPI()


def process_image(path: str):
    image = read_img_from_path(path)
    cropped_images = detect_caps(image)
    return [tuple(int(v) for v in rct) for (img, rct) in cropped_images]


@app.get("/")
async def root():
    return {"message": "Welcome to the file upload API!"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    # Save the uploaded file to disk
    with open(file.filename, "wb") as f:
        f.write(contents)
    # Process the saved file
    result = process_image(file.filename)
    json_result = json.dumps(result)
    return {"filename": file.filename, "result": json_result}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8080)
