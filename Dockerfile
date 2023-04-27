FROM python:3.9

COPY . ./
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt


WORKDIR ./ScriptsMain
ENTRYPOINT [ "python", "DetectAPI.py"]

