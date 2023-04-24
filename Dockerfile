FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=C:\Users\cosmi\Desktop\BottleCaps\ScriptsMain

COPY . .

ENTRYPOINT [ "python", "./ScriptsMain/DetectAPI.py" ]
