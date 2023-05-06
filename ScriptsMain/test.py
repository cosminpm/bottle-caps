import requests

url = "https://bottlecaps-production.up.railway.app/upload"
file_path = "C:/Users/cosmi/Desktop/BottleCaps/database/test-images/test-i-have/5.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.status_code)
print(response.text)