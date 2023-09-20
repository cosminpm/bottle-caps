import datetime
from typing import List

import firebase_admin
from firebase_admin import credentials, storage
import os

BETA_TESTER = {
    "id": "BetaTester"
}


class Firebase:
    def __init__(self):
        cred = credentials.Certificate('./ScriptsMain/bottlecaps-keys.json')
        firebase_admin.initialize_app(cred)
        self.bucket = storage.bucket("bottlecaps-85ba4.appspot.com")
        self.users_folder = "users"
        self.bottle_caps_folder = "bottle_caps"
        self.uploaded_images_folder = "uploaded_images"

    def upload_image(self, path_local: str, path_remote: str) -> None:
        blob = self.bucket.blob(path_remote)
        blob.upload_from_filename(path_local)

    def get_image(self, path_query: str):
        blob = self.bucket.blob(path_query)
        path = blob.generate_signed_url(expiration=datetime.timedelta(minutes=15), method="GET")
        return path

    @staticmethod
    def join_path_remote(items: List[str]) -> str:
        result = ""
        for item in items[:-1]:
            result += item + "/"
        result += items[-1]

        return result
