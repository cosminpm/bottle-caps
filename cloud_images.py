import firebase_admin
from firebase_admin import credentials, storage


def upload_image():
    # Replace 'path/to/your/credentials.json' with the actual path to your JSON file


    bucket = storage.bucket("bottlecaps-85ba4.appspot.com")
    # Path to the local image file on your machine
    local_image_path = 'database/test-images/one-image/7.png'  # Replace this with the actual local path to your image

    # Upload the image to Firebase Storage
    blob = bucket.blob(local_image_path)
    blob.upload_from_filename(local_image_path)


def get_image():
    bucket = storage.bucket("bottlecaps-85ba4.appspot.com")

    # Get the blob for the image
    blob = bucket.blob("database/test-images/one-image/7.png")

    # Download the image to a local file
    local_image_path = './local_image.jpg'
    blob.download_to_filename(local_image_path)

    return local_image_path


if __name__ == '__main__':
    cred = credentials.Certificate('bottlecaps-keys.json')
    firebase_admin.initialize_app(cred)
    get_image()

