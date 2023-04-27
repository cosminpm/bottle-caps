import os


def is_valid_path(path):
    return os.path.exists(path)


if __name__ == '__main__':
    # Example usage
    image_path = "C:\\Users\\cosmi\\Desktop\\BottleCaps\\database\\test-images\\one-image\\img.png"
    if is_valid_path(image_path):
        print("The path is valid.")
    else:
        print("The path is invalid or does not exist.")
