import cv2

from Classes.DetectionManager import DetectionManager
from aux_scripts import read_img


def look_in_all_images(photo_str: str):
    photo_img = read_img(photo_str)
    det_manager = DetectionManager(photo_img)
    squares = det_manager.detect_non_overlapping_squares()
    det_manager.draw_squares_detections(squares)
    det_manager.draw_percentage(squares)
    det_manager.draw_name(squares)

    cv2.imshow("Result", det_manager.photo_image)
    cv2.waitKey(0)


def get_value_const():
    pass
    # TODO: Get value of const depending on images


if __name__ == '__main__':
    look_in_all_images("./test_images/4.jpg")
