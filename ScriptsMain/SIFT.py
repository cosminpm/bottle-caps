import json
import os
from typing import Optional, Any
from pathlib import Path
import cv2
import numpy as np

from ScriptsMain.HTC import hough_transform_circle
from ScriptsMain.LABColor import get_avg_lab_from_np, find_closest_match_in_cluster_json
from ScriptsMain.blobs import get_avg_size_all_blobs
from ScriptsMain.utils import resize_image, rgb_to_bgr, read_img, get_dcp_and_kps

MATCHER = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
file_path_sift = os.path.join(script_dir, 'SIFT_variable.json')

file_path_lab = os.path.join(script_dir, 'LABColor_variables.json')
VARIABLES_LAB = json.load(open(file_path_lab))
VARIABLES_SIFT = json.load(open(file_path_sift))



# TODO: Add this to variables
PATH = Path(os.getcwd())
SORTED_CLUSTER_FILE = 'database\sorted_cluster.json'
FULL_PATH_SORTED_CLUSTER_FILE = os.path.join(PATH.parent.absolute(), SORTED_CLUSTER_FILE)


def get_rectangles(circles: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int]]:
    """
    Based in the center of the circle and the ratio, transform it into a rectangle so the image can be cropped

    :param list[tuple[nt,int,int]] circles: A list with tuples of the circles, x,y (center) and radius
    :return: Returns the list of rectangles transforming into width and height
    """
    rectangles = []
    for x, y, r in circles:
        x1 = x - r
        y1 = y - r
        width = r * 2
        height = r * 2
        rectangles.append((x1, y1, width, height))
    return rectangles


# TODO: Improve here and modify index to variable of max in LAB COLOR
def calculate_success(new: [dict], index: int) -> float:
    """
    Calculates how successful was the cap match based on the descriptors and the len of the matches

    :param dict new: entry with the dictionary of the cap
    :return: returns the percentage of the success rate
    """
    first_param = (new['num_matches'] / new['len_rectangle_dcp']) * 0.49
    second_param = (new['num_matches'] / new['len_cap_dcp']) * 0.49
    third_param = (VARIABLES_LAB['MAX_DISTANCE'] - index) / VARIABLES_LAB['MAX_DISTANCE'] * 0.02

    result = first_param + second_param + third_param
    return result


def get_best_match(dcp_rectangle: np.ndarray, best_matches_lab: list) -> Optional[dict]:
    """
    Gets the best match based on the success rate from all the json matches

    :param np.ndarray dcp_rectangle: Descriptors of the rectangle image
    :param list best_matches_lab: List of matches based on LAB path, avg_lab
    :return: Returns a dictionary with all the information about the cap
    """

    matches = compare_descriptors_rectangle_with_database_descriptors(dcp_rectangle, best_matches_lab)
    if matches is None:
        return None
    cap_file = {'num_matches': 0,
                'path_file': None,
                'success': 0}

    index = 1
    for match in matches:
        new = {'num_matches': len(match[0]),
               'path_file': match[1],
               'len_cap_dcp': match[2],
               'len_rectangle_dcp': match[3]}
        # Important, here is how we define the success rate

        new['success'] = calculate_success(new, index)
        if new['success'] > cap_file['success']:
            cap_file = new
        index += 1
    return cap_file


# TODO: Improve here so the comparison is not with all the images
def compare_descriptors_rectangle_with_database_descriptors(dcp_rectangle: np.ndarray, best_matches_lab: list):
    """
    Compare the current image with the database and returns a list with the matches,name,and both descriptors

    :param np.ndarray dcp_rectangle: the descritpros of the rectangle
    :param list best_matches_lab: list of matches based on LAB ["path":<PATH>, "avg_lab":<(1,2,3)>]
    :return: Returns all the matches of that image
    """
    matches = []

    entries = [entry['json_path'] for entry in best_matches_lab]

    for name_img in entries:
        kps_cap, dcps_cap = get_kps_and_dcps_from_json(name_img)

        # A match is a tuple which contains the matches, the PATH of the cap, the len of the photo cap and the len of
        # descriptors of the rectangle
        match = (
            get_matches_after_matcher_sift(dcps_cap, dcp_rectangle), name_img, len(dcps_cap), len(dcp_rectangle))
        matches.append(match)
    return matches


def get_name_from_json(path):
    """

    :param str path: Path of the image
    :return: Returns the name of the image based on their name
    """
    with open(path, "r") as file:
        data = json.load(file)
        return data["name"].split('.')[-2]


def get_kps_and_dcps_from_json(path: str) -> tuple:
    """
    Loads the descriptors and keypoints of the json to the correct format

    :param str path: PATH of the cap and the json
    :return: returns a tuple which contains the keypoints and descriptors
    """
    with open(path, "r") as file:
        data = json.load(file)
        keypoints = data["kps"]
        descriptors = np.array(data["dcps"])
        keypoints = [cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id) for
                     x, y, _size, _angle, _response, _octave, _class_id in keypoints]
    return keypoints, descriptors


def crop_image_into_rectangles(photo_image: np.ndarray, rectangles: list[tuple[int, int, int, int]]) -> list[
    tuple[Any, tuple[int, int, int, int]]]:
    """
    Crop the image based on the rectangles, if the position is negative put it to zero

    :param np.ndarray photo_image: the original photo
    :param list[tuple[int, int, int, int]] rectangles: a list of tuples with the x,y and width and height position
    :return: list[np.ndarray, tuple[int, int, int, int]] Returns a list of list which contains the cropped image and the position on where it was cropped
    """
    cropped_images = []
    for x, y, w, h in rectangles:
        # Sometimes we have to guarantee that rectangle size is greater than 0
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        cropped_image = photo_image[y:y + h, x:x + w]
        if len(cropped_image) > 0:
            cropped_images.append((cropped_image, (x, y, w, h)))
    return cropped_images


def get_matches_after_matcher_sift(cap_dcp: np.ndarray, rectangle_image: np.ndarray) -> list:
    """
    Compare descriptors of the cap image and the rectangle of the cap of the photo image

    :param np.ndarray cap_dcp: descriptors pf the cap image of the database
    :param np.ndarray rectangle_image: descriptors of the image of the rectangle
    :return: returns the matches (a list) sorted limited to a max size
    """
    if cap_dcp.dtype != rectangle_image.dtype:
        if cap_dcp.dtype == np.float32:
            rectangle_image = np.array(rectangle_image, dtype=np.float32)
        else:
            cap_dcp = np.array(cap_dcp, dtype=np.float32)
    matches = MATCHER.match(cap_dcp, rectangle_image)
    return sorted(matches, key=lambda x: x.distance)[:VARIABLES_SIFT['MAX_MATCHES']]


def preprocess_image_size(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for SIFT currently it resizes it

    :param np.ndarray img: Original image, preprocess it for SIFT
    :return: np.ndarray The image preprocessed for SIFT
    """
    height, width = img.shape[:2]
    size = height * width
    max_size_img = VARIABLES_SIFT["MAX_WIDTH_IMAGE"] * VARIABLES_SIFT["MAX_HEIGHT_IMAGE"]
    resized = img
    while size > max_size_img:
        resized = resize_image(resized, 0.66)
        height, width = resized.shape[:2]
        size = height * width
    return resized


def detect_caps(img):
    # Preprocess image
    img = preprocess_image_size(img)

    _, avg_size = get_avg_size_all_blobs(img)
    cropped_images = []
    if avg_size != 0:
        _, circles = hough_transform_circle(img, avg_size)
        # Get the positions of the rectangles
        rectangles = get_rectangles(circles)
        # Crop the images from the rectangles
        cropped_images = crop_image_into_rectangles(img, rectangles)
        # Final dictionary which will contain all the positions and info from the cap
    return cropped_images


def get_best_lab_matches(rectangle_img: np.ndarray):
    avg_lab_rct = tuple(get_avg_lab_from_np(rectangle_img))
    print(avg_lab_rct)
    return find_closest_match_in_cluster_json(FULL_PATH_SORTED_CLUSTER_FILE, avg_lab_rct)


def get_dict_all_matches(path_to_image: str) -> (list[dict], np.ndarray):
    """
    This is one of the more important functions for this project, it creates the json for all the matches

    :param str path_to_image: Path to the image that is going to be analyzed
    :return: Returns a list of json with all the information about the match
    """
    img = read_img(path_to_image)
    cropped_images = detect_caps(img)
    caps_matches = []
    for rectangle_image, pos_rectangle in cropped_images:
        best_match_json = create_dict_for_one_match(rectangle_image=rectangle_image, pos_rectangle=pos_rectangle)
        caps_matches.append(best_match_json)
        print("-----")
    return caps_matches, img


def create_dict_for_one_match(rectangle_image: np.ndarray, pos_rectangle: tuple[int, int, int, int]) -> dict:
    """
    Creates the info of the json about one match, it focuses on getting the descriptors and keypoints and then getting
    the best match for that cap on the photo

    :param np.ndarray rectangle_image: Rectangle of the cap in the photo image
    :param tuple[int, int, int, int] pos_rectangle: Position of the rectangle in the original photo
    :return: dictionary with the information of the match, such as the position of the match, the name, success...
    """
    _, dcp_rectangle = get_dcp_and_kps(rectangle_image)

    # Get the best possible match for each cap
    if dcp_rectangle is not None:

        # TODO: Get best matches
        best_matches_lab = get_best_lab_matches(rectangle_image)
        print(best_matches_lab)
        best_match_json = get_best_match(dcp_rectangle, best_matches_lab)

        if best_match_json is not None and best_match_json['num_matches'] > 0:
            best_match_json['positions'] = {"x": pos_rectangle[0],
                                            "y": pos_rectangle[1],
                                            "w": pos_rectangle[2],
                                            "h": pos_rectangle[3]}
            best_match_json['name'] = get_name_from_json(best_match_json['path_file'])
            print(best_match_json)
            return best_match_json


def filter_if_best_martch_is_good_enough_all_matches(all_caps_matches: list[dict]) -> (list[dict], list[dict]):
    """
    Having all the matches, for each mach if their success ratio is enough for considering them a match

    :param list[dict] all_caps_matches: list of all the matches
    :return: returns the same input but splitting them into two different categories
    """
    good_matches = []
    bad_matches = []
    for match in all_caps_matches:
        if match is not None:
            if match['success'] > VARIABLES_SIFT['SUCCESS_MIN']:
                good_matches.append(match)
            else:
                bad_matches.append(match)
    return good_matches, bad_matches


def draw_match(img: np.ndarray, match: dict, color_name: tuple[int, int, int],
               color_circle: tuple[int, int, int]) -> np.ndarray:
    """
    Draws the info of the match, a circle around the cap, the name of the cap and the percentage of success

    :param np.ndarray img : photo original image
    :param dict match: dictionary which contains info about the match
    :param tuple[int] color_name: color used for drawing the name
    :param tuple[int] color_circle: color used for drawing the circle
    :return: np.ndarray the match drawn with the circle and the name of the cap, also the percentage of success
    """
    match_pos = match['positions']
    x, y, w, h = match_pos['x'], match_pos['y'], match_pos['w'], match_pos['h']
    center = (x + int(w / 2), y + int(h / 2))
    radius = int(w / 2)
    img = cv2.circle(img, center, radius, color_circle, 4)

    name = match['name'] + " " + "{:.2f}".format(match['success'])
    img = cv2.rectangle(img, (x, int(y + h / 2) - 10), (x + w + 25, int(y + h / 2) + 15), (0, 0, 0), -1)
    cv2.putText(img, name.upper(), (x, int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1 * 0.33, color_name, 1,
                cv2.LINE_AA)

    return img


def draw_matches_from_path(path_to_image: str, all_matches: list[dict]) -> None:
    """
    Iterates over all the matches and draws them on a single image

    :param str path_to_image:  Path of the image that is going to be drawn
    :param list[dict] all_matches: List of dict that contains the info about the matches
    """
    img = read_img(path_to_image)
    draw_matches(img=img, all_matches=all_matches)


def draw_matches(img: np.ndarray, all_matches: list[dict]):
    COLOR_NAME = rgb_to_bgr(255, 255, 0)
    GREEN_CIRCLE = rgb_to_bgr(50, 205, 50)
    RED_CIRCLE = rgb_to_bgr(255, 0, 0)

    good_matches, bad_matches = filter_if_best_martch_is_good_enough_all_matches(all_matches)

    # drawing good matches on image
    for match in good_matches:
        draw_match(img, match, COLOR_NAME, GREEN_CIRCLE)

    for match in bad_matches:
        draw_match(img, match, COLOR_NAME, RED_CIRCLE)

    cv2.imshow("Result:", img)
    cv2.waitKey(0)


def apply_main_method_to_all_images(folder_photos: str) -> None:
    """
    Main function, given a folder detect and identify all the caps, only iterates and applies the main method

    param str folder_photos: Folder of the photos that are going to be analyzed
    """
    entries = os.listdir(folder_photos)
    for entry in entries:
        path_to_image = os.path.join(folder_photos, entry)
        all_matches, img = get_dict_all_matches(path_to_image)
        if len(all_matches) > 0:
            draw_matches(img=img, all_matches=all_matches)
        else:
            print("No caps found in : {}".format(path_to_image))


def main():
    folder_photos = '../database/test-images/one-image'
    apply_main_method_to_all_images(folder_photos=folder_photos)


if __name__ == '__main__':
    main()
