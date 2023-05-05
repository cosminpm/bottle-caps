import json
import os
from typing import Optional, Any
from pathlib import Path
import cv2
import numpy as np

from ScriptsMain.DatabaseScripts.LABColor import get_avg_lab_from_np, find_closest_match_in_cluster_json
from ScriptsMain.Detection.DetectCaps import detect_caps
from ScriptsMain.UtilsFun import read_img_from_path, get_dcp_and_kps, rgb_to_bgr

MATCHER = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
MY_CAPS_IMGS_FOLDER = "ScriptsMain/database/cluster"
MAX_MATCHES = 200
SUCCESS_MIN = 0.30

MAX_DISTANCE = 150

PATH = Path(os.getcwd())
SORTED_CLUSTER_FILE = 'BottleCaps\database\sorted_cluster.json'
FULL_PATH_SORTED_CLUSTER_FILE = os.path.join(PATH.parent.parent.absolute(), SORTED_CLUSTER_FILE)





# TODO: Improve here and modify index to variable of max in LAB COLOR
def calculate_success(new: [dict], index: int) -> float:
    """
    Calculates how successful was the cap match based on the descriptors and the len of the matches

    :param dict new: entry with the dictionary of the cap
    :param int index: order index
    :return: returns the percentage of the success rate
    """
    first_param = (new['num_matches'] / new['len_rectangle_dcp']) * 0.49
    second_param = (new['num_matches'] / new['len_cap_dcp']) * 0.49
    third_param = (MAX_DISTANCE - index) / MAX_DISTANCE * 0.02

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
    result_matches = []
    for match in matches:
        new = {'num_matches': len(match[0]),
               'path_file': match[1],
               'len_cap_dcp': match[2],
               'len_rectangle_dcp': match[3]}
        # Important, here is how we define the success rate

        new['success'] = calculate_success(new, index)
        result_matches.append(new)
        if new['success'] > cap_file['success']:
            cap_file = new
        index += 1
    result_matches = sorted(result_matches, key=lambda x: -x['success'])
    print(result_matches)
    a = [(i['path_file'], i['success']) for i in result_matches][:10]
    for i in a:
        print(i)

    return cap_file


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
    return sorted(matches, key=lambda x: x.distance)[:MAX_MATCHES]








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
    img = read_img_from_path(path_to_image)
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
            if match['success'] > SUCCESS_MIN:
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
    img = read_img_from_path(path_to_image)
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
