import json
import os

from ScriptsMain.SIFT import get_dict_all_matches


def get_all_names_from_all_matches(all_matches:list[dict]):
    result = set()
    for match in all_matches:
        result.add(match['name'])
    return result

def test_get_dict_all_matches_i_have():
    folder_photos = '../database/test-images/test-i-have'
    entries = os.listdir(folder_photos)

    file_solution = open('../database/test-images/solution-test-i-have.json')
    json_solution = json.load(file_solution)

    for entry in entries:
        path_to_image = os.path.join(folder_photos, entry)
        all_matches = get_dict_all_matches(path_to_image)

        result_all_matches = get_all_names_from_all_matches(all_matches)
        expected_result = set(json_solution[entry])
        assert expected_result == result_all_matches


"""


[{'num_matches': 18, 'path_file': 'C:\\Users\\cosmi\\Desktop\\BottleCaps\\database\\cluster\\(128, 128, 128)_gray\\cap-772_100.json', 'len_cap_dcp': 29, 'len_rectangle_dcp': 333, 'success': 0.4790307548928239, 'positions': {'x': 432, 'y': 236, 'w': 228, 'h': 228}, 'name': 'cap-772_100'}, {'num_matches': 35, 'path_file': 'C:\\Users\\cosmi\\Desktop\\BottleCaps\\database\\cluster\\(128, 128, 128)_gray\\cap-873_100.json', 'len_cap_dcp': 63, 'len_rectangle_dcp': 517, 'success': 0.4335912314635719, 'positions': {'x': 518, 'y': 560, 'w': 240, 'h': 240}, 'name': 'cap-873_100'}, {'num_matches': 39, 'path_file': 'C:\\Users\\cosmi\\Desktop\\BottleCaps\\database\\cluster\\(255, 192, 203)_pink\\8-fresh_100.json', 'len_cap_dcp': 59, 'len_rectangle_dcp': 196, 'success': 0.5455076098235905, 'positions': {'x': 91, 'y': 205, 'w': 250, 'h': 250}, 'name': '8-fresh_100'}]

"""


