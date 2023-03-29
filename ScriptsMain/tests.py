import json
import os

from ScriptsMain.SIFT import get_dict_all_matches


def get_all_names_from_all_matches(all_matches: list[dict]):
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
        all_matches, _ = get_dict_all_matches(path_to_image)

        result_all_matches = get_all_names_from_all_matches(all_matches)
        expected_result = set(json_solution[entry])
        assert expected_result == result_all_matches
