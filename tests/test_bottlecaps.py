import pytest
import json
from aux_scripts import get_number_of_caps_in_image


def test_number_bottlecaps_detected():
    file = open('./tests/number_bottlecaps.json')
    file_test_json = json.load(file)

    for element in file_test_json:
        if element['analyze'] == 1:
            number_of_caps = get_number_of_caps_in_image(element['name'])
            assert number_of_caps == element['nCaps']


if __name__ == '__main__':
    test_number_bottlecaps_detected()
