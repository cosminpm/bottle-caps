import pytest
import json
import os
from Scripts.hough_transform_circles import hough_transform_circle

file = open('./tests/data.json')
data = json.load(file)

file = open('./tests/number_bottlecaps.json')
number_caps = json.load(file)


@pytest.fixture
def name_file():
    return data["image name"]


def test_number_bottlecaps_detected(name_file):
    if name_file in number_caps:
        assert data["number bottlecaps"] == number_caps[name_file]
    else:
        assert False
