from kp_and_descriptors import get_kp_and_dcp
from main import read_img
from test_aux_fun import aux_fun


def main_test():
    img = read_img("../test_images/3.jpg")
    print(type(img))
    kp, _ = get_kp_and_dcp(img)
    aux_fun.draw_kp("../test_images/3.jpg", kp[:35])


main_test()
