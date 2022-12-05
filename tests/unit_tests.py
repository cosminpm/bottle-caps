from aux_scripts import resize_image, get_name_from_path, resize_img_pix_with_name


def test_best_size():
    path_tests_images = "./"

    size_px = 50
    path_cap = r"../caps_imgs/amstel00.jpg"
    resize_img_pix_with_name(path_cap, path_tests_images,size_px)


if __name__ == '__main__':
    test_best_size()
