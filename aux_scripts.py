def find_dominant_color(img):
    colors = {}
    for pix in img[0]:
        pix = tuple(pix)
        pix = (pix[0] // 2, pix[1] // 2, pix[2] // 2)
        if pix not in colors:
            colors[pix] = 1
        else:
            colors[pix] += 1
    dominant = max(colors, key=colors.get)
    return tuple((int(dominant[0] * 2), int(dominant[1] * 2), int(dominant[2] * 2)))


def compare_if_same_color(c1, c2, ratio):
    sum_numb_1 = c1[0] / (255 * 3) + c1[1] / (255 * 3) + c1[2] / (255 * 3)
    sum_numb_2 = c2[0] / (255 * 3) + c2[1] / (255 * 3) + c2[2] / (255 * 3)
    print(abs(sum_numb_1 - sum_numb_2))
    return abs(sum_numb_1 - sum_numb_2) > ratio
