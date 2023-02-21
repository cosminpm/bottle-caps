import cv2
import numpy as np

from ScriptsMain.CreateDatabase import get_frequency_quantized_colors
from ScriptsMain.utils import read_img, get_higher_frequency, rgb_to_bgr, colors_for_clustering


def show_image_quantized(path: str):
    image = read_img(path)
    color_frequencies = get_frequency_quantized_colors(image=image)
    representative_color = get_higher_frequency(frequencies=color_frequencies)
    print(representative_color)
    color_image = np.zeros((200, 200, 3), dtype=np.uint8)
    color_bgr = rgb_to_bgr(representative_color[0], representative_color[1], representative_color[2])
    color_image[:] = color_bgr


    # Set the font type, scale, color, and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # white color in BGR format
    thickness = 2
    text = colors_for_clustering[representative_color]
    height, width, _ = image.shape
    # Calculate the position of the text in the middle of the image
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int((width - text_size[0]) / 2)
    text_y = int((height + text_size[1]) / 2)

    # Add the text to the image
    cv2.putText(color_image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow(colors_for_clustering[representative_color], color_image)
    cv2.imshow(path, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    show_image_quantized('../database/caps-resized/2-galicia_100.jpg')


if __name__ == '__main__':
    main()
