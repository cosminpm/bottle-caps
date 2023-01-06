import cv2
import numpy as np

DEBUG_HOUGH_TRANSFORM = True


def hough_transform_circle(img, max_radius):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=18, minRadius=int(max_radius * 0.9), maxRadius=int(max_radius * 1.1))
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    if DEBUG_HOUGH_TRANSFORM:
        cv2.imshow("Hough-Transform-Debug", img)
        cv2.waitKey(0)

    return img
