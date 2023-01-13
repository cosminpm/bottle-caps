import cv2
import numpy as np

DEBUG_HOUGH_TRANSFORM = False


def combine_overlapping_circles(circles):
    circles = np.round(circles[0, :]).astype("int")
    combined_circles = []
    for (x, y, r) in circles:
        found_overlap = False
        for (cx, cy, cr) in combined_circles:
            if (x - cx) ** 2 + (y - cy) ** 2 < (r + cr) ** 2:
                # Overlap found, compute weighted average
                w1 = r ** 2
                w2 = cr ** 2
                cx = (x * w1 + cx * w2) / (w1 + w2)
                cy = (y * w1 + cy * w2) / (w1 + w2)
                cr = (r * w1 + cr * w2) / (w1 + w2)
                found_overlap = True
                break
        if not found_overlap:
            combined_circles.append((x, y, r))
    return combined_circles


def hough_transform_circle(img, max_radius) -> (np.ndarray, int):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=18, minRadius=int(max_radius * 0.9), maxRadius=int(max_radius * 1.1))
    circles = np.uint16(np.around(circles))

    circles = combine_overlapping_circles(circles)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Draw combined circles on image
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)

    if DEBUG_HOUGH_TRANSFORM:
        cv2.imshow("Hough-Transform-Debug", img)
        cv2.waitKey(0)

    return img, len(circles)
