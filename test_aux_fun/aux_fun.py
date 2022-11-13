import cv2

# Draw kp from an image path
def draw_kp(img_path: str, kp_1: tuple):
    img = cv2.imread(img_path)
    img_kp = cv2.drawKeypoints(img, kp_1, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image', img_kp)
    cv2.waitKey(0)
