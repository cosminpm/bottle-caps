import cv2


# Detect sift keypoints and descriptors for an img of a bottle cap
def detect_caps(img: str):
    img1 = cv2.imread(img)
    sift = cv2.SIFT_create()
    kp_1, d_1 = sift.detectAndCompute(img1, None)
    return kp_1, d_1


def draw_kp(img_path: str, kp_1: tuple):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_kp = cv2.drawKeypoints(gray, kp_1, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image', img_kp)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_url = './caps_imgs/t_cap_blue.jpg'
    kp, _ = detect_caps(img_url)
    draw_kp(img_url, kp)
