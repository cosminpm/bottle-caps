import cv2


def detect_caps():
    img1 = cv2.imread('./img/3YN39ZT2.jpg')
    sift = cv2.SIFT_create()
    kp_1, d_1 = sift.detectAndCompute(img1, None)
    print([kp_1, d_1])
    return [kp_1, d_1]


if __name__ == '__main__':
    a = [2, 4, 6, 8, 10]
    print(a[1:])
