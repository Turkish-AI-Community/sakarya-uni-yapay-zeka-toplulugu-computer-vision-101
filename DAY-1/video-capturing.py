import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)


# height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
# width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
# count = capture.get(cv.CAP_PROP_FRAME_COUNT)
# fps = capture.get(cv.CAP_PROP_FPS)
# print(height, width, count, fps)


cv.namedWindow("Output")

def nothing(x):
    pass

cv.createTrackbar("Mode", "Output", 0, 2, nothing)


def process(image, mode):

    if mode == 0:
        return image

    elif mode == 1:
        return cv.bitwise_not(image)

    elif mode == 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 100, 200)

        # Gri → BGR (yan yana koymak için)
        return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)


while True:

    ret, frame = capture.read()

    if not ret: # eğer frame okumadıysan döngüden çık
        break

    mode = cv.getTrackbarPos("Mode", "Output")

    result = process(frame, mode)

    # Yan yana birleştir
    combined = np.hstack((frame, result))

    cv.imshow("Output", combined)

    if cv.waitKey(1) == 27:
        break


capture.release()
cv.destroyAllWindows()
