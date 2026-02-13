import cv2 as cv
import numpy as np

# Kamera
capture = cv.VideoCapture(0)

cv.namedWindow("Output")

def nothing(x):
    pass

# 0: Normal
# 1: Invert
# 2: Canny
# 3: Motion Detection
cv.createTrackbar("Mode", "Output", 0, 3, nothing)

# Background Subtractor (state tutar)
fgbg = cv.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=100
)

# Burada MOG2 background subtractor oluşturuluyor.
# Bu algoritma şunu yapar:
# Kameradan gelen görüntüleri zaman içinde izler.
# “Normal” arka planı öğrenir.
# Sonradan değişen pikselleri “hareket” olarak işaretler.
# Yani sistem kendi kendine şu modeli kurar:
# “Burası genelde sabit → background”
# “Burası değişiyor → foreground (hareket)” 
# Bu yüzden “stateful”dır, yani geçmişi tutar.

kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))


def process(image, mode):

    # Orijinal
    if mode == 0:
        return image

    # Negatif
    elif mode == 1:
        return cv.bitwise_not(image)

    # Kenar
    elif mode == 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 100, 200)

        return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # Hareketli Nesne Tespiti
    elif mode == 3:

        output = image.copy()

        # Background subtraction
        mask = fgbg.apply(image)
        # Mevcut frame’i modele verir.
        # Arka planla karşılaştırır.
        # Bir binary maske üretir.
        # Maske şu şekildedir:
        # Beyaz (255) → hareket var
        # Siyah (0) → arka plan
        # Yani artık elimizde sadece hareketli bölgeler var.

        # Gürültü temizleme
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Kontur bul
        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv.contourArea(cnt)

            # Küçük gürültüleri at
            if area < 150:
                continue

            rect = cv.minAreaRect(cnt)

            # Yeşil elips
            cv.ellipse(output, rect, (0, 255, 0), 2)

            # Merkez noktası
            cx = int(rect[0][0])
            cy = int(rect[0][1])

            cv.circle(output, (cx, cy), 3, (255, 0, 0), -1)

        return output


while True:

    ret, frame = capture.read()

    if not ret:
        break

    mode = cv.getTrackbarPos("Mode", "Output")

    result = process(frame, mode)

    # Yan yana göster
    combined = np.hstack((frame, result))

    cv.imshow("Output", combined)

    if cv.waitKey(1) == 27:  # ESC
        break


capture.release()
cv.destroyAllWindows()
