import cv2
import numpy as np
import os


# https://github.com/SSARCandy/panoramas-image-stitching

# -----------------------------
# Dosya Yolları
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")

images = []

for file in sorted(os.listdir(IMG_DIR)):

    if file.endswith(".jpg") or file.endswith(".png"):

        path = os.path.join(IMG_DIR, file)
        img = cv2.imread(path)

        if img is None:
            print("UYARI: Okunamadı ->", file)
        else:
            images.append(img)


if len(images) < 3:
    print("HATA: En az 3 resim gerekli")
    exit()


img1, img2, img3 = images[:3]


# -----------------------------
# Gray'e çevir
# -----------------------------
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)


# -----------------------------
# ORB
# -----------------------------
orb = cv2.ORB_create(nfeatures=3000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
kp3, des3 = orb.detectAndCompute(gray3, None)


if des1 is None or des2 is None or des3 is None:
    print("HATA: Descriptor üretilemedi")
    exit()


# -----------------------------
# Matcher
# -----------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# df binary descriptor
# Hamming distance
# crossCheck: iki yönlü eşleşme varsa tekini kabul ediyor

matches12 = bf.match(des1, des2)
matches23 = bf.match(des2, des3)

print("Match 1-2:", len(matches12))
print("Match 2-3:", len(matches23))


if len(matches12) < 20 or len(matches23) < 20:
    print("HATA: Yeterli eşleşme yok")
    exit()


matches12 = sorted(matches12, key=lambda x: x.distance)
matches23 = sorted(matches23, key=lambda x: x.distance)


# -----------------------------
# Homography
# -----------------------------
def get_homography(matches, kpA, kpB): # işin kalbi

    # Outlier’ları eler
    # En iyi perspektif matrisi bulur
    # imgA → imgB koordinatlarına map eder.

    src_pts = np.float32(
        [kpA[m.queryIdx].pt for m in matches]
    ).reshape(-1,1,2)

    dst_pts = np.float32(
        [kpB[m.trainIdx].pt for m in matches]
    ).reshape(-1,1,2)

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        cv2.RANSAC, 5.0
    )

    return H


H12 = get_homography(matches12[:100], kp1, kp2)
H23 = get_homography(matches23[:100], kp2, kp3)


if H12 is None:
    print("HATA: img1 → img2 hizalanamadı")
    exit()

if H23 is None:
    print("HATA: img2 → img3 hizalanamadı")
    exit()


print("Homography OK")


# -----------------------------
# Canvas Oluştur
# -----------------------------
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
h3, w3 = img3.shape[:2]

canvas_width = w1 + w2 + w3
canvas_height = max(h1, h2, h3)

canvas = np.zeros(
    (canvas_height, canvas_width, 3),
    dtype=np.uint8
)


# img2'yi ortaya yakın yerleştir (negatif warp kesilmesin)
offset_x = w1

# Canvas'a geçiş için çeviri matrisi
T = np.array([
    [1, 0, offset_x],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float64)



# -----------------------------
# Ortaya img2 koy
# -----------------------------
canvas[0:h2, offset_x:offset_x + w2] = img2


# -----------------------------
# img1 ekle
# -----------------------------
warp1 = cv2.warpPerspective(
    img1, T @ H12,
    (canvas_width, canvas_height)
)

mask1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2GRAY) > 0

# Sadece dolu pikselleri yaz
canvas[mask1] = warp1[mask1]


# -----------------------------
# img3 ekle
# -----------------------------
warp3 = cv2.warpPerspective(
    img3, T @ np.linalg.inv(H23),
    (canvas_width, canvas_height)
)

mask3 = cv2.cvtColor(warp3, cv2.COLOR_BGR2GRAY) > 0

empty = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) == 0
mask3_final = mask3 & empty
canvas[mask3_final] = warp3[mask3_final]


# -----------------------------
# Ekrana sığdır
# -----------------------------
max_width = 1200

h, w = canvas.shape[:2]

scale = min(1.0, max_width / w)

panorama_show = cv2.resize(
    canvas,
    None,
    fx=scale,
    fy=scale
)


# -----------------------------
# Göster
# -----------------------------
cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
cv2.resizeWindow(
    "Panorama",
    int(w*scale),
    int(h*scale)
)

cv2.imshow("Panorama", panorama_show)

cv2.waitKey(0)
cv2.destroyAllWindows()
