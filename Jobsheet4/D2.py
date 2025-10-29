import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

# === Inisialisasi Kamera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2 dsb.")

# === Inisialisasi Pose Detector ===
detector = PoseDetector(
    staticMode=False,
    modelComplexity=1,
    enableSegmentation=False,
    detectionCon=0.5,
    trackCon=0.5
)

# === Loop utama ===
while True:
    success, img = cap.read()
    if not success:
        break

    # Temukan pose tubuh pada frame
    img = detector.findPose(img)

    # Temukan landmark dan bounding box
    lmList, bboxInfo = detector.findPosition(
        img,
        draw=True,
        bboxWithHands=False
    )

    if lmList:
        # Ambil pusat tubuh
        center = bboxInfo["center"]

        # Gambar lingkaran di pusat tubuh
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Hitung jarak antara landmark bahu kiri (11) dan pergelangan tangan kiri (15)
        length, img, info = detector.findDistance(
            lmList[11][0:2],
            lmList[15][0:2],
            img=img,
            color=(255, 0, 0),
            scale=10
        )

        # Hitung sudut antara bahu, siku, dan pergelangan tangan kiri (11, 13, 15)
        angle, img = detector.findAngle(
            lmList[11][0:2],
            lmList[13][0:2],
            lmList[15][0:2],
            img=img,
            color=(0, 0, 255),
            scale=10
        )

        # Periksa apakah sudut mendekati 50° ± 10
        isCloseAngle50 = detector.angleCheck(
            myAngle=angle,
            targetAngle=50,
            offset=10
        )

        # Cetak hasil
        print("Sudut mendekati 50°:", isCloseAngle50)

    # Tampilkan hasil di jendela
    cv2.imshow("Pose + Angle", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Akhiri ===
cap.release()
cv2.destroyAllWindows()
