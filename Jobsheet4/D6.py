import cv2
import numpy as np
from collections import deque
from cvzone.PoseModule import PoseDetector

# ==================== KONFIGURASI ====================
MODE = "squat"  # tekan 'm' untuk ganti ke "pushup"
KNEE_DOWN, KNEE_UP = 80, 160  # ambang squat (sudut lutut)
DOWN_R, UP_R = 0.85, 1.00  # ambang push-up (rasio)
SAMPLE_OK = 4  # jumlah frame konsisten sebelum hitung gerakan

# ==================== INISIALISASI KAMERA ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi detektor pose
detector = PoseDetector(
    staticMode=False,
    modelComplexity=1,
    enableSegmentation=False,
    detectionCon=0.5,
    trackCon=0.5
)

# ==================== VARIABEL COUNTER ====================
count, state = 0, "up"
debounce = deque(maxlen=6)  # menyimpan status 'up/down' beberapa frame terakhir

# ==================== FUNGSI UNTUK PUSH-UP ====================
def ratio_pushup(lm):
    # gunakan sisi kiri tubuh: 11=shoulderL, 15=wristL, 23=hipL
    sh = np.array(lm[11][1:3])
    wr = np.array(lm[15][1:3])
    hp = np.array(lm[23][1:3])
    return np.linalg.norm(sh - wr) / (np.linalg.norm(sh - hp) + 1e-8)

# ==================== LOOP UTAMA ====================
while True:
    ok, img = cap.read()
    if not ok:
        break

    img = detector.findPose(img, draw=True)
    lmList, _ = detector.findPosition(img, draw=False)

    flag = None  # status sementara: "up" atau "down"

    if lmList:
        # ---------- MODE SQUAT ----------
        if MODE == "squat":
            # hitung sudut lutut kiri (23,25,27) dan kanan (24,26,28)
            angL, img = detector.findAngle(lmList[23][0:2], lmList[25][0:2], lmList[27][0:2],
                                           img=img, color=(0, 0, 255), scale=10)
            angR, img = detector.findAngle(lmList[24][0:2], lmList[26][0:2], lmList[28][0:2],
                                           img=img, color=(0, 255, 0), scale=10)
            ang = (angL + angR) / 2.0  # rata-rata dua lutut
            if ang < KNEE_DOWN:
                flag = "down"
            elif ang > KNEE_UP:
                flag = "up"
            cv2.putText(img, f"Knee: {ang:5.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ---------- MODE PUSH-UP ----------
        else:
            r = ratio_pushup(lmList)
            if r < DOWN_R:
                flag = "down"
            elif r > UP_R:
                flag = "up"
            cv2.putText(img, f"Ratio: {r:4.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ---------- SISTEM DEBOUNCE (ANTI GANDA) ----------
        debounce.append(flag)
        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1  # gerakan berhasil dihitung

    # ---------- TAMPILKAN INFORMASI ----------
    cv2.putText(img, f"Mode: {MODE.upper()}  Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"State: {state}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ---------- TAMPILKAN FRAME ----------
    cv2.imshow("Pose Counter", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        MODE = "pushup" if MODE == "squat" else "squat"

# ==================== SELESAI ====================
cap.release()
cv2.destroyAllWindows()
