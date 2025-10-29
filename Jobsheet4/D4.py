import cv2
from cvzone.HandTrackingModule import HandDetector

# === Inisialisasi Kamera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# === Inisialisasi HandDetector ===
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

# === Loop utama ===
while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi tangan
    hands, img = detector.findHands(img, draw=True, flipType=True)  # flipType=True supaya mirror
    if hands:
        hand = hands[0]  # ambil data tangan pertama
        fingers = detector.fingersUp(hand)  # daftar 5 angka (0=turun, 1=naik)
        count = sum(fingers)  # jumlah jari yang terangkat

        # Tampilkan hasil di layar
        cv2.putText(img, f"Fingers: {count} {fingers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan jendela
    cv2.imshow("Hands + Fingers", img)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
