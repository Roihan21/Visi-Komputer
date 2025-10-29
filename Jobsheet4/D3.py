import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector


L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133

# Fungsi menghitung jarak Euclidean antara dua titik
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# === Inisialisasi kamera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# === Inisialisasi FaceMeshDetector ===
detector = FaceMeshDetector(
    staticMode=False,      # deteksi dilakukan setiap frame
    maxFaces=2,            # maksimal 2 wajah
    minDetectionCon=0.5,   # kepercayaan minimal untuk deteksi wajah
    minTrackCon=0.5        # kepercayaan minimal untuk pelacakan
)

# === Variabel untuk menghitung kedipan ===
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3   # jumlah frame berturut-turut dianggap satu kedipan
EYE_AR_THRESHOLD = 0.20       # ambang batas EAR (Eye Aspect Ratio)
is_closed = False

# === Loop utama ===
while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi wajah dan landmark (468 titik wajah)
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]  # hanya ambil wajah pertama dari list

        # Hitung jarak vertikal dan horizontal pada mata kiri
        v = dist(face[L_TOP], face[L_BOTTOM])    # jarak vertikal (kelopak atas–bawah)
        h = dist(face[L_LEFT], face[L_RIGHT])    # jarak horizontal (lebar mata)

        # Hitung rasio EAR (Eye Aspect Ratio)
        ear = v / (h + 1e-8)

        # Tampilkan nilai EAR di layar
        cv2.putText(img, f"EAR(L): {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Logika deteksi kedipan:
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        # Tampilkan jumlah kedipan pada frame
        cv2.putText(img, f"Blink: {blink_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame hasil deteksi
    cv2.imshow("FaceMesh + EAR", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Akhiri ===
cap.release()
cv2.destroyAllWindows()
