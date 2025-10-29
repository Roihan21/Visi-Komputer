import cv2, time

# === Inisialisasi kamera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2 dsb.")

# Variabel untuk menghitung FPS (Frame Per Second)
frames, t0 = 0, time.time()

# === Loop utama untuk menampilkan video ===
while True:
    ok, frame = cap.read()  # Ambil satu frame dari kamera
    if not ok:
        break

    frames += 1
    # Hitung FPS tiap 1 detik
    if time.time() - t0 >= 1.0:
        cv2.setWindowTitle("Preview", f"Preview (FPS ~ {frames})")
        frames, t0 = 0, time.time()

    # Tampilkan hasil frame di jendela
    cv2.imshow("Preview", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Tutup kamera dan jendela ===
cap.release()
cv2.destroyAllWindows()
