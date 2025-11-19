import cv2
import os
from pathlib import Path

"""
simple_capture_train.py

This script captures face images for training.
Improvements:
- Histogram equalization for better lighting normalization
- Slight padding on face crops
- Stable Haar detection parameters
- No workflow changes (SPACE = capture, ESC = exit)
"""

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'python-module' / 'data'
PHOTOS_DIR = DATA_DIR / 'student_photos'

def ensure_dir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except:
        pass


def capture_photos(student_id, student_name):
    student_id = str(student_id)
    name_clean = "".join(c for c in student_name if c.isalnum() or c == "_")
    student_folder = PHOTOS_DIR / f"{student_id}_{name_clean}"
    ensure_dir(student_folder)

    print("📸 Starting camera… Press SPACE to capture. ESC to exit.")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Cannot open camera")
        return

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    img_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Frame not captured, retrying…")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # better for training

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(100, 100)
        )

        # Draw detection rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capture Photos", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            print("⏹️ Capture stopped by user.")
            break

        if key == 32:  # SPACE to capture
            if len(faces) == 0:
                print("⚠️ No face detected, try again.")
                continue

            # Use the first detected face
            x, y, w, h = faces[0]

            # Optional padding for better ROI
            pad = int(0.06 * w)  
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)

            face_crop = gray[y1:y2, x1:x2]

            # Resize and equalize (improves LBPH)
            face_crop = cv2.resize(face_crop, (200, 200))
            face_crop = cv2.equalizeHist(face_crop)

            img_count += 1
            filename = student_folder / f"user.{student_id}.{img_count}.jpg"
            cv2.imwrite(str(filename), face_crop)

            print(f"✅ Photo {img_count} saved → {filename.name}")

    cam.release()
    cv2.destroyAllWindows()
    print("📁 Capture complete. Photos saved in:", student_folder)
