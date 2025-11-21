import cv2
import os
import json
import subprocess
import sys
from pathlib import Path

"""
simple_capture_train.py
Interactive menu-driven face recognition training system.

Features:
- Capture student faces (5 photos with histogram equalization)
- Train LBPH face recognizer
- List registered students
- Launch Flask webapp
- Original capture logic preserved
"""

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'python-module' / 'data'
PHOTOS_DIR = DATA_DIR / 'student_photos'
STUDENTS_FILE = DATA_DIR / 'students.json'
CLASSIFIER_FILE = DATA_DIR / 'classifier.xml'
WEBAPP_DIR = BASE_DIR / 'webapp'

def ensure_dir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except:
        pass

def load_students():
    """Load students from JSON file"""
    if STUDENTS_FILE.exists():
        with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_students(students_data):
    """Save students to JSON file"""
    ensure_dir(DATA_DIR)
    with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(students_data, f, indent=2, ensure_ascii=False)

def capture_photos(student_id, student_name):
    """Original capture logic - PRESERVED"""
    student_id = str(student_id)
    name_clean = "".join(c for c in student_name if c.isalnum() or c == "_")
    student_folder = PHOTOS_DIR / f"{student_id}_{name_clean}"
    ensure_dir(student_folder)
    
    print("📸 Starting camera… Press SPACE to capture. ESC to exit.")
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Cannot open camera")
        return False
    
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
    return True

def capture_student_faces():
    """Menu option 1: Capture student faces"""
    print("\n" + "=" * 70)
    print("  CAPTURE STUDENT FACES")
    print("=" * 70)
    
    student_id = input("Enter Student Roll Number: ").strip()
    student_name = input("Enter Student Name: ").strip()
    
    if not student_id or not student_name:
        print("❌ Invalid input!")
        return
    
    # Capture photos
    success = capture_photos(student_id, student_name)
    
    if success:
        # Save to students.json
        students = load_students()
        students[student_id] = {
            "name": student_name,
            "roll_no": student_id
        }
        save_students(students)
        print(f"✅ Student {student_name} ({student_id}) registered successfully!")

def train_model():
    """Menu option 2: Train face recognizer"""
    print("\n" + "=" * 70)
    print("  TRAINING FACE RECOGNIZER")
    print("=" * 70)
    
    if not PHOTOS_DIR.exists():
        print("❌ No student photos found!")
        return
    
    face_samples = []
    ids = []
    
    print("📂 Loading student photos...")
    
    for student_folder in PHOTOS_DIR.iterdir():
        if not student_folder.is_dir():
            continue
        
        # Extract student ID from folder name
        folder_name = student_folder.name
        try:
            student_id = int(folder_name.split('_')[0])
        except:
            print(f"⚠️ Skipping invalid folder: {folder_name}")
            continue
        
        # Load all images from student folder
        for img_path in student_folder.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                face_samples.append(img)
                ids.append(student_id)
    
    if len(face_samples) == 0:
        print("❌ No training data found!")
        return
    
    print(f"📊 Found {len(face_samples)} face samples from {len(set(ids))} students")
    print("🤖 Training LBPH Face Recognizer...")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,
        neighbors=16,
        grid_x=8,
        grid_y=8
    )
    
    recognizer.train(face_samples, np.array(ids))
    
    ensure_dir(DATA_DIR)
    recognizer.save(str(CLASSIFIER_FILE))
    
    print(f"✅ Model trained successfully!")
    print(f"📁 Saved to: {CLASSIFIER_FILE}")

def list_students():
    """Menu option 3: List registered students"""
    print("\n" + "=" * 70)
    print("  REGISTERED STUDENTS")
    print("=" * 70)
    
    students = load_students()
    
    if not students:
        print("❌ No students registered yet!")
        return
    
    print(f"\n{'Roll No':<15} {'Name':<30}")
    print("-" * 45)
    
    for roll_no, info in sorted(students.items()):
        name = info.get('name', 'Unknown')
        print(f"{roll_no:<15} {name:<30}")
    
    print(f"\n📊 Total Students: {len(students)}")

def launch_webapp():
    """Menu option 4: Launch Flask webapp"""
    print("\n" + "=" * 70)
    print("  LAUNCHING ATTENDANCE WEBAPP")
    print("=" * 70)
    
    app_file = WEBAPP_DIR / 'app.py'
    
    if not app_file.exists():
        print(f"❌ app.py not found at: {app_file}")
        return
    
    print("🚀 Starting Flask server...")
    print("🌐 Webapp will open at http://localhost:5000")
    print("⏹️ Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, str(app_file)], cwd=str(WEBAPP_DIR))
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped.")

def main_menu():
    """Interactive menu"""
    while True:
        print("\n" + "=" * 70)
        print("  ATTENDANCE SYSTEM - FACE RECOGNITION")
        print("=" * 70)
        print("1. Capture Student Faces (5 photos)")
        print("2. Train Face Recognizer")
        print("3. List Registered Students")
        print("4. Launch Attendance WebApp")
        print("5. Exit")
        print("=" * 70)
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            capture_student_faces()
        elif choice == '2':
            train_model()
        elif choice == '3':
            list_students()
        elif choice == '4':
            launch_webapp()
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main_menu()
