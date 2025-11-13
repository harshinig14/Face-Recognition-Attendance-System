"""
Simple Face Capture and Training System - WITH WEBAPP OPTION
Captures 5 photos per student and trains LBPH recognizer
"""

import cv2
import os
import json
from PIL import Image
import numpy as np
from pathlib import Path
import webbrowser
import subprocess
import sys

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "student_photos"
STUDENTS_FILE = BASE_DIR / "data" / "students.json"
CLASSIFIER_FILE = BASE_DIR / "data" / "classifier.xml"
ATTENDANCE_FILE = BASE_DIR / "data" / "attendance.json"
NUM_PHOTOS = 5

def load_students():
    """Load existing students from JSON"""
    if STUDENTS_FILE.exists():
        try:
            with open(STUDENTS_FILE, 'r') as f:
                content = f.read().strip()
                if not content or content == "":
                    return []
                data = json.loads(content)
                # FIX: Handle both list and dict formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data] if data else []
                else:
                    return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Error loading students.json: {e}")
            print("Creating new students list...")
            return []
    return []

def save_students(students):
    """Save students to JSON"""
    with open(STUDENTS_FILE, 'w') as f:
        json.dump(students, f, indent=2)

def get_next_student_id():
    """Get next available student ID"""
    students = load_students()
    if not students:
        return 1
    return max([s['id'] for s in students]) + 1

def capture_student_faces():
    """Capture 5 face photos for a student"""
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    def detect_face(img):
        """Detect and return the largest face"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]
        return img[y:y+h, x:x+w]
    
    print("=" * 70)
    print("  FACE CAPTURE - ATTENDANCE SYSTEM")
    print("=" * 70)
    
    # Get student info
    student_name = input("\nEnter Student Name: ").strip()
    register_number = input("Enter Register Number: ").strip()
    
    if not student_name or not register_number:
        print("❌ Name and Register Number required!")
        return
    
    # Check if student already exists
    students = load_students()
    for student in students:
        if student.get('register_number') == register_number:
            print(f"❌ Student with register number {register_number} already exists!")
            return
    
    student_id = get_next_student_id()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return
    
    print(f"\n📸 Capturing {NUM_PHOTOS} photos for {student_name}")
    print("\nInstructions:")
    print("  - Look directly at camera")
    print("  - Ensure good lighting")
    print("  - Press SPACE to capture")
    print("  - Press ESC to cancel\n")
    
    img_count = 0
    
    while img_count < NUM_PHOTOS:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror
        
        # Display info
        text = f"Photo {img_count + 1}/{NUM_PHOTOS} - Press SPACE"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        cv2.putText(frame, student_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        detected = detect_face(frame)
        if detected is not None:
            cv2.putText(frame, "Face Detected!", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Capture Faces - Attendance System", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n❌ Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == 32:  # SPACE
            if detected is not None:
                img_count += 1
                face = cv2.resize(detected, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
                filename = DATA_DIR / f"user.{student_id}.{img_count}.jpg"
                cv2.imwrite(str(filename), face)
                print(f"✅ Photo {img_count} captured!")
                
                import time
                time.sleep(0.3)
            else:
                print("⚠️ No face detected! Try again.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save student info
    students.append({
        'id': student_id,
        'name': student_name,
        'register_number': register_number,
        'photos_captured': img_count
    })
    save_students(students)
    
    print(f"\n✅ Successfully captured {img_count} photos for {student_name}!")
    print(f"📁 Student ID: {student_id}")
    print("\n📌 Next: Run Option 2 to train the model")


def train_model():
    """Train LBPH face recognizer"""
    
    print("=" * 70)
    print("  TRAIN FACE RECOGNIZER")
    print("=" * 70)
    
    if not DATA_DIR.exists():
        print(f"\n❌ '{DATA_DIR}' not found!")
        return
    
    image_files = list(DATA_DIR.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"\n❌ No training images found!")
        print("Run Option 1 to capture faces first!")
        return
    
    print(f"\n📂 Found {len(image_files)} images")
    
    faces = []
    ids = []
    
    print("⏳ Processing images...")
    
    for image_file in image_files:
        try:
            pil_image = Image.open(image_file).convert('L')
            image_np = np.array(pil_image, 'uint8')
            
            # Extract student ID from filename (user.1.5.jpg -> ID=1)
            student_id = int(image_file.stem.split(".")[1])
            
            faces.append(image_np)
            ids.append(student_id)
        except Exception as e:
            print(f"⚠️ Error: {image_file.name} - {str(e)}")
    
    if len(faces) == 0:
        print("\n❌ No valid face images to train!")
        return
    
    ids = np.array(ids)
    
    print(f"✅ Processed {len(faces)} images")
    print(f"✅ Student IDs: {sorted(set(ids))}")
    
    print("\n🤖 Training LBPH recognizer...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)
    recognizer.write(str(CLASSIFIER_FILE))
    
    print(f"\n✅ Training complete!")
    print(f"📦 Model saved: {CLASSIFIER_FILE}")
    print("\n📌 Now run Option 4 to take attendance via webapp!")


def list_students():
    """List all registered students"""
    students = load_students()
    
    if not students:
        print("\n📋 No students registered yet.")
        return
    
    print("\n" + "=" * 70)
    print("  REGISTERED STUDENTS")
    print("=" * 70)
    
    for student in students:
        print(f"\nID: {student.get('id', 'N/A')}")
        print(f"Name: {student.get('name', 'N/A')}")
        print(f"Register #: {student.get('register_number', 'N/A')}")
        print(f"Photos: {student.get('photos_captured', 0)}")
        print("-" * 70)


def launch_webapp():
    """Launch the Flask webapp for attendance"""
    
    print("\n" + "=" * 70)
    print("  LAUNCHING ATTENDANCE WEB APP")
    print("=" * 70)
    
    # Check if model exists
    if not CLASSIFIER_FILE.exists():
        print("\n❌ Model not trained yet!")
        print("Please run Option 2 to train the model first!")
        return
    
    # Initialize attendance.json if it doesn't exist
    if not ATTENDANCE_FILE.exists():
        ATTENDANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ATTENDANCE_FILE, 'w') as f:
            json.dump({"records": []}, f, indent=2)
    
    print("\n✅ Model found!")
    print("✅ Starting web server...")
    print("\n🌐 Web app will open at: http://localhost:5000")
    print("🔑 Login credentials: admin / admin123")
    print("\n📌 Press Ctrl+C to stop the server\n")
    
    # Get webapp directory
    webapp_dir = BASE_DIR.parent / "webapp"
    app_file = webapp_dir / "app.py"
    
    if not app_file.exists():
        print(f"❌ Web app not found at {app_file}")
        return
    
    # Launch Flask app
    try:
        os.chdir(webapp_dir)
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped!")
    except Exception as e:
        print(f"\n❌ Error starting webapp: {e}")


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
