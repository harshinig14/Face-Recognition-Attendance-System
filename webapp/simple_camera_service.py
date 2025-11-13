import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import threading
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

class SimpleCameraService:
    def __init__(self):
        # CRITICAL FIX: Correct path to data directory
        self.BASE_DIR = Path(__file__).parent.parent  # COE/attendance/
        self.DATA_DIR = self.BASE_DIR / 'python-module' / 'data'
        self.PHOTOS_DIR = self.DATA_DIR / 'student_photos'
        
        print(f"\n📁 SimpleCameraService Paths:")
        print(f"   BASE_DIR: {self.BASE_DIR}")
        print(f"   DATA_DIR: {self.DATA_DIR}")
        print(f"   PHOTOS_DIR: {self.PHOTOS_DIR}")
        
        # Camera state
        self.camera = None
        self.is_running = False
        self.recognition_thread = None
        
        # Recognition state
        self.current_status = {
            'camera_active': False,
            'last_recognized': None,
            'last_recognition_time': None,
            'confidence': 0
        }
        
        # Attendance tracking
        self.marked_today = set()
        
        # Load model and data
        self.load_model()
        self.load_students()
        
    def load_model(self):
        """Load trained face recognition model"""
        classifier_path = self.DATA_DIR / 'classifier.xml'
        
        print(f"\n🔍 Looking for model:")
        print(f"   classifier.xml: {classifier_path}")
        print(f"   Exists: {classifier_path.exists()}")
        
        if classifier_path.exists():
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(str(classifier_path))
                print(f"✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.recognizer = None
        else:
            self.recognizer = None
            print("⚠️ NO MODEL FOUND")
        
        # Create label dictionary from students.json
        self.create_labels_from_students()
    
    def create_labels_from_students(self):
        """Create labels from students.json"""
        students_path = self.DATA_DIR / 'students.json'
        
        print(f"\n🔍 Loading labels:")
        print(f"   students.json: {students_path}")
        print(f"   Exists: {students_path.exists()}")
        
        if students_path.exists():
            try:
                with open(students_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        self.label_dict = {}
                        print("⚠️ students.json is EMPTY")
                        return
                    
                    students = json.loads(content)
                    self.label_dict = {}
                    
                    if isinstance(students, list):
                        for s in students:
                            if 'id' in s and 'name' in s:
                                sid = int(s['id'])
                                self.label_dict[sid] = s['name']
                                self.label_dict[str(sid)] = s['name']
                    elif isinstance(students, dict):
                        if 'id' in students and 'name' in students:
                            sid = int(students['id'])
                            self.label_dict[sid] = students['name']
                            self.label_dict[str(sid)] = students['name']
                    
                    print(f"✅ Labels created: {len(self.label_dict) // 2} students")
                    
            except Exception as e:
                print(f"❌ Error creating labels: {e}")
                self.label_dict = {}
        else:
            self.label_dict = {}
            print(f"❌ students.json NOT FOUND")
    
    def load_students(self):
        """Load student data"""
        students_path = self.DATA_DIR / 'students.json'
        
        if students_path.exists():
            try:
                with open(students_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        self.students = []
                        return
                    
                    data = json.loads(content)
                    self.students = data if isinstance(data, list) else [data] if data else []
                    print(f"✅ Students loaded: {len(self.students)}")
                    
            except Exception as e:
                print(f"❌ Error loading students: {e}")
                self.students = []
        else:
            self.students = []
    
    def start_camera(self):
        """Start camera and recognition"""
        if self.is_running:
            return False
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            return False
        
        self.is_running = True
        self.current_status['camera_active'] = True
        
        # Start recognition thread
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.recognition_thread.start()
        
        print("✅ Camera started")
        return True
    
    def stop_camera(self):
        """Stop camera"""
        self.is_running = False
        self.current_status['camera_active'] = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        cv2.destroyAllWindows()
    
    def _recognition_loop(self):
        """Main recognition loop WITH WINDOW"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        cv2.namedWindow('Face Recognition Attendance', cv2.WINDOW_NORMAL)
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Recognize face
                if self.recognizer and self.label_dict:
                    try:
                        label, confidence = self.recognizer.predict(face_roi)
                        confidence_score = 100 - confidence
                        
                        name = self.label_dict.get(label, self.label_dict.get(str(label), "Unknown"))
                        
                        if confidence_score > 50:
                            text = f"{name} ({confidence_score:.1f}%)"
                            color = (0, 255, 0)
                            
                            # Mark attendance
                            self.mark_attendance(name, label, confidence_score)
                            
                            # Update status
                            self.current_status['last_recognized'] = name
                            self.current_status['last_recognition_time'] = datetime.now().isoformat()
                            self.current_status['confidence'] = confidence_score
                        else:
                            text = f"Unknown ({confidence_score:.1f}%)"
                            color = (0, 0, 255)
                        
                        cv2.putText(display_frame, text, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    except Exception as e:
                        cv2.putText(display_frame, "Error", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "No Model", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Show window
            cv2.imshow('Face Recognition Attendance', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_camera()
                break
        
        cv2.destroyAllWindows()
    
    def mark_attendance(self, name, student_id, confidence):
        """Mark attendance"""
        today = datetime.now().strftime('%Y-%m-%d')
        key = f"{name}_{today}"
        
        if key in self.marked_today:
            return
        
        attendance_file = self.DATA_DIR / 'attendance.json'
        
        try:
            if attendance_file.exists():
                with open(attendance_file, 'r') as f:
                    attendance = json.load(f)
            else:
                attendance = {}
            
            if today not in attendance:
                attendance[today] = []
            
            record = {
                'name': name,
                'id': str(student_id),
                'time': datetime.now().strftime('%H:%M:%S'),
                'confidence': round(confidence, 2)
            }
            
            attendance[today].append(record)
            
            with open(attendance_file, 'w') as f:
                json.dump(attendance, f, indent=2)
            
            self.marked_today.add(key)
            print(f"✅ Attendance marked: {name}")
            
        except Exception as e:
            print(f"❌ Error marking attendance: {e}")
    
    def get_status(self):
        return self.current_status
    
    def export_attendance_excel(self, date=None):
        """Export attendance to beautiful formatted Excel file"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        attendance_file = self.DATA_DIR / 'attendance.json'
        
        try:
            if not attendance_file.exists():
                return None
            
            with open(attendance_file, 'r') as f:
                attendance = json.load(f)
            
            if date not in attendance:
                return None
            
            records = attendance[date]
            
            # Create Excel workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "Attendance"
            
            # Title
            ws.merge_cells('A1:E1')
            title_cell = ws['A1']
            title_cell.value = f"ATTENDANCE REPORT - {date}"
            title_cell.font = Font(size=16, bold=True, color="FFFFFF")
            title_cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            title_cell.alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[1].height = 30
            
            # Summary
            ws.merge_cells('A2:B2')
            ws['A2'] = "Total Students:"
            ws['A2'].font = Font(bold=True)
            ws['C2'] = len(self.students)
            
            ws.merge_cells('A3:B3')
            ws['A3'] = "Present Today:"
            ws['A3'].font = Font(bold=True)
            ws['C3'] = len(records)
            ws['C3'].fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
            
            ws.merge_cells('A4:B4')
            ws['A4'] = "Attendance %:"
            ws['A4'].font = Font(bold=True)
            attendance_percent = (len(records) / len(self.students) * 100) if len(self.students) > 0 else 0
            ws['C4'] = f"{attendance_percent:.1f}%"
            ws['C4'].fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
            
            # Empty row
            ws.append([])
            
            # Headers
            headers = ["S.No", "Name", "ID", "Time", "Confidence %"]
            header_row = 6
            for col_num, header in enumerate(headers, 1):
                cell = ws.cell(row=header_row, column=col_num)
                cell.value = header
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            
            # Data rows - FIXED: Handle missing 'id' key
            for idx, record in enumerate(records, 1):
                row = header_row + idx
                data = [
                    idx,
                    record.get('name', 'Unknown'),
                    record.get('id', 'N/A'),  # FIX: Use .get() with default
                    record.get('time', '00:00:00'),
                    f"{record.get('confidence', 0)}%"
                ]
                
                for col_num, value in enumerate(data, 1):
                    cell = ws.cell(row=row, column=col_num)
                    cell.value = value
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = Border(
                        left=Side(style='thin'),
                        right=Side(style='thin'),
                        top=Side(style='thin'),
                        bottom=Side(style='thin')
                    )
                    
                    # Alternate row colors
                    if idx % 2 == 0:
                        cell.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
            
            # Adjust column widths
            ws.column_dimensions['A'].width = 8
            ws.column_dimensions['B'].width = 25
            ws.column_dimensions['C'].width = 15
            ws.column_dimensions['D'].width = 15
            ws.column_dimensions['E'].width = 15
            
            # Save file
            export_path = self.DATA_DIR / f'attendance_{date}.xlsx'
            wb.save(export_path)
            
            print(f"✅ Excel exported: {export_path}")
            return str(export_path)
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            import traceback
            traceback.print_exc()
            return None
