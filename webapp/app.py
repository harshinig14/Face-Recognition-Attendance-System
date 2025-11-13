from flask import Flask, render_template, send_file, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path
from datetime import datetime
import json

# CRITICAL FIX: Set paths correctly
BASE_DIR = Path(__file__).parent.parent  # COE/attendance/
DATA_DIR = BASE_DIR / 'python-module' / 'data'

sys.path.append(str(BASE_DIR / 'python-module' / 'webapp'))

from simple_camera_service import SimpleCameraService

app = Flask(__name__)
app.secret_key = 'your-secret-key-12345'
CORS(app)

camera_service = SimpleCameraService()

STUDENTS_FILE = DATA_DIR / 'students.json'
ATTENDANCE_FILE = DATA_DIR / 'attendance.json'

print(f"\n🔍 APP.PY DEBUG:")
print(f"   DATA_DIR: {DATA_DIR}")
print(f"   STUDENTS_FILE: {STUDENTS_FILE}")
print(f"   File exists: {STUDENTS_FILE.exists()}")

def load_students():
    """Load students from JSON"""
    try:
        if STUDENTS_FILE.exists():
            with open(STUDENTS_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    print("⚠️ students.json is EMPTY")
                    return []
                
                data = json.loads(content)
                
                # Handle both list and dict formats
                if isinstance(data, list):
                    students = data
                elif isinstance(data, dict):
                    students = [data] if data else []
                else:
                    students = []
                
                print(f"✅ Loaded {len(students)} students: {students}")
                return students
        else:
            print(f"❌ students.json NOT FOUND at {STUDENTS_FILE}")
            return []
    except Exception as e:
        print(f"❌ Error loading students: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_attendance():
    """Load attendance records"""
    try:
        if ATTENDANCE_FILE.exists():
            with open(ATTENDANCE_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"❌ Error loading attendance: {e}")
        return {}

@app.route('/')
def index():
    """Dashboard with students and attendance"""
    students = load_students()
    attendance = load_attendance()
    today = datetime.now().strftime('%Y-%m-%d')
    todays_attendance = attendance.get(today, [])
    
    print(f"\n📊 Dashboard Data:")
    print(f"   Total students: {len(students)}")
    print(f"   Present today: {len(todays_attendance)}")
    
    return render_template('dashboard.html',
                         teacher_username="Teacher",
                         today=today,
                         total_students=len(students),
                         present_count=len(todays_attendance),
                         students_list=students,
                         attendance_records=todays_attendance)

@app.route('/dashboard')
def dashboard():
    return index()

@app.route('/api/start_camera')
def start_camera():
    """Start camera for face recognition"""
    success = camera_service.start_camera()
    return jsonify({'success': success})

@app.route('/api/stop_camera')
def stop_camera():
    """Stop camera"""
    camera_service.stop_camera()
    return jsonify({'success': True})

@app.route('/api/current_recognition')
def current_recognition():
    """Get current recognition status"""
    status = camera_service.get_status()
    return jsonify(status)

@app.route('/api/students_list')
def students_list():
    """Get list of all students"""
    students = load_students()
    print(f"📊 API /api/students_list: Returning {len(students)} students")
    return jsonify({
        'students': students,
        'total': len(students)
    })

@app.route('/api/todays_attendance')
def todays_attendance():
    """Get today's attendance - REFRESHES LIVE"""
    attendance = load_attendance()
    today = datetime.now().strftime('%Y-%m-%d')
    todays_records = attendance.get(today, [])
    
    print(f"📊 API /api/todays_attendance: {len(todays_records)} present")
    
    return jsonify({
        'date': today,
        'attendance': todays_records,
        'present_count': len(todays_records)
    })

@app.route('/api/export_attendance')
def export_attendance():
    """Export attendance to Excel - BEAUTIFUL FORMAT"""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    print(f"📊 Exporting attendance for {date}")
    
    filepath = camera_service.export_attendance_excel(date)
    
    if filepath and Path(filepath).exists():
        print(f"✅ Sending file: {filepath}")
        return send_file(filepath, 
                        as_attachment=True,
                        download_name=f'attendance_{date}.xlsx',
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    return jsonify({'error': 'No attendance data for this date'}), 404

if __name__ == '__main__':
    import webbrowser
    import threading
    
    def open_browser():
        webbrowser.open('http://localhost:5000')
    
    students = load_students()
    
    print("=" * 70)
    print("🚀 FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 70)
    print(f"🌐 Server: http://localhost:5000")
    print(f"📁 DATA_DIR: {DATA_DIR}")
    print(f"📊 Total Students: {len(students)}")
    print("=" * 70)
    
    threading.Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
