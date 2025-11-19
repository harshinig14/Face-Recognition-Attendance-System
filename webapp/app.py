from flask import Flask, render_template, send_file, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import threading
import webbrowser
import time
from collections import defaultdict, Counter

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent  # COE/attendance/
DATA_DIR = BASE_DIR / 'python-module' / 'data'
WEB_APP_DIR = Path(__file__).parent

# Ensure python-module/webapp is importable
sys.path.append(str(WEB_APP_DIR))

try:
    from simple_camera_service import SimpleCameraService
except Exception as e:
    print(f"Could not import SimpleCameraService: {e}")
    raise

app = Flask(__name__)
app.secret_key = 'your-secret-key-12345'
CORS(app)

# Import the improved camera service (Part 1)
camera_service = SimpleCameraService()

STUDENTS_FILE = DATA_DIR / 'students.json'
ATTENDANCE_FILE = DATA_DIR / 'attendance.json'
LOG_FILE = DATA_DIR / 'logs' / 'recognition_log.json'

def load_students():
    """Load students from JSON (resilient to formats)"""
    try:
        if STUDENTS_FILE.exists():
            with open(STUDENTS_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return []
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data] if data else []
        return []
    except Exception as e:
        print(f"Error loading students.json: {e}")
        return []

def load_attendance_full():
    """Return the attendance data structure (dict)"""
    try:
        if ATTENDANCE_FILE.exists():
            with open(ATTENDANCE_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        return {}
    except Exception as e:
        print(f"Error reading attendance.json: {e}")
        return {}

@app.route('/')
def index():
    students = load_students()
    attendance = load_attendance_full()
    today = datetime.now().strftime('%Y-%m-%d')
    todays_attendance = attendance.get(today, {})
    
    # ✅ FIX: Convert dict to list for template with proper field names
    attendance_list = []
    for student_id, record in todays_attendance.items():
        # ✅ FIXED: Use 'in_time' and 'out_time' (with underscores)
        in_time = record.get('in_time') or record.get('intime') or '-'
        out_time = record.get('out_time') or record.get('outtime') or '-'
        
        attendance_list.append({
            'id': student_id,
            'name': record.get('name', 'Unknown'),
            'intime': in_time,  # Display key
            'outtime': out_time,  # Display key
            'status': record.get('status', 'ABSENT')
        })
    
    # ✅ FIX: Calculate present and absent counts correctly
    present_count = sum(1 for r in todays_attendance.values() 
                       if r.get('status') in ['PRESENT', 'HALF DAY'])
    absent_count = sum(1 for r in todays_attendance.values() 
                      if r.get('status') == 'ABSENT')
    
    return render_template(
        'dashboard.html',
        teacher_username='Teacher',
        today=today,
        total_students=len(students),
        present_count=present_count,  # ✅ FIXED
        absent_count=absent_count,    # ✅ NEW
        students_list=students,
        attendance_records=attendance_list  # ✅ Pass as list
    )

@app.route('/dashboard')
def dashboard():
    return index()

@app.route('/api/start_camera')
def start_camera():
    success = camera_service.start_camera()
    return jsonify({'success': success})

@app.route('/api/stop_camera')
def stop_camera():
    camera_service.stop_camera()
    return jsonify({'success': True})

@app.route('/api/current_recognition')
def current_recognition():
    status = camera_service.get_status()
    return jsonify(status)

@app.route('/api/students_list')
def students_list():
    students = load_students()
    return jsonify({'students': students, 'total': len(students)})

@app.route('/api/todays_attendance')
def todays_attendance():
    attendance = load_attendance_full()
    today = datetime.now().strftime('%Y-%m-%d')
    todays_records = attendance.get(today, {})
    
    # ✅ FIX: Convert dict to list for JavaScript with proper field names
    attendance_list = []
    for student_id, record in todays_records.items():
        # ✅ FIXED: Use 'in_time' and 'out_time' (with underscores)
        in_time = record.get('in_time') or record.get('intime') or '-'
        out_time = record.get('out_time') or record.get('outtime') or '-'
        
        attendance_list.append({
            'id': student_id,
            'name': record.get('name', 'Unknown'),
            'intime': in_time,  # Display key
            'outtime': out_time,  # Display key
            'status': record.get('status', 'ABSENT')
        })
    
    # ✅ FIX: Calculate present and absent counts correctly
    present_count = sum(1 for r in todays_records.values() 
                       if r.get('status') in ['PRESENT', 'HALF DAY'])
    absent_count = sum(1 for r in todays_records.values() 
                      if r.get('status') == 'ABSENT')
    
    return jsonify({
        'date': today,
        'attendance': attendance_list,  # ✅ Return as list
        'present_count': present_count,  # ✅ FIXED
        'absent_count': absent_count      # ✅ NEW
    })

# ✅ NEW: Get attendance for a specific date
@app.route('/api/attendance_by_date')
def attendance_by_date():
    date = request.args.get('date')
    if not date:
        return jsonify({'error': 'Date parameter required'}), 400
    
    attendance = load_attendance_full()
    day_records = attendance.get(date, {})
    
    attendance_list = []
    for student_id, record in day_records.items():
        in_time = record.get('in_time') or record.get('intime') or '-'
        out_time = record.get('out_time') or record.get('outtime') or '-'
        
        attendance_list.append({
            'id': student_id,
            'name': record.get('name', 'Unknown'),
            'intime': in_time,
            'outtime': out_time,
            'status': record.get('status', 'ABSENT')
        })
    
    # ✅ FIX: Calculate present and absent counts correctly
    present_count = sum(1 for r in day_records.values() 
                       if r.get('status') in ['PRESENT', 'HALF DAY'])
    absent_count = sum(1 for r in day_records.values() 
                      if r.get('status') == 'ABSENT')
    
    return jsonify({
        'date': date,
        'attendance': attendance_list,
        'present_count': present_count,  # ✅ FIXED
        'absent_count': absent_count      # ✅ NEW
    })

# ✅ NEW: Get available dates with attendance
@app.route('/api/attendance_dates')
def attendance_dates():
    attendance = load_attendance_full()
    dates = sorted(attendance.keys(), reverse=True)
    return jsonify({'dates': dates})

# ✅ NEW: Weekly report
@app.route('/api/weekly_report')
def weekly_report():
    attendance = load_attendance_full()
    students = load_students()
    total_students = len(students)
    
    # Get last 7 days
    today = datetime.now()
    week_data = []
    
    for i in range(6, -1, -1):
        date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        day_records = attendance.get(date, {})
        
        present = sum(1 for r in day_records.values() if r.get('status') in ['PRESENT', 'HALF DAY'])
        absent = sum(1 for r in day_records.values() if r.get('status') == 'ABSENT')
        
        week_data.append({
            'date': date,
            'present': present,
            'absent': absent,
            'total': total_students
        })
    
    return jsonify({'weekly_data': week_data})

# ✅ NEW: Monthly report
@app.route('/api/monthly_report')
def monthly_report():
    attendance = load_attendance_full()
    students = load_students()
    total_students = len(students)
    
    # Get current month
    today = datetime.now()
    month_start = today.replace(day=1)
    days_in_month = (month_start.replace(month=month_start.month % 12 + 1, day=1) - timedelta(days=1)).day if month_start.month < 12 else 31
    
    month_data = []
    status_summary = Counter()
    
    for day in range(1, min(today.day + 1, days_in_month + 1)):
        date = today.replace(day=day).strftime('%Y-%m-%d')
        day_records = attendance.get(date, {})
        
        present = sum(1 for r in day_records.values() if r.get('status') == 'PRESENT')
        half_day = sum(1 for r in day_records.values() if r.get('status') == 'HALF DAY')
        absent = sum(1 for r in day_records.values() if r.get('status') == 'ABSENT')
        
        status_summary['PRESENT'] += present
        status_summary['HALF DAY'] += half_day
        status_summary['ABSENT'] += absent
        
        month_data.append({
            'date': date,
            'present': present,
            'half_day': half_day,
            'absent': absent,
            'total': total_students
        })
    
    return jsonify({
        'monthly_data': month_data,
        'summary': dict(status_summary)
    })

# ✅ NEW: Student-wise attendance report
@app.route('/api/student_report/<student_id>')
def student_report(student_id):
    attendance = load_attendance_full()
    students = load_students()
    
    # Find student name
    student_name = next((s['name'] for s in students if str(s['id']) == str(student_id)), 'Unknown')
    
    student_attendance = []
    status_count = Counter()
    
    for date, day_records in sorted(attendance.items()):
        if student_id in day_records:
            record = day_records[student_id]
            status = record.get('status', 'ABSENT')
            status_count[status] += 1
            
            student_attendance.append({
                'date': date,
                'in_time': record.get('in_time', '-'),
                'out_time': record.get('out_time', '-'),
                'status': status
            })
    
    total_days = len(attendance)
    present_days = status_count['PRESENT'] + status_count.get('HALF DAY', 0)
    attendance_percentage = (present_days / total_days * 100) if total_days > 0 else 0
    
    return jsonify({
        'student_id': student_id,
        'student_name': student_name,
        'attendance_history': student_attendance,
        'summary': dict(status_count),
        'attendance_percentage': round(attendance_percentage, 2),
        'total_days': total_days
    })

@app.route('/api/export_attendance')
def export_attendance():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    filepath = camera_service.export_attendance_excel(date)
    
    if filepath and Path(filepath).exists():
        return send_file(
            filepath,
            as_attachment=True,
            download_name=f'attendance_{date}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    return jsonify({'error': 'No attendance data for this date'}), 404

@app.route('/api/recognition_logs')
def recognition_logs():
    """Return recognition logs JSON. Read-only endpoint for debugging/audit."""
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return jsonify([])
                data = json.loads(content)
                return jsonify(data)
        else:
            return jsonify([])
    except Exception as e:
        print(f"Error reading log file: {e}")
        return jsonify([]), 500

@app.route('/api/status')
def status():
    """Optional endpoint to stream minimal status (keeps dashboard responsive)"""
    try:
        status = camera_service.get_status()
        students = load_students()
        attendance = load_attendance_full()
        today = datetime.now().strftime('%Y-%m-%d')
        todays_records = attendance.get(today, {})
        
        # ✅ FIX: Calculate present and absent counts correctly
        present_count = sum(1 for r in todays_records.values() 
                           if r.get('status') in ['PRESENT', 'HALF DAY'])
        absent_count = sum(1 for r in todays_records.values() 
                          if r.get('status') == 'ABSENT')
        
        status.update({
            'total_students': len(students),
            'present_today': present_count,   # ✅ FIXED
            'absent_today': absent_count       # ✅ NEW
        })
        return jsonify(status)
    except Exception as e:
        print(f"Status error: {e}")
        return jsonify({}), 500

if __name__ == '__main__':
    def open_browser():
        time.sleep(1)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            pass
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("=" * 60)
    print("🎓 FACE RECOGNITION ATTENDANCE - STARTING")
    print(f"📁 DATA_DIR: {DATA_DIR}")
    print(f"📋 Students file: {STUDENTS_FILE.exists()}")
    print(f"📊 Attendance file: {ATTENDANCE_FILE.exists()}")
    print("=" * 60)
    
    # auto-open browser + run
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
