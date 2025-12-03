
from flask_cors import CORS
import cv2, numpy as np, sqlite3, os, json
from datetime import datetime, time
import pytz
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session

app = Flask(__name__)
CORS(app)
app.secret_key = "super-secret-key-123"
# IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Global vars
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create folders and JSON files
os.makedirs('training', exist_ok=True)
os.makedirs('data', exist_ok=True)


ADMIN_EMAIL = "admin@company.com"
ADMIN_PASSWORD = "admin123"

# Initialize JSON files
def init_json_files():
    if not os.path.exists('data/students.json'):
        with open('data/students.json', 'w') as f:
            json.dump([], f)
    if not os.path.exists('data/attendance.json'):
        with open('data/attendance.json', 'w') as f:
            json.dump([], f)

init_json_files()

def load_students():
    with open('data/students.json', 'r') as f:
        return json.load(f)

def save_students(students):
    with open('data/students.json', 'w') as f:
        json.dump(students, f, indent=2)

def load_attendance():
    with open('data/attendance.json', 'r') as f:
        return json.load(f)

def save_attendance(attendance):
    with open('data/attendance.json', 'w') as f:
        json.dump(attendance, f, indent=2)
def is_logged_in():
    return session.get("logged_in") is True

@app.route('/')
def index():
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['admin_email'] = email
            return redirect(url_for('index'))
        else:
            error = "Invalid email or password"
            return render_template('login.html', error=error)

    return render_template('login.html', error=None)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/studentslist')
def students_list():
    students = load_students()
    return jsonify([{'id': s['rollno'], 'name': s['name']} for s in students])

@app.route('/api/registerstudent', methods=['POST'])
def register_student():
    try:
        data = request.json
        rollno = data.get('studentid', '').strip().upper()
        name = data.get('studentname', '').strip()

        if not rollno or not name:
            return jsonify({'success': False, 'message': 'Please enter both roll number and name!'}), 400

        students = load_students()
        if any(s['rollno'] == rollno for s in students):
            return jsonify({'success': False, 'message': f'Student {rollno} already registered'}), 400

        students.append({
            'rollno': rollno,
            'name': name,
            'registered': datetime.now(ist).isoformat()
        })
        save_students(students)

        # --- Camera capture ---
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Could not open camera'}), 500

        student_dir = f'training/{rollno}_{name.replace(" ", "_")}'
        os.makedirs(student_dir, exist_ok=True)

        print(f"Capturing 5 images for {name} ({rollno})...")
        count = 0

        while count < 5:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                cv2.imshow('Register Student - SPACE: capture, ESC: exit', frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # SPACE
                    cv2.imwrite(f'{student_dir}/{count}.jpg', face)
                    print(f"Image {count+1} captured")
                    count += 1
                elif key == 27:      # ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    return jsonify({'success': False, 'message': 'Registration cancelled'}), 400

            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({'success': False, 'message': 'Registration cancelled'}), 400

        cap.release()
        cv2.destroyAllWindows()

        return jsonify({
            'success': True,
            'message': f'Registered {name} ({rollno}). Captured {count} images. Click Train Model next.'
        })

    except Exception as e:
        print('Error in register_student:', e)
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500
@app.route('/api/trainmodel', methods=['POST'])
def train_model():
    try:
        students = load_students()
        if not students:
            return jsonify({'success': False, 'message': 'No students registered yet'}), 400
        
        faces = []
        labels = []
        label_map = {}
        label_id = 0
        
        for student in students:
            rollno = student['rollno']
            student_dir = f'training/{rollno}_{student["name"].replace(" ", "_")}'
            
            if os.path.exists(student_dir):
                label_map[label_id] = rollno
                for filename in os.listdir(student_dir):
                    if filename.endswith('.jpg'):
                        path = os.path.join(student_dir, filename)
                        face_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        faces.append(face_img)
                        labels.append(label_id)
                label_id += 1
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No training images found'}), 400
        
        recognizer.train(faces, np.array(labels))
        recognizer.save('trainer.yml')
        
        # Save label map
        with open('label_map.json', 'w') as f:
            json.dump(label_map, f)
        
        return jsonify({
            'success': True, 
            'message': f'Trained model with {len(faces)} images from {len(label_map)} students'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Training error: {str(e)}'}), 500
# ---------- ATTENDANCE HELPERS ----------

def get_today_date_str():
    return datetime.now(ist).strftime('%Y-%m-%d')

def get_now_time_str():
    return datetime.now(ist).strftime('%H:%M:%S')

def get_now_time_obj():
    return datetime.now(ist).time()

def find_student_by_roll(rollno):
    students = load_students()
    for s in students:
        if s['rollno'] == rollno:
            return s
    return None

def get_today_records():
    today = get_today_date_str()
    all_att = load_attendance()
    return [r for r in all_att if r['date'] == today]

def save_or_update_attendance(rollno, status, intime=None, outtime=None):
    all_att = load_attendance()
    today = get_today_date_str()

    # find existing record for this student today
    rec = None
    for r in all_att:
        if r['date'] == today and r['rollno'] == rollno:
            rec = r
            break

    # if no record yet, create one
    if rec is None:
        rec = {
            'date': today,
            'rollno': rollno,
            'intime': intime or get_now_time_str(),
            'outtime': outtime or '',
            'status': status
        }
        all_att.append(rec)
    else:
        # update existing
        if intime is not None:
            rec['intime'] = intime
        if outtime is not None:
            rec['outtime'] = outtime
        rec['status'] = status

    save_attendance(all_att)
    return rec
def decide_attendance_action(existing_rec):
    """
    Returns (action, status) where:
    action: 'IN', 'OUT', 'CLOSED'
    status: 'PRESENT' / 'HALF DAY' / None
    """
    now_t = get_now_time_obj()

    # Time windows
    in_start   = time(7, 45)
    in_end     = time(10, 0)
    out_open   = time(12, 0)   # OUT starts after 12:00
    half_start = time(12, 1)
    half_end   = time(14, 0)
    full_start = time(14, 50)
    full_end   = time(17, 0)
    close_time = time(17, 0)

    # 1) No record yet -> only IN allowed in window
    if existing_rec is None:
        # Before IN window, or after 10:00 -> no IN allowed
        if not (in_start <= now_t <= in_end):
            return 'CLOSED', None
        # IN allowed
        return 'IN', 'PRESENT'  # initial status; final depends on OUT

    # 2) Has IN but no OUT yet
    if existing_rec.get('intime') and not existing_rec.get('outtime'):
        # Attendance is closed after 5:00 PM
        if now_t > close_time:
            return 'CLOSED', None

        # Before or at 12:00 -> cannot mark OUT yet
        if now_t <= out_open:
            return 'CLOSED', None

        # OUT between 12:01 and 2:00 PM -> HALF DAY
        if half_start <= now_t <= half_end:
            return 'OUT', 'HALF DAY'

        # OUT between 2:50 and 5:00 PM -> PRESENT (full day)
        if full_start <= now_t <= full_end:
            return 'OUT', 'PRESENT'

        # Between 14:00–14:50 (2:00–2:50) -> treat as closed (no status change)
        return 'CLOSED', None

    # 3) Already has IN and OUT -> no more changes
    return 'CLOSED', existing_rec.get('status')

@app.route('/api/takeattendance', methods=['POST'])
def take_attendance():
    try:
        # Load trained model and label map
        if not os.path.exists('trainer.yml') or not os.path.exists('label_map.json'):
            return jsonify({'success': False, 'message': 'Model not trained yet'}), 400

        recognizer.read('trainer.yml')
        with open('label_map.json', 'r') as f:
            label_map = json.load(f)

        students = load_students()
        if not students:
            return jsonify({'success': False, 'message': 'No students registered'}), 400

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Could not open camera'}), 500

        recognized_rollno = None
        recognized_name = None
        confidence_val = None

        print("Attendance camera started. Press SPACE to capture, ESC to cancel.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))

                label, confidence = recognizer.predict(face)
                confidence_val = round(100 - confidence, 2)  # convert to similarity-like score

                rollno = label_map.get(str(label)) or label_map.get(label)
                student = find_student_by_roll(rollno) if rollno is not None else None

                if student and confidence < 80:  # lower is better; threshold tweakable
                    name_text = f"{student['name']} ({rollno}) {confidence_val}%"
                    color = (0, 255, 0)
                    recognized_rollno = rollno
                    recognized_name = student['name']
                else:
                    name_text = "Unknown"
                    color = (0, 0, 255)
                    recognized_rollno = None
                    recognized_name = None

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name_text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Take Attendance - SPACE: mark, ESC: exit', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # user confirms capture
                break
            elif key == 27:
                # ESC cancel
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({'success': False, 'message': 'Attendance cancelled'})

        cap.release()
        cv2.destroyAllWindows()

        if not recognized_rollno:
            return jsonify({'success': False, 'message': 'Face not recognized or low confidence (Unknown)'})

        # Apply attendance logic
        today_recs = get_today_records()
        existing = None
        for r in today_recs:
            if r['rollno'] == recognized_rollno:
                existing = r
                break

        action, status = decide_attendance_action(existing)

        if action == 'CLOSED':
            return jsonify({'success': False, 'message': 'Attendance window is closed for now'})

        if action == 'IN':
            rec = save_or_update_attendance(
                rollno=recognized_rollno,
                status='PRESENT',   # provisional, final status after OUT
                intime=get_now_time_str()
            )
            return jsonify({
                'success': True,
                'message': f'IN marked for {recognized_name} ({recognized_rollno}) at {rec["intime"]}',
                'name': recognized_name,
                'rollno': recognized_rollno,
                'confidence': confidence_val,
                'status': rec['status'],
                'intime': rec['intime'],
                'outtime': rec['outtime']
            })

        if action == 'OUT':
            rec = save_or_update_attendance(
                rollno=recognized_rollno,
                status=status,
                outtime=get_now_time_str()
            )
            return jsonify({
                'success': True,
                'message': f'OUT marked for {recognized_name} ({recognized_rollno}) at {rec["outtime"]} ({status})',
                'name': recognized_name,
                'rollno': recognized_rollno,
                'confidence': confidence_val,
                'status': rec['status'],
                'intime': rec['intime'],
                'outtime': rec['outtime']
            })

    except Exception as e:
        print('Error in take_attendance:', e)
        return jsonify({'success': False, 'message': f'Attendance error: {str(e)}'}), 500
@app.route('/api/todays_attendance')
def todays_attendance():
    today = get_today_date_str()
    students = load_students()
    all_att = load_attendance()

    # Build a dict of today’s records: rollno -> record
    today_map = {}
    for r in all_att:
        if r['date'] == today:
            today_map[r['rollno']] = r

    rows = []
    present_count = 0
    absent_count = 0

    for idx, s in enumerate(students, start=1):
        roll = s['rollno']
        rec = today_map.get(roll)

        if rec:
            status = rec['status']
            intime = rec['intime'] or '-'
            outtime = rec['outtime'] or '-'
        else:
            # No record yet today → ABSENT by default, dashes for times
            status = 'ABSENT'
            intime = '-'
            outtime = '-'

        if status == 'PRESENT' or status == 'HALF DAY':
            present_count += 1
        else:
            absent_count += 1

        rows.append({
            'sno': idx,
            'id': roll,
            'name': s['name'],
            'intime': intime,
            'outtime': outtime,
            'status': status
        })

    return jsonify({
        'success': True,
        'date': today,
        'total_students': len(students),
        'present': present_count,
        'absent': absent_count,
        'records': rows
    })

def create_attendance_excel(records, title, date_or_range):
    """Create Excel for daily attendance."""
    from openpyxl import Workbook
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    
    # Colors
    green_fill = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
    red_fill = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
    header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
    
    header_font = Font(bold=True, color="FFFFFF", size=12)
    title_font = Font(bold=True, size=14)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Title
    ws.merge_cells('A1:F1')
    title_cell = ws['A1']
    title_cell.value = title
    title_cell.font = title_font
    title_cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Date range
    ws.merge_cells('A2:F2')
    date_cell = ws['A2']
    date_cell.value = date_or_range
    date_cell.font = Font(italic=True)
    date_cell.alignment = Alignment(horizontal='center')
    
    # Headers
    headers = ['S.NO', 'ID', 'NAME', 'IN TIME', 'OUT TIME', 'STATUS']
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=4, column=col)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border
    
    # Data rows
    for row_idx, rec in enumerate(records, 5):
        cells_data = [
            rec.get('sno', ''),
            rec.get('id', ''),
            rec.get('name', ''),
            rec.get('intime', '-'),
            rec.get('outtime', '-'),
            rec.get('status', '')
        ]
        
        for col, value in enumerate(cells_data, 1):
            cell = ws.cell(row=row_idx, column=col)
            cell.value = value
            cell.border = border
            cell.alignment = Alignment(horizontal='center' if col in [1, 6] else 'left')
            
            if col == 6:  # STATUS column
                if value == 'PRESENT':
                    cell.fill = green_fill
                elif value == 'HALF DAY':
                    cell.fill = yellow_fill
                elif value == 'ABSENT':
                    cell.fill = red_fill
    
    # Column widths
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 20
    ws.column_dimensions['D'].width = 15
    ws.column_dimensions['E'].width = 15
    ws.column_dimensions['F'].width = 15
    
    return wb


def create_student_report_excel(student_name, student_id, start_date, end_date, records, summary_stats):
    """
    Create individual student attendance report Excel.
    summary_stats = {
        'total_days': int,
        'present': int,
        'half_days': int,
        'absent': int,
        'attendance_percentage': float,
        'is_defaulter': bool
    }
    """
    from openpyxl import Workbook
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Student Report"
    
    # Colors
    green_fill = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
    red_fill = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
    header_fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
    summary_fill = PatternFill(start_color="E8F4F8", end_color="E8F4F8", fill_type="solid")
    
    header_font = Font(bold=True, color="FFFFFF", size=12)
    title_font = Font(bold=True, size=14, color="2C3E50")
    summary_font = Font(bold=True, size=11)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # ===== TITLE SECTION =====
    ws.merge_cells('A1:F1')
    title_cell = ws['A1']
    title_cell.value = f"Student Report - {student_name} ({student_id})"
    title_cell.font = title_font
    title_cell.alignment = Alignment(horizontal='center', vertical='center')
    ws.row_dimensions[1].height = 25
    
    # Date range
    ws.merge_cells('A2:F2')
    date_cell = ws['A2']
    date_cell.value = f"From {start_date} To {end_date}"
    date_cell.font = Font(italic=True, size=11)
    date_cell.alignment = Alignment(horizontal='center')
    ws.row_dimensions[2].height = 18
    
    # ===== SUMMARY SECTION =====
    # Summary on left side (A-C)
    summary_row = 4
    summary_data = [
        ('Student', f'{student_name} (ID: {student_id})'),
        ('Date Range', f'{start_date} to {end_date}'),
        ('Total Days', str(summary_stats['total_days'])),
        ('Present', str(summary_stats['present'])),
        ('Half Days', str(summary_stats['half_days'])),
        ('Absent', str(summary_stats['absent'])),
        ('Attendance %', f"{summary_stats['attendance_percentage']}%"),
        ('Status', 'DEFAULTER' if summary_stats['is_defaulter'] else 'GOOD STANDING')
    ]
    
    for label, value in summary_data:
        # Label cell
        label_cell = ws.cell(row=summary_row, column=1)
        label_cell.value = label
        label_cell.font = Font(bold=True, size=10)
        label_cell.fill = summary_fill
        label_cell.border = border
        label_cell.alignment = Alignment(horizontal='left', vertical='center')
        
        # Value cell
        value_cell = ws.cell(row=summary_row, column=2)
        value_cell.value = value
        value_cell.font = Font(size=10)
        value_cell.fill = summary_fill
        value_cell.border = border
        value_cell.alignment = Alignment(horizontal='left', vertical='center')
        
        summary_row += 1
    
    # ===== ATTENDANCE TABLE SECTION =====
    table_start_row = 13
    
    # Table headers
    headers = ['S.NO', 'DATE', 'IN TIME', 'OUT TIME', 'STATUS', 'LATE?', 'EARLY?']
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=table_start_row, column=col)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border
    
    ws.row_dimensions[table_start_row].height = 20
    
    # Table data rows
    for row_idx, rec in enumerate(records, table_start_row + 1):
        cells_data = [
            rec.get('sno', ''),
            rec.get('date', ''),
            rec.get('intime', '-'),
            rec.get('outtime', '-'),
            rec.get('status', ''),
            'YES' if rec.get('islate', False) else 'NO',
            'YES' if rec.get('isearly', False) else 'NO'
        ]
        
        for col, value in enumerate(cells_data, 1):
            cell = ws.cell(row=row_idx, column=col)
            cell.value = value
            cell.border = border
            cell.alignment = Alignment(horizontal='center' if col in [1, 5, 6, 7] else 'left', vertical='center')
            
            # Color status column
            if col == 5:  # STATUS column
                if value == 'PRESENT':
                    cell.fill = green_fill
                elif value == 'HALF DAY':
                    cell.fill = yellow_fill
                elif value == 'ABSENT':
                    cell.fill = red_fill
    
    # Column widths
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 15
    ws.column_dimensions['E'].width = 15
    ws.column_dimensions['F'].width = 10
    ws.column_dimensions['G'].width = 10
    
    return wb

@app.route('/api/attendance_dates')
def attendance_dates():
    all_att = load_attendance()
    dates = sorted({r['date'] for r in all_att}, reverse=True)
    return jsonify({'success': True, 'dates': dates})
@app.route('/api/attendancebydate')
def attendance_by_date():
    date = request.args.get('date')
    if not date:
        return jsonify({'success': False, 'message': 'Date is required'}), 400

    all_att = load_attendance()
    students = load_students()
    name_map = {s['rollno']: s['name'] for s in students}

    day_recs = [r for r in all_att if r['date'] == date]
    rows = []
    for i, r in enumerate(day_recs, start=1):
        rows.append({
            'sno': i,
            'id': r['rollno'],
            'name': name_map.get(r['rollno'], 'Unknown'),
            'intime': r['intime'] or '-',
            'outtime': r['outtime'] or '-',
            'status': r['status'],
            'date': r['date']
        })

    return jsonify({'success': True, 'date': date, 'records': rows})
@app.route('/api/exportattendance')
def export_attendance():
    date = request.args.get('date')
    if not date:
        return jsonify({'success': False, 'message': 'Date is required'}), 400

    all_att = load_attendance()
    students = load_students()
    name_map = {s['rollno']: s['name'] for s in students}

    day_recs = [r for r in all_att if r['date'] == date]
    records = []
    for i, r in enumerate(day_recs, start=1):
        records.append({
            'sno': i,
            'id': r['rollno'],
            'name': name_map.get(r['rollno'], 'Unknown'),
            'intime': r['intime'] or '-',
            'outtime': r['outtime'] or '-',
            'status': r['status'],
            'date': r['date']
        })

    wb = create_attendance_excel(records, f'Attendance Report - {date}', f'Date: {date}')
    
    filename = f'attendance_{date}.xlsx'
    filepath = f'exports/{filename}'
    os.makedirs('exports', exist_ok=True)
    wb.save(filepath)
    
    return send_file(filepath, as_attachment=True, download_name=filename)
@app.route('/api/students_analytics')
def students_analytics():
    students = load_students()
    all_att = load_attendance()

    stats = {}
    for s in students:
        stats[s['rollno']] = {
            'name': s['name'],
            'present': 0,
            'half': 0,
            'absent': 0,
            'total': 0
        }

    for r in all_att:
        roll = r['rollno']
        if roll not in stats:
            continue
        if r['status'] == 'PRESENT':
            stats[roll]['present'] += 1
            stats[roll]['total'] += 1
        elif r['status'] == 'HALF DAY':
            stats[roll]['half'] += 1
            stats[roll]['total'] += 1
        elif r['status'] == 'ABSENT':
            stats[roll]['absent'] += 1
            stats[roll]['total'] += 1

    defaulters = []
    for roll, st in stats.items():
        if st['total'] == 0:
            attendance_pct = 0.0
        else:
            effective_present = st['present'] + 0.5 * st['half']
            attendance_pct = round(100.0 * effective_present / st['total'], 2)

        if attendance_pct < 75.0:
            defaulters.append({
                'id': roll,
                'name': st['name'],
                'attendancepercentage': attendance_pct,
                'presentdays': st['present'],
                'absentdays': st['absent'],
                'halfdays': st['half']
            })

    return jsonify({'success': True, 'defaulters': defaulters})
@app.route('/api/individualreport/<student_id>')
def individual_report(student_id):
    start = request.args.get('startdate')
    end = request.args.get('enddate')
    if not start or not end:
        return jsonify({'error': 'Start and end dates are required'}), 400

    stu = find_student_by_roll(student_id)
    if not stu:
        return jsonify({'error': 'Student not found'}), 404

    all_att = load_attendance()
    history = []
    total_days = 0
    present = half = absent = 0

    for r in all_att:
        if r['rollno'] != student_id:
            continue
        if not (start <= r['date'] <= end):
            continue
        total_days += 1
        status = r['status']
        if status == 'PRESENT':
            present += 1
        elif status == 'HALF DAY':
            half += 1
        elif status == 'ABSENT':
            absent += 1

        history.append({
            'date': r['date'],
            'intime': r['intime'] or '-',
            'outtime': r['outtime'] or '-',
            'status': status,
            'islate': False,
            'isearly': False
        })

    if total_days == 0:
        attendance_pct = 0.0
    else:
        effective_present = present + 0.5 * half
        attendance_pct = round(100.0 * effective_present / total_days, 2)

    return jsonify({
        'studentid': student_id,
        'studentname': stu['name'],
        'totaldays': total_days,
        'presentdays': present,
        'halfdays': half,
        'absentdays': absent,
        'attendancepercentage': attendance_pct,
        'isdefaulter': attendance_pct < 75.0,
        'attendancehistory': history
    })
@app.route('/api/exportstudentreport/<student_id>')
def export_student_report(student_id):
    start = request.args.get('startdate')
    end = request.args.get('enddate')
    if not start or not end:
        return jsonify({'error': 'Dates required'}), 400

    stu = find_student_by_roll(student_id)
    if not stu:
        return jsonify({'error': 'Student not found'}), 404

    all_att = load_attendance()
    records = []
    total_days = 0
    present = half = absent = 0

    for r in all_att:
        if r['rollno'] != student_id or not (start <= r['date'] <= end):
            continue
        
        total_days += 1
        status = r['status']
        
        if status == 'PRESENT':
            present += 1
        elif status == 'HALF DAY':
            half += 1
        elif status == 'ABSENT':
            absent += 1

        records.append({
            'sno': len(records) + 1,
            'date': r['date'],
            'intime': r['intime'] or '-',
            'outtime': r['outtime'] or '-',
            'status': status,
            'islate': False,
            'isearly': False
        })

    if total_days == 0:
        attendance_pct = 0.0
    else:
        effective_present = present + 0.5 * half
        attendance_pct = round(100.0 * effective_present / total_days, 2)

    summary_stats = {
        'total_days': total_days,
        'present': present,
        'half_days': half,
        'absent': absent,
        'attendance_percentage': attendance_pct,
        'is_defaulter': attendance_pct < 75.0
    }

    wb = create_student_report_excel(
        student_name=stu['name'],
        student_id=student_id,
        start_date=start,
        end_date=end,
        records=records,
        summary_stats=summary_stats
    )

    filename = f"student_report_{student_id}_{start}_to_{end}.xlsx"
    filepath = f'exports/{filename}'
    os.makedirs('exports', exist_ok=True)
    wb.save(filepath)

    return send_file(filepath, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
