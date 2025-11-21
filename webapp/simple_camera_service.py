# simple_camera_service.py
"""
Complete SimpleCameraService with IN/OUT attendance with TIME-BASED RULES

✅ FIXED RULES:
- IN allowed ONLY from 7:45 AM to 10:00 AM
- OUT starts after 12:00 PM
- OUT between 12:01 PM - 2:00 PM → HALF DAY
- OUT after 2:50 PM to 5:00 PM → PRESENT (full day, shown in green)
- After 5:00 PM → attendance closed for the day
- No IN and no OUT → ABSENT
- Only IN without OUT → HALF DAY
- UNKNOWN faces are NOT logged, NOT buffered, NOT marked
- Attendance saved in attendance.json as:
  { "YYYY-MM-DD": { "<id>": { "name": "...", "in_time": "...", "out_time": "...", "status": "PRESENT/HALF DAY/ABSENT" } } }
"""

import cv2
import numpy as np
import json
import time
import threading
from datetime import datetime, time as time_obj
from pathlib import Path
from collections import deque, Counter

class SimpleCameraService:
    def __init__(self, camera_index=0, target_fps=15):
        # Paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / 'python-module' / 'data'
        self.PHOTOS_DIR = self.DATA_DIR / 'student_photos'
        self.LOGS_DIR = self.DATA_DIR / 'logs'
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Camera params
        self.camera_index = camera_index
        self.camera = None
        self.is_running = False
        self.target_fps = target_fps
        self.frame_interval = 1.0 / float(self.target_fps)

        # Recognizer / students
        self.recognizer = None
        self.label_dict = {}  # numeric id -> name and str key
        self.students = []

        # Tracking
        self.next_track_id = 1
        self.tracks = {}  # track_id -> {...}
        self.track_lock = threading.Lock()
        self.track_disappear_t = 1.2

        # Attendance & logs
        self.attendance_file = self.DATA_DIR / 'attendance.json'
        self.log_file = self.LOGS_DIR / 'recognition_log.json'
        self.log_lock = threading.Lock()

        # State/status
        self.current_status = {
            'camera_active': False,
            'last_recognized': None,
            'last_recognition_time': None,
            'confidence': 0
        }

        # Today key
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.todays_attendance = {}  # loaded on init

        # Load model and students
        self._load_model()
        self._load_students()
        self._load_today_attendance()

        # ✅ FIX: Initialize absent students immediately on load
        self._initialize_absent_students()

        # Start auto scheduler for auto-absent (daemon)
        try:
            self._start_auto_scheduler()
        except Exception as e:
            print("⚠️ Scheduler start failed:", e)

    # -------------------------
    # Model & students loaders
    # -------------------------
    def _load_model(self):
        classifier_path = self.DATA_DIR / 'classifier.xml'
        if classifier_path.exists():
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(str(classifier_path))
                print("✅ Recognizer loaded.")
            except Exception as e:
                print("❌ Error loading recognizer:", e)
                self.recognizer = None
        else:
            self.recognizer = None
            print("⚠️ classifier.xml not found; recognizer disabled.")

    def _load_students(self):
        students_path = self.DATA_DIR / 'students.json'
        self.students = []
        self.label_dict = {}
        if students_path.exists():
            try:
                with open(students_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        if isinstance(data, list):
                            self.students = data
                        elif isinstance(data, dict):
                            self.students = [data]
                # build label dict
                for s in self.students:
                    try:
                        sid = int(s.get('id'))
                        name = s.get('name')
                        if name:
                            self.label_dict[sid] = name
                            # also set str key for safety
                            self.label_dict[str(sid)] = name
                    except Exception:
                        continue
                print(f"✅ Loaded {len(self.students)} students.")
            except Exception as e:
                print("❌ Error loading students.json:", e)
        else:
            print("⚠️ students.json not found; no students loaded.")

    # -------------------------
    # Today's attendance load/save
    # -------------------------
    def _load_today_attendance(self):
        self.todays_attendance = {}
        if self.attendance_file.exists():
            try:
                with open(self.attendance_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        all_data = json.loads(content)
                        self.todays_attendance = all_data.get(self.today, {})
            except Exception as e:
                print("❌ Error reading attendance.json:", e)
                self.todays_attendance = {}
        else:
            self.todays_attendance = {}

    def _save_attendance(self):
        try:
            if self.attendance_file.exists():
                with open(self.attendance_file, 'r') as f:
                    content = f.read().strip()
                    all_data = json.loads(content) if content else {}
            else:
                all_data = {}
            all_data[self.today] = self.todays_attendance
            with open(self.attendance_file, 'w') as f:
                json.dump(all_data, f, indent=2)
        except Exception as e:
            print("❌ Error writing attendance.json:", e)

    # ✅ NEW: Initialize absent students on load
    def _initialize_absent_students(self):
        """
        ✅ NEW: Mark all students without IN time as ABSENT immediately
        This ensures absent count shows correctly before 10:00 AM auto-check
        """
        try:
            for s in self.students:
                sid = str(s.get('id'))
                name = s.get('name')
                if sid not in self.todays_attendance:
                    # No record exists → mark as ABSENT
                    self.todays_attendance[sid] = {
                        "name": name,
                        "in_time": None,
                        "out_time": None,
                        "status": "ABSENT"
                    }
            # Save the initialized attendance
            self._save_attendance()
            print(f"✅ Initialized {len(self.students)} students in attendance (absent if no IN)")
        except Exception as e:
            print(f"❌ Error initializing absent students: {e}")

    # -------------------------
    # Auto absent check + HALF DAY for missing OUT
    # -------------------------
    def run_auto_absent_check(self):
        """
        ✅ NEW LOGIC:
        1. Mark ABSENT for all students who don't have IN by 10:00 AM
        2. Mark HALF DAY for students with IN but no OUT time
        """
        try:
            print("⏰ Auto-absent check running...")
            for s in self.students:
                sid = str(s.get('id'))
                name = s.get('name')
                if sid not in self.todays_attendance:
                    # No IN recorded → mark ABSENT
                    self.todays_attendance[sid] = {
                        "name": name,
                        "in_time": None,
                        "out_time": None,
                        "status": "ABSENT"
                    }
                else:
                    # Has IN but no OUT → mark HALF DAY (if not already PRESENT)
                    rec = self.todays_attendance[sid]
                    if rec.get("in_time") and not rec.get("out_time"):
                        if rec.get("status") != "PRESENT":
                            rec["status"] = "HALF DAY"
            self._save_attendance()
            print("✅ Auto-absent check complete.")
        except Exception as e:
            print("❌ Auto-absent check error:", e)

    def _start_auto_scheduler(self):
        """
        Background daemon that triggers:
        1. Auto-absent check at 10:00 AM
        2. Auto HALF DAY check at end of day (23:59)
        """
        def runner():
            already_ran_absent = None
            already_ran_halfday = None
            while True:
                try:
                    now_dt = datetime.now()
                    now_date = now_dt.strftime('%Y-%m-%d')
                    now_time = now_dt.time()

                    # ✅ Run auto-absent once per day at/after 10:00 AM
                    if now_date != already_ran_absent and now_time >= time_obj(10, 0):
                        self._load_students()
                        self._load_today_attendance()
                        self.run_auto_absent_check()
                        already_ran_absent = now_date

                    # ✅ Run auto HALF DAY check once per day at/after 23:55
                    if now_date != already_ran_halfday and now_time >= time_obj(23, 55):
                        self._load_today_attendance()
                        self.run_auto_absent_check()  # This also handles HALF DAY
                        already_ran_halfday = now_date
                except Exception as e:
                    print("⚠️ Auto-scheduler error:", e)
                time.sleep(30)

        t = threading.Thread(target=runner, daemon=True)
        t.start()

    # -------------------------
    # Logging (known only)
    # -------------------------
    def _append_log(self, entry: dict):
        with self.log_lock:
            data = []
            if self.log_file.exists():
                try:
                    with open(self.log_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                except Exception:
                    data = []
            data.append(entry)
            try:
                with open(self.log_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print("❌ Error writing recognition_log.json:", e)

    # -------------------------
    # Centroid tracker helpers
    # -------------------------
    def _centroid(self, bbox):
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))

    def _match_tracks(self, detections):
        now = time.time()
        det_centroids = [self._centroid(b) for b in detections]
        assigned = {}

        with self.track_lock:
            active_tracks = {tid: t for tid, t in self.tracks.items()
                           if now - t['last_seen'] <= self.track_disappear_t}
            used_tracks = set()

            for i, c in enumerate(det_centroids):
                best_tid = None
                best_dist = None
                for tid, t in active_tracks.items():
                    if tid in used_tracks:
                        continue
                    tc = t['centroid']
                    dist = (tc[0]-c[0])**2 + (tc[1]-c[1])**2
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None and best_dist is not None and best_dist < (160**2):
                    assigned[i] = best_tid
                    used_tracks.add(best_tid)
                    self.tracks[best_tid]['bbox'] = detections[i]
                    self.tracks[best_tid]['centroid'] = det_centroids[i]
                    self.tracks[best_tid]['last_seen'] = now
                    self.tracks[best_tid]['age'] += 1
                else:
                    assigned[i] = None

        return assigned

    def _create_track(self, bbox):
        with self.track_lock:
            tid = self.next_track_id
            self.next_track_id += 1
            now = time.time()
            self.tracks[tid] = {
                'bbox': bbox,
                'centroid': self._centroid(bbox),
                'last_seen': now,
                'age': 1,
                'buffer': deque(maxlen=5),
                'marked_key': None
            }
        return tid

    def _cleanup_tracks(self):
        now = time.time()
        remove = []
        with self.track_lock:
            for tid, t in list(self.tracks.items()):
                if now - t['last_seen'] > self.track_disappear_t:
                    remove.append(tid)
        for tid in remove:
            try:
                del self.tracks[tid]
            except:
                pass

    # -------------------------
    # Recognition smoothing (KNOWN only)
    # -------------------------
    def _process_face_for_track(self, face_gray, tid):
        """
        Predict with LBPH. Only push KNOWN and reasonably confident predictions into buffer.
        Returns (stable_label:int, stable_name:str, avg_conf:float) when stabilized.
        Otherwise (None, None, None).
        UNKNOWN predictions are ignored (not buffered).
        """
        if self.recognizer is None:
            return (None, None, None)

        try:
            face_resized = cv2.resize(face_gray, (200, 200))
        except Exception:
            return (None, None, None)

        try:
            label, conf = self.recognizer.predict(face_resized)
            conf_score = 100.0 - conf
        except Exception:
            return (None, None, None)

        # If label not in dict or low confidence -> treat as UNKNOWN and don't buffer
        if label not in self.label_dict or conf_score < 40:
            return (None, None, None)

        with self.track_lock:
            buf = self.tracks[tid]['buffer']
            buf.append((label, conf_score))

            # if buffer full -> check stability
            if len(buf) == buf.maxlen:
                labels = [l for l, c in buf]
                # all equal -> stable
                if labels.count(labels[0]) == len(labels):
                    avg_conf = sum(c for l, c in buf) / len(buf)
                    stable_label = labels[0]
                    stable_name = self.label_dict.get(stable_label, "Unknown")
                    return (stable_label, stable_name, avg_conf)

                # majority rule
                cnt = Counter(labels)
                top_label, top_count = cnt.most_common(1)[0]
                if top_count >= 3:
                    confs = [c for l, c in buf if l == top_label]
                    avg_conf = sum(confs) / len(confs) if confs else 0
                    if avg_conf > 45:
                        stable_label = top_label
                        stable_name = self.label_dict.get(stable_label, "Unknown")
                        return (stable_label, stable_name, avg_conf)

        return (None, None, None)

    # -------------------------
    # ✅ FIXED: IN / OUT Marking with TIME RULES
    # -------------------------
    def _mark_in_time(self, student_id, name):
        """
        ✅ NEW RULE: IN allowed ONLY from 7:45 AM to 10:00 AM
        student_id: string
        """
        now = datetime.now()
        current_time = now.time()

        # ✅ Check if IN window is open (7:45 AM to 10:00 AM)
        if not (time_obj(7, 45) <= current_time < time_obj(10, 0)):
            return False

        # ensure todays_attendance entry
        if student_id not in self.todays_attendance:
            self.todays_attendance[student_id] = {
                "name": name,
                "in_time": None,
                "out_time": None,
                "status": "ABSENT"
            }

        rec = self.todays_attendance[student_id]
        if rec.get("in_time") is None:
            rec["in_time"] = now.strftime("%H:%M:%S")
            # Initially mark as HALF DAY until OUT is given
            rec["status"] = "HALF DAY"
            self._save_attendance()
            print(f"🟢 IN recorded for {name} ({student_id}) at {rec['in_time']}")
            return True

        return False

    def _mark_out_time(self, student_id, name):
        """
        ✅ FIXED RULES:
        - OUT starts after 12:00 PM
        - OUT between 12:00 PM - 2:00 PM → HALF DAY
        - OUT after 2:50 PM to 5:00 PM → PRESENT (full day)
        - After 5:00 PM → attendance closed
        - Can only mark OUT if IN exists
        student_id: string
        """
        now = datetime.now()
        current_time = now.time()

        # ✅ Check if OUT window is open (after 12:00 PM and before 5:00 PM)
        if current_time < time_obj(12, 0):
            return False
        
        # ✅ After 5:00 PM, attendance is closed
        if current_time >= time_obj(17, 0):
            return False

        # Check if student has IN time
        if student_id not in self.todays_attendance:
            return False

        rec = self.todays_attendance[student_id]
        if rec.get("in_time") is None:
            # Cannot mark OUT without IN
            return False

        if rec.get("out_time") is None:
            rec["out_time"] = now.strftime("%H:%M:%S")

            # ✅ FIXED: Determine status based on OUT time
            # Parse the OUT time to compare properly
            out_time_parts = rec["out_time"].split(":")
            out_hour = int(out_time_parts[0])
            out_minute = int(out_time_parts[1])
            out_second = int(out_time_parts[2]) if len(out_time_parts) > 2 else 0
            out_time_obj = time_obj(out_hour, out_minute, out_second)

            if time_obj(12, 0) <= out_time_obj < time_obj(14, 0):
                # OUT between 12:00 PM - 2:00 PM → HALF DAY
                rec["status"] = "HALF DAY"
                print(f"🟡 OUT recorded for {name} ({student_id}) at {rec['out_time']} → HALF DAY")
            elif out_time_obj >= time_obj(14, 50):
                # ✅ FIXED: OUT at or after 2:50 PM → PRESENT (full day)
                rec["status"] = "PRESENT"
                print(f"🟢 OUT recorded for {name} ({student_id}) at {rec['out_time']} → PRESENT")
            else:
                # Between 2:00 PM - 2:50 PM (waiting window)
                rec["status"] = "HALF DAY"
                print(f"🟡 OUT recorded for {name} ({student_id}) at {rec['out_time']} → HALF DAY (before 2:50 PM)")

            self._save_attendance()
            return True

        return False

    # -------------------------
    # Recognition loop
    # -------------------------
    def start_camera(self):
        if self.is_running:
            return False

        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            print("❌ Camera open failed")
            return False

        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except:
            pass

        self.is_running = True
        self.current_status['camera_active'] = True
        self._thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._thread.start()
        print("✅ Camera started")
        return True

    def stop_camera(self):
        self.is_running = False
        self.current_status['camera_active'] = False
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        cv2.destroyAllWindows()
        print("⏹️ Camera stopped")

    def _recognition_loop(self):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        window_name = 'Face Recognition Attendance'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        last_frame_time = 0.0

        while self.is_running:
            start_time = time.time()
            elapsed = start_time - last_frame_time
            if elapsed < self.frame_interval:
                time.sleep(max(0.0, self.frame_interval - elapsed))
            last_frame_time = time.time()

            ret, frame = self.camera.read()
            if not ret:
                continue

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(100, 100)
            )

            detections = []
            for (x, y, w, h) in faces:
                x = max(0, int(x)); y = max(0, int(y))
                w = max(10, int(w)); h = max(10, int(h))
                detections.append((x, y, w, h))

            assignment = self._match_tracks(detections)

            # create new tracks if needed
            for i, bbox in enumerate(detections):
                tid = assignment.get(i)
                if tid is None:
                    tid = self._create_track(bbox)
                    assignment[i] = tid

            for i, bbox in enumerate(detections):
                tid = assignment[i]
                x, y, w, h = bbox

                ex = int(w * 0.05); ey = int(h * 0.05)
                x1 = max(0, x - ex); y1 = max(0, y - ey)
                x2 = min(frame.shape[1], x + w + ex); y2 = min(frame.shape[0], y + h + ey)

                face_gray = gray[y1:y2, x1:x2]
                if face_gray.size == 0:
                    continue

                stable_label, stable_name, stable_conf = self._process_face_for_track(face_gray, tid)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if stable_name:
                    # stable known identity detected
                    cv2.putText(display_frame, f"{stable_name} ({stable_conf:.1f}%)", (x, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # mark per-track key to avoid repeated track-marking
                    with self.track_lock:
                        t = self.tracks.get(tid)
                        if t is not None and not t.get('marked_key'):
                            t['marked_key'] = f"{stable_label}_{datetime.now().strftime('%Y-%m-%d')}"

                            # ✅ Decide IN or OUT based on time windows
                            now = datetime.now()
                            ct = now.time()
                            sid_str = str(stable_label)

                            # ✅ IN window: 7:45 AM to 10:00 AM
                            if time_obj(7, 45) <= ct < time_obj(10, 0):
                                self._mark_in_time(sid_str, stable_name)
                            # ✅ OUT window: 12:00 PM to 5:00 PM
                            elif time_obj(12, 0) <= ct < time_obj(17, 0):
                                self._mark_out_time(sid_str, stable_name)

                            # log known recognition event
                            log_entry = {
                                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'name': stable_name,
                                'id': str(stable_label),
                                'confidence': round(stable_conf, 2),
                                'track_id': int(tid)
                            }
                            self._append_log(log_entry)
                else:
                    # show transient predicted name if buffer hints, otherwise show Unknown
                    transient = None
                    with self.track_lock:
                        t = self.tracks.get(tid)
                        if t and len(t['buffer']) > 0:
                            labels = [l for l, c in t['buffer']]
                            cnt = Counter(labels)
                            top_label, top_count = cnt.most_common(1)[0]
                            if top_count >= 2 and top_label in self.label_dict:
                                transient = self.label_dict.get(top_label)

                    if transient:
                        cv2.putText(display_frame, f"{transient} (..)", (x, y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(display_frame, "Unknown", (x, y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

            # cleanup old tracks
            self._cleanup_tracks()

            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # stop on 'q'
                self.stop_camera()
                break

        cv2.destroyAllWindows()

    # -------------------------
    # Status for API
    # -------------------------
    def get_status(self):
        return self.current_status

    # -------------------------
    # Excel export
    # -------------------------
    def export_attendance_excel(self, date=None):
        from openpyxl import Workbook
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            if not self.attendance_file.exists():
                return None

            with open(self.attendance_file, 'r') as f:
                data = json.load(f)

            if date not in data:
                return None

            # data[date] is a dict keyed by student_id -> record
            records_map = data[date]

            # convert to list for writing with stable order
            records = []
            for sid, rec in records_map.items():
                records.append({
                    'id': sid,
                    'name': rec.get('name'),
                    'in_time': rec.get('in_time'),
                    'out_time': rec.get('out_time'),
                    'status': rec.get('status')
                })

            wb = Workbook()
            ws = wb.active
            ws.title = "Attendance"
            headers = ["S.No", "ID", "Name", "IN Time", "OUT Time", "Status"]
            ws.append(headers)

            for idx, r in enumerate(records, 1):
                ws.append([idx, r['id'], r['name'], r['in_time'] or "", r['out_time'] or "", r['status'] or ""])

            export_path = self.DATA_DIR / f'attendance_{date}.xlsx'
            wb.save(export_path)
            return str(export_path)
        except Exception as e:
            print("❌ Export error:", e)
            return None

# End of simple_camera_service.py
