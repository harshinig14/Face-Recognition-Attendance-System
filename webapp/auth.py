import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

class TeacherAuth:
    def __init__(self):
        self.teachers_file = os.path.join(os.path.dirname(__file__), 'teachers.json')
        self.initialize_teachers()

    def initialize_teachers(self):
        """Create default teacher account if not exists"""
        if not os.path.exists(self.teachers_file):
            # Default teacher credentials
            default_teachers = {
                "admin": {
                    "id": 1,
                    "username": "admin",
                    "password_hash": generate_password_hash("admin123"),
                    "full_name": "Administrator",
                    "email": "admin@school.edu",
                    "created_date": "2025-11-06"
                },
                "teacher": {
                    "id": 2,
                    "username": "teacher",
                    "password_hash": generate_password_hash("teacher123"),
                    "full_name": "Teacher",
                    "email": "teacher@school.edu",
                    "created_date": "2025-11-06"
                }
            }
            
            with open(self.teachers_file, 'w') as f:
                json.dump(default_teachers, f, indent=4)
            
            print("📝 Default teacher accounts created:")
            print("   Username: admin, Password: admin123")
            print("   Username: teacher, Password: teacher123")

    def authenticate(self, username, password):
        """Authenticate teacher login"""
        try:
            with open(self.teachers_file, 'r') as f:
                teachers = json.load(f)
            
            if username in teachers:
                stored_hash = teachers[username]['password_hash']
                return check_password_hash(stored_hash, password)
            
            return False
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False

    def get_teacher_info(self, username):
        """Get teacher information"""
        try:
            with open(self.teachers_file, 'r') as f:
                teachers = json.load(f)
            
            if username in teachers:
                teacher_info = teachers[username].copy()
                teacher_info.pop('password_hash', None)  # Remove password hash
                return teacher_info
            
            return None
        except Exception as e:
            print(f"Error getting teacher info: {str(e)}")
            return None


# ADD THESE STANDALONE FUNCTIONS - This is what app.py needs!
def authenticate_teacher(username, password):
    """Standalone function for teacher authentication"""
    auth = TeacherAuth()
    if auth.authenticate(username, password):
        teacher_info = auth.get_teacher_info(username)
        return teacher_info
    return None
