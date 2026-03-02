from flask import Flask, render_template, Response, jsonify, send_file, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
from engagement import calculate_engagement
from tensorflow.keras.models import load_model
import pandas as pd
import time
import os
from datetime import datetime
import threading
from time import time as current_time

# ==============================
# Load Emotion Model
# ==============================
model = load_model("emotion_model.hdf5", compile=False)

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
emotion_count = {label: 0 for label in emotion_labels}

# ==============================
# Flask App Setup
# ==============================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ==============================
# Database Models
# ==============================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='messages')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==============================
# Camera & Engagement Globals
# ==============================
camera = cv2.VideoCapture(0)

engagement_value = 0
session_data = []           # list of engagement values over time
heatmap_data = []           # list of {timestamp, engagement}

# Student tracking
active_students = {}
next_student_id = 0
students_lock = threading.Lock()
TRACKER_TIMEOUT = 2.0       # seconds before removing a student

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# ==============================
# Video Streaming Generator
# ==============================
def generate_frames():
    global engagement_value, active_students, next_student_id, emotion_count

    while True:
        success, frame = camera.read()
        if not success or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        now = current_time()

        # ---- Face tracking ----
        if len(faces) == 0:
            with students_lock:
                to_delete = [sid for sid, data in active_students.items()
                             if now - data['last_seen'] > TRACKER_TIMEOUT]
                for sid in to_delete:
                    del active_students[sid]
            face_to_id = {}
        else:
            unmatched_faces = list(enumerate(faces))   # (index, (x,y,w,h))
            face_to_id = {}
            with students_lock:
                # Match existing trackers
                for student_id, data in list(active_students.items()):
                    best_iou = 0.3
                    best_idx = -1
                    for i, (idx, box) in enumerate(unmatched_faces):
                        cur_iou = iou(data['bbox'], box)
                        if cur_iou > best_iou:
                            best_iou = cur_iou
                            best_idx = i
                    if best_idx != -1:
                        idx, box = unmatched_faces.pop(best_idx)
                        face_to_id[idx] = student_id
                        active_students[student_id]['bbox'] = box
                        active_students[student_id]['last_seen'] = now
                    else:
                        if now - data.get('last_seen', now) > TRACKER_TIMEOUT:
                            del active_students[student_id]

                # New students for leftover faces
                for idx, box in unmatched_faces:
                    student_id = next_student_id
                    next_student_id += 1
                    active_students[student_id] = {
                        'bbox': box,
                        'emotion': 'Unknown',
                        'attention': 0,
                        'status': 'Unknown',
                        'last_seen': now
                    }
                    face_to_id[idx] = student_id

        # ---- Emotion recognition and drawing ----
        total_score = 0
        face_count = 0

        for idx, (x, y, w, h) in enumerate(faces):
            student_id = face_to_id.get(idx)

            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi / 255.0
            roi = np.reshape(roi, (1, 64, 64, 1))

            prediction = model.predict(roi, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            emotion_count[emotion] += 1

            individual_score = calculate_engagement(emotion, True)
            total_score += individual_score
            face_count += 1

            if emotion == "Sad":
                status = "Sleepy"
            elif emotion in ["Angry", "Disgust"]:
                status = "Distracted"
            else:
                status = "Focused"

            # Update tracker
            if student_id is not None:
                with students_lock:
                    if student_id in active_students:
                        active_students[student_id]['emotion'] = emotion
                        active_students[student_id]['attention'] = individual_score
                        active_students[student_id]['status'] = status

            # Draw face box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Attention: {individual_score}%",
                        (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, status,
                        (x, y+h+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Update global engagement
        if face_count > 0:
            engagement_value = round(total_score / face_count, 2)
        else:
            engagement_value = 0

        session_data.append(engagement_value)
        heatmap_data.append({
            "timestamp": now,
            "engagement": engagement_value
        })

        # Low attention warning
        if engagement_value < 40:
            cv2.putText(frame, "LOW CLASSROOM ATTENTION!",
                        (120, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3)

        cv2.putText(frame, f'Class Engagement: {engagement_value}%',
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==============================
# Routes – Authentication
# ==============================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ==============================
# Protected Routes (all users)
# ==============================
@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)

@app.route('/video')
@login_required
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/engagement')
@login_required
def engagement():
    return jsonify({"engagement": engagement_value})

@app.route('/emotion_details')
@login_required
def emotion_details():
    return jsonify(emotion_count)

@app.route('/average')
@login_required
def average():
    if len(session_data) == 0:
        return jsonify({"average": 0})
    return jsonify({"average": round(sum(session_data)/len(session_data), 2)})

@app.route('/heatmap')
@login_required
def heatmap():
    return jsonify(heatmap_data)

@app.route('/download')
@login_required
def download():
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    filename = "session_report.csv"
    df = pd.DataFrame(session_data, columns=["Engagement"])
    df.to_csv(filename, index=False)
    return send_file(filename, as_attachment=True)

@app.route('/snapshot')
@login_required
def snapshot():
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    success, frame = camera.read()
    if success:
        filename = "snapshot.jpg"
        cv2.imwrite(filename, frame)
        return send_file(filename, as_attachment=True)
    return "Camera Error"

@app.route('/reset')
@login_required
def reset():
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    global session_data, emotion_count, heatmap_data, active_students, next_student_id
    session_data = []
    heatmap_data = []
    for key in emotion_count:
        emotion_count[key] = 0
    with students_lock:
        active_students.clear()
        next_student_id = 0
    flash("Session reset", "info")
    return redirect(url_for('index'))

@app.route('/recommendation')
@login_required
def recommendation():
    if engagement_value < 40:
        return jsonify({"message": "Engagement low. Suggest interactive activity."})
    elif engagement_value < 70:
        return jsonify({"message": "Engagement moderate. Consider asking questions."})
    else:
        return jsonify({"message": "Engagement high. Continue current teaching strategy."})

@app.route('/engagement_graph')
@login_required
def engagement_graph():
    if not current_user.is_admin:
        return jsonify({"error": "Access denied"}), 403
    now = current_time()
    recent = [d for d in heatmap_data if d['timestamp'] > now - 30]
    return jsonify(recent)

@app.route('/active_students')
@login_required
def get_active_students():
    if not current_user.is_admin:
        return jsonify({"error": "Access denied"}), 403
    with students_lock:
        students = []
        for sid, data in active_students.items():
            students.append({
                'id': sid,
                'emotion': data['emotion'],
                'attention': data['attention'],
                'status': data['status']
            })
    return jsonify(students)

# ==============================
# Admin Panel – User Management (admin only)
# ==============================
@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('index'))
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/add', methods=['POST'])
@login_required
def admin_add_user():
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    username = request.form.get('username')
    password = request.form.get('password')
    if not username or not password:
        flash('Username and password required', 'danger')
        return redirect(url_for('admin_users'))
    if User.query.filter_by(username=username).first():
        flash('Username already exists', 'danger')
        return redirect(url_for('admin_users'))
    user = User(username=username, is_admin=False)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    flash(f'User {username} added successfully', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/delete/<int:user_id>')
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot delete yourself', 'danger')
        return redirect(url_for('admin_users'))
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.username} deleted', 'success')
    return redirect(url_for('admin_users'))

# ==============================
# Chat (all logged-in users)
# ==============================
@app.route('/chat')
@login_required
def chat():
    messages = ChatMessage.query.order_by(ChatMessage.timestamp.desc()).limit(50).all()
    messages.reverse()
    return render_template('chat.html', messages=messages)

@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    msg = request.form.get('message')
    if msg and msg.strip():
        chat_msg = ChatMessage(
            user_id=current_user.id,
            username=current_user.username,
            message=msg.strip()
        )
        db.session.add(chat_msg)
        db.session.commit()
    return redirect(url_for('chat'))

@app.route('/get_messages')
@login_required
def get_messages():
    messages = ChatMessage.query.order_by(ChatMessage.timestamp.desc()).limit(50).all()
    messages.reverse()
    return jsonify([{
        'username': m.username,
        'message': m.message,
        'timestamp': m.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    } for m in messages])

# ==============================
# Create DB Tables and default admin
# ==============================
with app.app_context():
    db.create_all()
    if User.query.count() == 0:
        admin = User(username='admin', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("Initial admin user created: admin / admin123")

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)