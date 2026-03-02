from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import numpy as np
from engagement import calculate_engagement
from tensorflow.keras.models import load_model
import pandas as pd
import time

# ==============================
# Load Models
# ==============================
model = load_model("emotion_model.hdf5", compile=False)
yolo_model = YOLO("yolov8n.pt")
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

emotion_count = {e: 0 for e in emotion_labels}

# ==============================
# Flask Setup
# ==============================
app = Flask(__name__)
app.secret_key = "supersecretkey"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ==============================
# Dummy Users (Replace with DB later)
# ==============================
class User(UserMixin):
    def __init__(self, id, username, is_admin=False):
        self.id = id
        self.username = username
        self.is_admin = is_admin

users = {
    "1": User("1", "Admin", True),
    "2": User("2", "Student", False)
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ==============================
# Camera Setup
# ==============================
camera = cv2.VideoCapture(0)
engagement_value = 0
session_data = []
graph_data = []

# ==============================
# Video Processing
# ==============================
def generate_frames():
    global engagement_value

    while True:
        success, frame = camera.read()
        if not success:
            continue

        small_frame = cv2.resize(frame, (416, 320))
        results = yolo_model(small_frame, conf=0.5, verbose=False)

        total_score = 0
        face_count = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = small_frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi = cv2.resize(roi, (64, 64)) / 255.0
                    roi = np.reshape(roi, (1, 64, 64, 1))

                    prediction = model.predict(roi, verbose=0)
                    emotion = emotion_labels[np.argmax(prediction)]

                    emotion_count[emotion] += 1

                    score = calculate_engagement(emotion, True)
                    total_score += score
                    face_count += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, emotion, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        engagement_value = round(total_score/face_count,2) if face_count else 0

        session_data.append(engagement_value)
        graph_data.append({
            "timestamp": int(time.time()),
            "engagement": engagement_value
        })

        cv2.putText(frame, f'Engagement: {engagement_value}%',
                    (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ==============================
# Routes
# ==============================

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    login_user(users["1"])  # Auto-login as Admin
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/video')
@login_required
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/engagement')
def engagement():
    return jsonify({"engagement": engagement_value})

@app.route('/emotion_details')
def emotion_details():
    return jsonify(emotion_count)

@app.route('/average')
def average():
    if not session_data:
        return jsonify({"average": 0})
    return jsonify({"average": round(sum(session_data)/len(session_data),2)})

@app.route('/engagement_graph')
def engagement_graph():
    return jsonify(graph_data[-30:])

@app.route('/active_students')
def active_students():
    dummy = [{
        "id": i+1,
        "emotion": list(emotion_count.keys())[i % 7],
        "attention": np.random.randint(40,100),
        "status": np.random.choice(["Focused","Distracted","Sleepy"])
    } for i in range(5)]
    return jsonify(dummy)

@app.route('/admin_users')
def admin_users():
    return "Admin Panel"

@app.route('/chat')
def chat():
    return "Chat Page"

@app.route('/download')
def download():
    df = pd.DataFrame(session_data, columns=["Engagement"])
    df.to_csv("session_report.csv", index=False)
    return "Report Generated"

@app.route('/reset')
def reset():
    global session_data, emotion_count
    session_data = []
    for key in emotion_count:
        emotion_count[key] = 0
    return "Session Reset"

@app.route('/recommendation')
def recommendation():
    if engagement_value < 40:
        msg = "Engagement low. Suggest interactive activity."
    elif engagement_value < 70:
        msg = "Engagement moderate. Consider asking questions."
    else:
        msg = "Engagement high. Continue current teaching strategy."
    return jsonify({"message": msg})

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    app.run(debug=True)