# AI-Based Classroom Engagement Analyzer

A Flask web application that uses computer vision and emotion recognition to analyze classroom engagement in real-time. The system tracks student emotions, attention levels, and provides insights for educators.

## Features

- **Real-time Video Analysis**: Uses webcam to detect faces and analyze emotions
- **Emotion Recognition**: Identifies 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Engagement Tracking**: Calculates overall classroom engagement percentage
- **Student Tracking**: Tracks individual students and their attention levels
- **User Authentication**: Secure login system with admin and regular user roles
- **Admin Dashboard**: Manage users, view reports, and download session data
- **Chat System**: Real-time messaging for all logged-in users
- **Data Visualization**: Heatmaps and graphs showing engagement over time

## Technologies Used

- **Backend**: Flask, SQLAlchemy, Flask-Login
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: TensorFlow, OpenCV, Ultralytics YOLO
- **Database**: SQLite
- **Video Processing**: OpenCV with Haar cascades

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ankit-HubNation/Ai-Based-Classroom-Enagagement-Analyzer.git
cd Ai-Based-Classroom-Enagagement-Analyzer
```

2. Create a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python init_db.py
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and go to `http://127.0.0.1:5000`

3. Login with default admin credentials:
   - Username: `admin`
   - Password: `admin123`

## Project Structure

```
├── app.py                 # Main Flask application
├── app2.py               # Alternative app version
├── init_db.py            # Database initialization
├── requirements.txt      # Python dependencies
├── emotion_model.hdf5    # Pre-trained emotion recognition model
├── yolov8n.pt           # YOLO model for object detection
├── model.pt             # Additional ML model
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
├── instance/            # Database files
└── __pycache__/         # Python cache
```

## API Endpoints

- `/` - Main dashboard
- `/video` - Video feed with analysis
- `/login` - User authentication
- `/admin/users` - User management (admin only)
- `/engagement` - Current engagement data (JSON)
- `/heatmap` - Engagement heatmap data
- `/download` - Download session report (CSV)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Disclaimer

This application requires a webcam and may use significant CPU/GPU resources for real-time video processing. Ensure your system meets the requirements before running.