# 🎾 Tennis Video Analysis — Object Detection, Tracking & Action Recognition

## Overview
This repository implements a full pipeline to analyse tennis match videos. It detects players and the ball, detects court points, tracks objects across frames, extracts features, recognises actions, and produces JSON outputs.  
Those JSON files can be sent to a **FastAPI** service to generate recommendations, and a **Streamlit** app can display interactive reports and visualisations.

---

## ✨ Features
- **Object Detection & Tracking**: Players & ball detection with YOLO/tracker.
- **Court Point Detection**: Identify key points and compute homography.
- **Action Recognition**: Classify strokes and player actions.
- **JSON Export**: Unified output for analytics/recommendations.
- **FastAPI Integration**: Send JSON to API for tactical or performance recommendations.
- **Streamlit Dashboard**: Upload JSON/video and view analysis.

---

## 🗂 Repository Structure
object_detection_Tracking/
├─ analysis/
│ ├─ init.py
│ ├─ action_recognition.py # Action/stroke recognition logic
├─ court_points_detector/
│ ├─ pycache/
│ ├─ init.py
│ ├─ court_points_detector.py # Detect court lines/points & compute homography
├─ output_video/
│ ├─ results.json # Example pipeline output
│ ├─ tennis_actions2.json # Example action recognition output
├─ trackers/ # Tracking logic (DeepSort/Kalman etc.)
├─ training/ # Training scripts for detectors or action models
├─ utils/ # Utility functions (video IO, drawing, helpers)
├─ main_object_tracking.py # Entry point to run detection+tracking pipeline
.gitignore # Ignored files (models, venv etc.)

yaml
Copy code

---

## ⚙️ Requirements
- Python 3.10+
- OpenCV, NumPy, Pandas
- PyTorch + torchvision
- Ultralytics YOLO (for detection)
- FastAPI + Uvicorn (for recommendation service)
- Streamlit (for dashboard)

Install:
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
pip install -r requirements.txt
🚀 Usage
1. Run the main pipeline
Use main_object_tracking.py to process a video:

bash
Copy code
python main_object_tracking.py --video path/to/video.mp4 --output output_video/results.json
This will run detection, tracking, court point detection, and action recognition, then export JSON to output_video/.

2. Start FastAPI (recommendation engine)
In your FastAPI folder (create api/main.py as shown in the README):

bash
Copy code
uvicorn api.main:app --reload --port 8000
Send JSON:

bash
Copy code
curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" -d @output_video/results.json
3. Launch Streamlit dashboard
bash
Copy code
streamlit run app/streamlit_app.py
Upload your JSON file and view stats, heatmaps, action timelines, and recommendations.

📝 JSON Format (example)
json
Copy code
{
  "match_id": "match1",
  "video_path": "data/match1.mp4",
  "players": [
    {
      "player_id": 1,
      "actions": [
        {"frame_start": 120, "frame_end": 132, "label": "serve", "confidence": 0.88}
      ],
      "stats": {
        "average_speed": 3.2,
        "shots": 45
      }
    }
  ],
  "ball": {
    "detections": [
      {"frame": 121, "bbox": [x,y,w,h], "velocity": 25.4}
    ]
  },
  "court": {
    "court_points": {
      "baseline_left": [x,y],
      "baseline_right": [x,y]
    }
  }
}
🐳 Deployment
Use Streamlit Cloud or Docker to deploy the dashboard.

Keep large model files out of the repo (add them to .gitignore or use Git LFS).

📜 License
MIT License — see LICENSE for details.

🤝 Contributing
Pull requests and issues are welcome. Please include clear descriptions and test data where possible.

📬 Contact
For questions, open an issue on GitHub.

yaml
Copy code

---

Would you like me to also **add a section about FastAPI and Streamlit setup** (with example `api/main.py` a
