from utils import (read_video, save_video)
from trackers import PlayerTracker , BallTracker
from court_points_detector import CourtPointsDetector
import cv2
import json

def main_object_tracking():
    #read video
    input_video_path = r'D:\studying\NTI training\Tennis-analysis-project\object_detction_Tracking\WhatsApp Video 2025-09-19 at 2.37.29 AM.mp4' # still empty 
    video_frames = read_video(input_video_path)

    # detect players and Ball
    player_tracker = PlayerTracker(model_path=r"D:\studying\NTI training\Tennis-analysis-project\object_detction_Tracking\models\yolov8x.pt")
    ball_tracker = BallTracker(model_path=r"D:\studying\NTI training\Tennis-analysis-project\object_detction_Tracking\models\yolo11best.pt")

    player_detections = player_tracker.detect_frames(video_frames)
    ball_detections = ball_tracker.detect_frames(video_frames)

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections) # @@

    # Court Line Detector model
    court_model_path = r"D:\studying\NTI training\Tennis-analysis-project\object_detction_Tracking\models\keypoints_model.pth"
    court_line_detector = CourtPointsDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose player@@
    player_detections = player_tracker.choose_and_filter_players(court_keypoints ,player_detections)

    # draw bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames,ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    #Draw frame number on top left
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    save_video(output_video_frames,r"D:\studying\NTI training\Tennis-analysis-project\object_detction_Tracking\output_video\output_video3.avi")

    return {"player_detections": player_detections,
            "ball_detections": ball_detections
            }


if __name__ == "__main__":
    results = main_object_tracking()
    with open(r"D:\studying\NTI training\Tennis-analysis-project\object_detction_Tracking\output_video\results.json", "w") as f:
        json.dump(results,f)