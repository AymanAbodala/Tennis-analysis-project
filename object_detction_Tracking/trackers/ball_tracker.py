from ultralytics import YOLO
import cv2
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self , frame):
        """
        Detects balls in a given frame.

        Args:
            frame (np.ndarray): The frame to detect balls in.

        Returns:
            dict: A dictionary, where each key is a ball track ID
              and the value is a list of bounding boxes for that ball.
        """
        results = self.model.predict(frame, conf=0.15)[0]    
        
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict
    
    def detect_frames(self , frames):
        """
        Detects balls in a given list of frames.

        Args:
            frames (list): A list of frames to detect balls in.

        Returns:
            list: A list of dictionaries, where each dictionary contains the track ID of a ball
              as the key and a list of bounding boxes for that ball as the value.
        """
        ball_detections = []

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        return ball_detections
    
    def draw_bboxes(self , video_frames , player_detections):
        """
        Draws bounding boxes around detected balls in a given list of frames.

        Args:
            video_frames (list): A list of frames to draw bounding boxes on.
            player_detections (list): A list of dictionaries, where each dictionary contains the track ID of a ball
              as the key and a list of bounding boxes for that ball as the value.

        Returns:
            list: A list of frames with bounding boxes drawn around the detected balls.
        """
        output_video_frames = []
        for frame , ball_dict in zip(video_frames , player_detections):
            for track_id , bbox in ball_dict.items():
                x1 , y1 , x2 , y2  = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)  
        return output_video_frames
    
    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates the missing ball positions in a given list of ball positions.

        Args:
            ball_positions (list): A list of dictionaries, where each dictionary contains the track ID of a ball
              as the key and a list of bounding boxes for that ball as the value.

        Returns:
            list: A list of dictionaries, where each dictionary contains the track ID of a ball
              as the key and a list of interpolated bounding boxes for that ball as the value.
        """
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions