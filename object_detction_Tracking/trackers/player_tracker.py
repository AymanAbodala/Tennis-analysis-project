from ultralytics import YOLO
import cv2
import sys
sys.path.append('../')
from object_detction_Tracking.my_utils import measure_distance , get_center_of_bbox

class PlayerTracker:
    def __init__(self , model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self , frames):

        """
        Detects players in a given list of frames.

        Args:
            frames (list): A list of frames to detect players in.

        Returns:
            list: A list of dictionaries, 
            where each dictionary contains the track ID of a player
              as the key and a list of bounding boxes as the value.
        """
        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        return player_detections
    
    def detect_frame(self , frame):
        """
        Detects players in a given frame.

        Args:
            frame (np.ndarray): The frame to detect players in.

        Returns:
            dict: A dictionary, where each key is a track ID of a player
              and the value is a list of bounding boxes for that player.

        """
        
        results = self.model.track(frame, persist=True)[0]    
        id_name_dict = results.names
        
        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name=='person':
                player_dict[track_id] = result
        return player_dict


    def draw_bboxes(self , video_frames , player_detections):
        """
        Draws bounding boxes around detected players in a given list of frames.

        Args:
            video_frames (list): A list of frames to draw bounding boxes on.
            player_detections (list): A list of dictionaries, where each dictionary contains the track ID of a player
              as the key and a list of bounding boxes as the value.

        Returns:
            list: A list of frames with bounding boxes drawn around the detected players.
        """
        output_video_frames = []
        for frame , player_dict in zip(video_frames , player_detections):
            for track_id , bbox in player_dict.items():
                x1 , y1 , x2 , y2  = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),
                                                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)  
        return output_video_frames

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Filters the detected players based on the chosen players in the first frame.

        Args:
            court_keypoints (list): A list of keypoints of the court.
            player_detections (list): A list of dictionaries, where each dictionary contains the track ID of a player
              as the key and a list of bounding boxes as the value.

        Returns:
            list: A list of dictionaries, where each dictionary contains the track ID of a chosen player
              as the key and a list of bounding boxes for that player as the value.
        """
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections
    
    def choose_players(self, court_keypoints, player_dict):
        """
        Chooses the two closest players to the court keypoints in the first frame.

        Args:
            court_keypoints (list): A list of keypoints of the court.
            player_dict (dict): A dictionary containing the track ID of a player as the key and a list of bounding boxes as the value.

        Returns:
            list: A list of chosen track IDs of the players.
        """
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        # sorrt the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players