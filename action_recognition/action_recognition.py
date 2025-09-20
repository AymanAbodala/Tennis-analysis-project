from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import json
import argparse
from collections import defaultdict

# ---------------- Helper Functions ---------------- #
def get_center(box):
    """Safely get center of bounding box"""
    if box is None or len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def euclidean_distance_center(box1, box2):
    """Calculate distance between centers of two bounding boxes"""
    center1 = get_center(box1)
    center2 = get_center(box2)
    
    if center1 is None or center2 is None:
        return float('inf')
    
    return np.linalg.norm(center1 - center2)

def crop(frame, box):
    """Safely crop frame with bounding box"""
    if box is None or len(box) != 4:
        return None
        
    try:
        x1, y1, x2, y2 = map(int, box)
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return frame[y1:y2, x1:x2]
    except (ValueError, TypeError, IndexError):
        return None

# Custom JSON encoder to handle numpy types and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

# ---------------- Action Recognition Model ---------------- #
class ActionModel:
    def __init__(self, num_classes=5, device="cpu"):
        self.device = device
        self.num_classes = num_classes
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
        ])

        self.classes = ["Forehand", "Backhand", "Serve", "Volley", "No Action"]

    def predict(self, img):
        if img is None:
            return "Unknown"
        try:
            img = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img)
                _, pred = torch.max(outputs, 1)
            return self.classes[pred.item()]
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown"

# ---------------- Tennis Pipeline ---------------- #
class TennisPipeline:
    def __init__(self, action_model, distance_threshold=100):
        self.action_model = action_model
        self.player_actions = {}
        self.distance_threshold = distance_threshold
        self.action_counts = defaultdict(lambda: defaultdict(int))
        self.player_stats = {}
        self.action_sequences = {}  # Store action-based sequences
    
    def get_active_player(self, players_dict, ball_box, frame_idx):
        """Find the active player closest to the ball"""
        if not players_dict or ball_box is None:
            return None, None
            
        min_distance = float('inf')
        active_player_id = None
        active_player_box = None
        
        for player_id, player_box in players_dict.items():
            if player_box is not None and len(player_box) == 4:
                distance = euclidean_distance_center(player_box, ball_box)
                if distance < min_distance:
                    min_distance = distance
                    active_player_id = player_id
                    active_player_box = player_box
                
        if min_distance <= self.distance_threshold:
            return active_player_id, active_player_box
        else:
            return None, None

    def calculate_serve_accuracy(self, player_id):
        """Calculate serve accuracy for a player"""
        if player_id not in self.player_actions:
            return 0
        
        serves = [action for action in self.player_actions[player_id] if action["action"] == "Serve"]
        if not serves:
            return 0
        
        # Count successful serves (those where ball was hit)
        successful_serves = sum(1 for serve in serves if serve["distance_to_ball"] is not None and serve["distance_to_ball"] < self.distance_threshold)
        
        return successful_serves / len(serves) if serves else 0

    def calculate_hit_distribution(self, player_id):
        """Calculate hit distribution for a player"""
        if player_id not in self.action_counts:
            return {}
        
        total_hits = sum(count for action, count in self.action_counts[player_id].items() if action != "No Action")
        if total_hits == 0:
            return {}
        
        distribution = {}
        for action, count in self.action_counts[player_id].items():
            if action != "No Action":
                distribution[action] = count / total_hits
        
        return distribution

    def calculate_unforced_errors(self, player_id):
        """Calculate unforced errors for a player"""
        if player_id not in self.player_actions:
            return 0
        
        # Count frames where player was close to ball but didn't hit it properly
        errors = 0
        for action_data in self.player_actions[player_id]:
            if (action_data["distance_to_ball"] is not None and 
                action_data["distance_to_ball"] < self.distance_threshold and 
                action_data["action"] == "No Action"):
                errors += 1
        
        return errors

    def create_action_based_sequence(self, player_id):
        """Create action-based sequence (don't repeat same action consecutively)"""
        if player_id not in self.player_actions:
            return []
        
        sequence = []
        prev_action = None
        
        for action_data in self.player_actions[player_id]:
            current_action = action_data["action"]
            if current_action != prev_action:
                sequence.append({
                    "action": current_action,
                    "start_frame": action_data["frame"],
                    "distance_to_ball": float(action_data["distance_to_ball"]) if action_data["distance_to_ball"] is not None else None
                })
                prev_action = current_action
            else:
                # Update end frame for the same action
                if sequence:
                    sequence[-1]["end_frame"] = action_data["frame"]
        
        return sequence

    def process_detections(self, frame, detection_data, frame_id):
        """Process player and ball detections for a single frame"""
        player_detections = detection_data.get("player_detections", [])
        ball_detections = detection_data.get("ball_detections", [])
        
        # Get ball bounding box for current frame
        ball_box = None
        if frame_id < len(ball_detections):
            ball_frame_data = ball_detections[frame_id]
            if ball_frame_data and "1" in ball_frame_data:
                ball_box = ball_frame_data["1"]
                if ball_box is not None and len(ball_box) != 4:
                    ball_box = None
        
        # Get player bounding boxes for current frame
        players_dict = {}
        if frame_id < len(player_detections):
            player_frame_data = player_detections[frame_id]
            for player_id, player_box in player_frame_data.items():
                if player_box is not None and len(player_box) == 4:
                    players_dict[player_id] = player_box
        
        # Process EACH player (not just the active one)
        for player_id, player_box in players_dict.items():
            if player_id not in self.player_actions:
                self.player_actions[player_id] = []
                self.player_stats[player_id] = {
                    "total_frames": 0,
                    "action_sequence": []
                }
            
            # Crop and recognize action for EVERY player
            try:
                player_crop = crop(frame, player_box)
                action = self.action_model.predict(player_crop)
                
                # Calculate distance to ball (if ball exists)
                distance_to_ball = euclidean_distance_center(player_box, ball_box) if ball_box else None
                
                # Update action counts
                self.action_counts[player_id][action] += 1
                
                # Update player stats
                self.player_stats[player_id]["total_frames"] += 1
                
                # Save detailed result (convert all values to JSON-serializable types)
                self.player_actions[player_id].append({
                    "frame": frame_id,
                    "action": action,
                    "distance_to_ball": float(distance_to_ball) if distance_to_ball is not None else None,
                    "player_bbox": [float(x) for x in player_box] if player_box is not None else None,
                    "ball_bbox": [float(x) for x in ball_box] if ball_box is not None else None,
                    "is_active": bool(distance_to_ball is not None and distance_to_ball <= self.distance_threshold)
                })
                
            except Exception as e:
                print(f"Error processing player {player_id}: {e}")
        
        # Also track frames where no players are detected
        if not players_dict:
            if "No_Player" not in self.player_actions:
                self.player_actions["No_Player"] = []
            self.player_actions["No_Player"].append({
                "frame": frame_id,
                "action": "No Action",
                "distance_to_ball": None,
                "player_bbox": None,
                "ball_bbox": [float(x) for x in ball_box] if ball_box is not None else None,
                "is_active": False
            })

    def export_results(self, output_file="actions.json"):
        # Create action-based sequences
        for player_id in self.player_actions.keys():
            self.action_sequences[player_id] = self.create_action_based_sequence(player_id)
        
        # Create the new output format with advanced analytics
        results = {
            "player_1": {
                "action_counts": dict(self.action_counts.get("1", {})),
                "total_actions": sum(self.action_counts.get("1", {}).values()),
                "advanced_analytics": {
                    "serve_accuracy": self.calculate_serve_accuracy("1"),
                    "hit_distribution": self.calculate_hit_distribution("1"),
                    "unforced_errors": self.calculate_unforced_errors("1"),
                    "total_shots": sum(count for action, count in self.action_counts.get("1", {}).items() if action != "No Action")
                },
                "action_sequence": self._make_json_serializable(self.action_sequences.get("1", [])),
                "detailed_stats": self._make_json_serializable({
                    "total_frames": self.player_stats.get("1", {}).get("total_frames", 0),
                    "active_frames": sum(1 for action in self.player_actions.get("1", []) if action.get("is_active", False))
                })
            },
            "player_2": {
                "action_counts": dict(self.action_counts.get("2", {})),
                "total_actions": sum(self.action_counts.get("2", {}).values()),
                "advanced_analytics": {
                    "serve_accuracy": self.calculate_serve_accuracy("2"),
                    "hit_distribution": self.calculate_hit_distribution("2"),
                    "unforced_errors": self.calculate_unforced_errors("2"),
                    "total_shots": sum(count for action, count in self.action_counts.get("2", {}).items() if action != "No Action")
                },
                "action_sequence": self._make_json_serializable(self.action_sequences.get("2", [])),
                "detailed_stats": self._make_json_serializable({
                    "total_frames": self.player_stats.get("2", {}).get("total_frames", 0),
                    "active_frames": sum(1 for action in self.player_actions.get("2", []) if action.get("is_active", False))
                })
            },
            "match_summary": {
                "total_frames_processed": max([len(actions) for actions in self.player_actions.values()] + [0]),
                "distance_threshold_used": self.distance_threshold,
                "player_with_most_actions": max(["1", "2"], key=lambda x: sum(self.action_counts.get(x, {}).values()), default="None"),
                "most_common_action": max(
                    [action for player in ["1", "2"] for action in self.action_counts.get(player, {}).items()],
                    key=lambda x: x[1],
                    default=("None", 0)
                )[0]
            }
        }
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4, cls=CustomJSONEncoder)
        print(f"[INFO] Results exported to {output_file}")
        
        # Print summary
        print("\n=== ADVANCED ANALYTICS SUMMARY ===")
        for player_id in ["1", "2"]:
            if player_id in self.action_counts:
                print(f"Player {player_id}:")
                print(f"  Serve Accuracy: {self.calculate_serve_accuracy(player_id):.2%}")
                print(f"  Unforced Errors: {self.calculate_unforced_errors(player_id)}")
                print(f"  Hit Distribution:")
                for action, percentage in self.calculate_hit_distribution(player_id).items():
                    print(f"    {action}: {percentage:.2%}")
                print()

    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-serializable types"""
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        elif obj is None:
            return None
        else:
            return obj

# ---------------- Run Example ---------------- #

def run_pipeline(video_path, detection_json, 
                 output_json=f"D:\\studying\\NTI training\\Tennis-analysis-project\\output_files\\tennis_actions{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                   distance_threshold=100):
    """
    Run the Tennis Action Recognition Pipeline with provided paths and parameters.
    Args:
        video_path (str): Path to input video file
        detection_json (str): Path to detection JSON file
        output_json (str): Path to output JSON file
        distance_threshold (float): Distance threshold for ball possession
    """
    # Load detection data from JSON file
    try:
        with open(detection_json, 'r') as f:
            detection_data = json.load(f)
        print(f"[INFO] Successfully loaded detection data from {detection_json}")
        print(f"Player detection frames: {len(detection_data.get('player_detections', []))}")
        print(f"Ball detection frames: {len(detection_data.get('ball_detections', []))}")
        if detection_data.get('player_detections'):
            print(f"Player IDs detected: {list(detection_data['player_detections'][0].keys())}")
    except FileNotFoundError:
        print(f"[ERROR] Detection JSON file not found at {detection_json}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON format in {detection_json}")
        return

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file {video_path}")
        return

    # Initialize models and pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    action_model = ActionModel(num_classes=5, device=device)
    pipeline = TennisPipeline(action_model, distance_threshold=distance_threshold)

    # Process each frame
    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Processing {total_frames} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pipeline.process_detections(frame, detection_data, frame_id)

        if frame_id % 100 == 0:
            print(f"[INFO] Processed frame {frame_id}/{total_frames}")

        frame_id += 1

    cap.release()
    pipeline.export_results(output_json)
    print(f"[INFO] Processing complete. Results saved to {output_json}")
    return output_json