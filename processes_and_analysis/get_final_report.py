import json
import numpy as np
from collections import defaultdict
from datetime import datetime

# ===== HELPER FUNCTIONS =====
def get_center(box):
    if box is None or len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def euclidean_distance_center(box1, box2):
    center1 = get_center(box1)
    center2 = get_center(box2)
    if center1 is None or center2 is None:
        return float('inf')
    return np.linalg.norm(center1 - center2)

# ===== BALL FEATURE FUNCTIONS =====
def calculate_ball_speed(ball_positions, fps):
    if len(ball_positions) < 2:
        return 0
    speeds = []
    for i in range(1, len(ball_positions)):
        if ball_positions[i] is not None and ball_positions[i-1] is not None:
            dx = ball_positions[i][0] - ball_positions[i-1][0]
            dy = ball_positions[i][1] - ball_positions[i-1][1]
            distance = np.sqrt(dx**2 + dy**2)
            speeds.append(distance * fps)
    return np.mean(speeds) if speeds else 0

def calculate_ball_angle(ball_positions):
    if len(ball_positions) < 2:
        return 0
    angles = []
    for i in range(1, len(ball_positions)):
        if ball_positions[i] is not None and ball_positions[i-1] is not None:
            dx = ball_positions[i][0] - ball_positions[i-1][0]
            dy = ball_positions[i][1] - ball_positions[i-1][1]
            angles.append(np.arctan2(dy, dx))
    return np.mean(angles) if angles else 0

# ===== PLAYER FEATURE FUNCTIONS =====
def calculate_distance_covered(player_positions):
    if len(player_positions) < 2:
        return 0
    total_distance = 0
    for i in range(1, len(player_positions)):
        if player_positions[i] is not None and player_positions[i-1] is not None:
            dx = player_positions[i][0] - player_positions[i-1][0]
            dy = player_positions[i][1] - player_positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
    return total_distance

def calculate_average_speed(player_positions, fps):
    distance = calculate_distance_covered(player_positions)
    if len(player_positions) < 2:
        return 0
    return (distance / len(player_positions)) * fps

def calculate_zone_coverage(player_positions, frame_height):
    if not player_positions:
        return {"back":0, "mid":0, "net":0}
    zone_counts = {"back":0, "mid":0, "net":0}
    for pos in player_positions:
        if pos is not None:
            y = pos[1]
            if y > frame_height * 0.7:
                zone_counts["back"] += 1
            elif y < frame_height * 0.3:
                zone_counts["net"] += 1
            else:
                zone_counts["mid"] += 1
    total = sum(zone_counts.values())
    if total > 0:
        return {zone: (count/total)*100 for zone, count in zone_counts.items()}
    else:
        return {"back":0, "mid":0, "net":0}

def calculate_serve_accuracy(serve_results):
    if not serve_results:
        return 0
    successful = sum(1 for r in serve_results if r == "success")
    return successful / len(serve_results)

def calculate_hit_distribution(actions):
    if not actions:
        return {}
    counts = defaultdict(int)
    for a in actions:
        if a and a != "No Action":
            counts[a] += 1
    total = sum(counts.values())
    return {k: v/total for k,v in counts.items()} if total>0 else {}

def count_actions(actions):
    counts = defaultdict(int)
    for a in actions:
        if a and a != "No Action":
            counts[a] +=1
    return dict(counts)

def calculate_unforced_errors(detailed_actions, distance_threshold=100):
    if not detailed_actions:
        return 0
    errors = 0
    for act in detailed_actions:
        if (act.get("distance_to_ball") is not None and 
            act.get("distance_to_ball") < distance_threshold and 
            act.get("action") == "No Action"):
            errors += 1
    return errors

# ===== MATCH SUMMARY =====
def get_player_with_most_actions(p1_actions, p2_actions):
    p1_count = len([a for a in p1_actions if a!="No Action"])
    p2_count = len([a for a in p2_actions if a!="No Action"])
    if p1_count > p2_count: return "p1"
    elif p2_count > p1_count: return "p2"
    else: return "equal"

def get_most_common_action(all_actions):
    if not all_actions: return "None"
    counts = defaultdict(int)
    for a in all_actions:
        if a and a!="No Action": counts[a]+=1
    return max(counts.items(), key=lambda x:x[1], default=("None",0))[0]

# ===== DATA EXTRACTION FUNCTIONS =====
def extract_tracking_data(tracking_data):
    """Extract ball and player positions from tracking JSON"""
    player_detections = tracking_data.get("player_detections", [])
    ball_detections = tracking_data.get("ball_detections", [])
    
    # Extract player positions
    player1_positions = []
    player2_positions = []
    
    for frame_data in player_detections:
        p1_box = frame_data.get("1")
        p2_box = frame_data.get("2")
        player1_positions.append(get_center(p1_box) if p1_box else None)
        player2_positions.append(get_center(p2_box) if p2_box else None)
    
    # Extract ball positions
    ball_positions = []
    for frame_data in ball_detections:
        ball_box = frame_data.get("1")
        ball_positions.append(get_center(ball_box) if ball_box else None)
    
    return ball_positions, player1_positions, player2_positions

def extract_action_data(action_data):
    """Extract action information from action recognition JSON"""
    p1_data = action_data.get("player_1", {})
    p2_data = action_data.get("player_2", {})
    
    # Extract action sequences
    p1_actions = []
    p2_actions = []
    
    p1_sequence = p1_data.get("action_sequence", [])
    p2_sequence = p2_data.get("action_sequence", [])
    
    for action in p1_sequence:
        p1_actions.append(action.get("action", "No Action"))
    
    for action in p2_sequence:
        p2_actions.append(action.get("action", "No Action"))
    
    # Extract serve results (assuming serve is successful if action is "Serve")
    serve_results_p1 = ["success" if action.get("action") == "Serve" else "fail" 
                       for action in p1_sequence if action.get("action") == "Serve"]
    serve_results_p2 = ["success" if action.get("action") == "Serve" else "fail" 
                       for action in p2_sequence if action.get("action") == "Serve"]
    
    return (p1_actions, p2_actions, serve_results_p1, serve_results_p2, 
            p1_sequence, p2_sequence)

# ===== MAIN ANALYSIS FUNCTION =====
def analyze_tennis_match(ball_positions, player1_positions, player2_positions, 
                        player1_actions, player2_actions, serve_results_p1, 
                        serve_results_p2, fps, frame_height, total_frames,
                        player1_detailed_actions=None, player2_detailed_actions=None,
                        distance_threshold=100):
    
    ball_avg_speed = calculate_ball_speed(ball_positions, fps)
    ball_avg_angle = calculate_ball_angle(ball_positions)

    p1_distance = calculate_distance_covered(player1_positions)
    p1_avg_speed = calculate_average_speed(player1_positions, fps)
    p1_zone_coverage = calculate_zone_coverage(player1_positions, frame_height)
    p1_serve_accuracy = calculate_serve_accuracy(serve_results_p1)
    p1_hit_distribution = calculate_hit_distribution(player1_actions)
    p1_action_counts = count_actions(player1_actions)
    p1_total_shots = len([a for a in player1_actions if a!="No Action"])
    p1_unforced_errors = calculate_unforced_errors(player1_detailed_actions, distance_threshold) if player1_detailed_actions else 0
    p1_active_frames = sum(1 for a in player1_detailed_actions if a.get("is_active", False)) if player1_detailed_actions else len([pos for pos in player1_positions if pos])

    p2_distance = calculate_distance_covered(player2_positions)
    p2_avg_speed = calculate_average_speed(player2_positions, fps)
    p2_zone_coverage = calculate_zone_coverage(player2_positions, frame_height)
    p2_serve_accuracy = calculate_serve_accuracy(serve_results_p2)
    p2_hit_distribution = calculate_hit_distribution(player2_actions)
    p2_action_counts = count_actions(player2_actions)
    p2_total_shots = len([a for a in player2_actions if a!="No Action"])
    p2_unforced_errors = calculate_unforced_errors(player2_detailed_actions, distance_threshold) if player2_detailed_actions else 0
    p2_active_frames = sum(1 for a in player2_detailed_actions if a.get("is_active", False)) if player2_detailed_actions else len([pos for pos in player2_positions if pos])

    player_with_most_actions = get_player_with_most_actions(player1_actions, player2_actions)
    most_common_action = get_most_common_action(player1_actions+player2_actions)

    report = {
        "match_summary": {
            "ball": {
                "average_speed": ball_avg_speed,
                "average_angle": ball_avg_angle
            },
            "from_json2": {
                "total_frames_processed": total_frames,
                "player_with_most_actions": player_with_most_actions,
                "most_common_action": most_common_action
            }
        },
        "players": {
            "p1": {
                "distance_covered": p1_distance,
                "average_speed": p1_avg_speed,
                "zone_coverage": p1_zone_coverage,
                "serve_accuracy": p1_serve_accuracy,
                "hit_distribution": p1_hit_distribution,
                "unforced_errors": p1_unforced_errors,
                "total_shots": p1_total_shots,
                "action_counts": p1_action_counts,
                "total_actions": p1_total_shots,
                "detailed_stats": {
                    "total_frames": total_frames,
                    "active_frames": p1_active_frames
                }
            },
            "p2": {
                "distance_covered": p2_distance,
                "average_speed": p2_avg_speed,
                "zone_coverage": p2_zone_coverage,
                "serve_accuracy": p2_serve_accuracy,
                "hit_distribution": p2_hit_distribution,
                "unforced_errors": p2_unforced_errors,
                "total_shots": p2_total_shots,
                "action_counts": p2_action_counts,
                "total_actions": p2_total_shots,
                "detailed_stats": {
                    "total_frames": total_frames,
                    "active_frames": p2_active_frames
                }
            }
        }
    }

    return report

# ===== WRAPPER FUNCTION TO PROCESS 2 JSON FILES =====
def process_json_files(tracking_json_path, actions_json_path, fps=30, frame_height=720, distance_threshold=100):
    # Load tracking data
    with open(tracking_json_path, "r") as f:
        tracking_data = json.load(f)
    
    # Load action data
    with open(actions_json_path, "r") as f:
        action_data = json.load(f)
    
    # Extract data from tracking JSON
    ball_positions, player1_positions, player2_positions = extract_tracking_data(tracking_data)
    total_frames = len(ball_positions)
    
    # Extract data from action JSON
    (player1_actions, player2_actions, serve_results_p1, 
     serve_results_p2, player1_detailed_actions, player2_detailed_actions) = extract_action_data(action_data)
    
    # Generate report
    report = analyze_tennis_match(
        ball_positions, player1_positions, player2_positions,
        player1_actions, player2_actions,
        serve_results_p1, serve_results_p2,
        fps, frame_height, total_frames,
        player1_detailed_actions, player2_detailed_actions,
        distance_threshold
    )
    output_json_path = f"D:\\NTI IB\\Tennis-analysis-project\\output_files\\finalreport{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    # Save report
    with open(output_json_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Report saved to {output_json_path}")
    return output_json_path
