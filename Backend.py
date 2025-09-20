from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import tempfile
import os
import yt_dlp
import cv2
import uuid

app = FastAPI(title="Tennis Player Recommendations with Gemini API")

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Gemini API settings ====
GEMENAI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMENAI_API_KEY = "AIzaSyBQ9p2Vditi10I4kEBg1o8Jbq2XqJacdrg"

# convert pixels to meters (1 pixel = 0.011885 meters)
METERS_PER_PIXEL = 0.011885


# Function to download YouTube video
def download_youtube_video(youtube_url):
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.close()

        ydl_opts = {
            'format': 'best[height<=720]',  # Download 720p or lower
            'outtmpl': temp_file.name,
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        return temp_file.name
    except Exception as e:
        raise Exception(f"Failed to download YouTube video: {str(e)}")


# Function to process video (placeholder - implement your actual video processing logic)
def process_video_file(video_path):
    """
    Placeholder function for video processing.
    In a real implementation, this would extract features from the video.
    """
    try:
        # For demonstration, we'll just create a mock analysis
        # In a real implementation, you would:
        # 1. Extract frames from the video
        # 2. Detect players and ball
        # 3. Track movements
        # 4. Calculate metrics

        # Mock processing - just count frames
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Create mock analysis data similar to the provided JSON structure
        mock_analysis = {
            "match_summary": {
                "ball": {
                    "average_speed": 627.3323024629406 * METERS_PER_PIXEL,
                    "average_angle": -0.1031901259004585
                },
                "from_json2": {
                    "total_frames_processed": frame_count,
                    "player_with_most_actions": "1",
                    "most_common_action": "Volley"
                }
            },
            "players": {
                "p1": {
                    "distance_covered": 16341.308819090176 * METERS_PER_PIXEL,
                    "average_speed": 2301.5927914212302 * METERS_PER_PIXEL,
                    "zone_coverage": {
                        "back": 58.87850467289719,
                        "mid": 22.897196261682243,
                        "net": 18.22429906542056
                    },
                    "serve_accuracy": 0.7272727272727273,
                    "hit_distribution": {
                        "Volley": 0.9485981308411215,
                        "Serve": 0.0514018691588785
                    },
                    "unforced_errors": 0,
                    "total_shots": 214,
                    "action_counts": {
                        "Volley": 203,
                        "Serve": 11
                    },
                    "total_actions": 214,
                    "detailed_stats": {
                        "total_frames": 214,
                        "active_frames": 46
                    }
                },
                "p2": {
                    "distance_covered": 6093.044855482349 * METERS_PER_PIXEL,
                    "average_speed": 858.1753317580366 * METERS_PER_PIXEL,
                    "zone_coverage": {
                        "back": 0.46728971962616817,
                        "mid": 0.46728971962616817,
                        "net": 99.06542056074767
                    },
                    "serve_accuracy": 1.0,
                    "hit_distribution": {
                        "Volley": 0.9626168224299065,
                        "Forehand": 0.028037383177570093,
                        "Serve": 0.009345794392523364
                    },
                    "unforced_errors": 0,
                    "total_shots": 214,
                    "action_counts": {
                        "Volley": 206,
                        "Forehand": 6,
                        "Serve": 2
                    },
                    "total_actions": 214,
                    "detailed_stats": {
                        "total_frames": 214,
                        "active_frames": 64
                    }
                }
            }
        }

        return mock_analysis
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to Tennis Player Recommendation API!"}


@app.post("/process-video")
async def process_video_endpoint(
        file: UploadFile = File(None),
        youtube_url: str = Form(None)
):
    try:
        video_path = None

        if file:
            # Save uploaded file to temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            video_path = temp_file.name
        elif youtube_url:
            # Download YouTube video
            video_path = download_youtube_video(youtube_url)
        else:
            return JSONResponse(
                content={"error": "Either file or youtube_url must be provided"},
                status_code=400
            )

        # Process the video
        analysis_data = process_video_file(video_path)

        # Clean up temporary file
        if os.path.exists(video_path):
            os.unlink(video_path)

        # Get recommendations from Gemini API
        results = {}
        for player_id, player_data in analysis_data["players"].items():
            # Create a detailed prompt with match context
            prompt = f"""
            Analyze this tennis player's performance data and provide specific recommendations for improvement.

            Player ID: {player_id}
            Distance Covered: {player_data.get('distance_covered', 'unknown')} meters
            Average Speed: {player_data.get('average_speed', 'unknown')} m/s
            Zone Coverage: Back: {player_data.get('zone_coverage', {}).get('back', 'unknown')}%, 
                          Mid: {player_data.get('zone_coverage', {}).get('mid', 'unknown')}%, 
                          Net: {player_data.get('zone_coverage', {}).get('net', 'unknown')}%
            Serve Accuracy: {player_data.get('serve_accuracy', 'unknown')}
            Hit Distribution: {player_data.get('hit_distribution', 'unknown')}
            Unforced Errors: {player_data.get('unforced_errors', 'unknown')}
            Total Shots: {player_data.get('total_shots', 'unknown')}

            Match Context:
            - Ball average speed: {analysis_data.get('match_summary', {}).get('ball', {}).get('average_speed', 'unknown')} m/s
            - Total frames processed: {analysis_data.get('match_summary', {}).get('from_json2', {}).get('total_frames_processed', 'unknown')}

            Please provide specific, actionable recommendations for this player to improve their performance.
            """

            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": GEMENAI_API_KEY
            }
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }

            try:
                response = requests.post(GEMENAI_URL, headers=headers, json=payload)
                response.raise_for_status()
                gemeni_response = response.json()

                # extract the text response
                candidates = gemeni_response.get("candidates", [])
                if candidates:
                    text = candidates[0]["content"]["parts"][0]["text"]
                    results[player_id] = text
                else:
                    results[player_id] = "No recommendation returned."
            except requests.exceptions.RequestException as e:
                results[player_id] = f"Error calling Gemini API: {e}"

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to process video: {str(e)}"},
            status_code=500
        )


@app.post("/recommendations")
async def get_recommendations(file: UploadFile = File(...)):
    try:
        # read JSON
        content = await file.read()
        data = json.loads(content)

        if "players" not in data:
            return JSONResponse(content={"error": "JSON must contain 'players' key"}, status_code=400)

        # ==== convert data to meters ====
        for player_id, player in data["players"].items():
            # Convert distance and speed from pixels to meters
            if player.get("distance_covered") is not None:
                player["distance_covered"] *= METERS_PER_PIXEL
            if player.get("average_speed") is not None:
                player["average_speed"] *= METERS_PER_PIXEL

            # Handle missing or empty data
            for key in ["serve_accuracy", "unforced_errors", "hit_distribution"]:
                if player.get(key) is None or player.get(key) == {}:
                    player[key] = "unknown"

        # Convert ball speed to meters
        if "match_summary" in data and "ball" in data["match_summary"]:
            if data["match_summary"]["ball"].get("average_speed") is not None:
                data["match_summary"]["ball"]["average_speed"] *= METERS_PER_PIXEL

        results = {}

        # ==== send each player to Gemini API ====
        for player_id, player_data in data["players"].items():
            # Create a more detailed prompt with match context
            prompt = f"""
            Analyze this tennis player's performance data and provide specific recommendations for improvement.

            Player ID: {player_id}
            Distance Covered: {player_data.get('distance_covered', 'unknown')} meters
            Average Speed: {player_data.get('average_speed', 'unknown')} m/s
            Zone Coverage: Back: {player_data.get('zone_coverage', {}).get('back', 'unknown')}%, 
                          Mid: {player_data.get('zone_coverage', {}).get('mid', 'unknown')}%, 
                          Net: {player_data.get('zone_coverage', {}).get('net', 'unknown')}%
            Serve Accuracy: {player_data.get('serve_accuracy', 'unknown')}
            Hit Distribution: {player_data.get('hit_distribution', 'unknown')}
            Unforced Errors: {player_data.get('unforced_errors', 'unknown')}
            Total Shots: {player_data.get('total_shots', 'unknown')}

            Match Context:
            - Ball average speed: {data.get('match_summary', {}).get('ball', {}).get('average_speed', 'unknown')} m/s
            - Total frames processed: {data.get('match_summary', {}).get('from_json2', {}).get('total_frames_processed', 'unknown')}

            Please provide specific, actionable recommendations for this player to improve their performance.
            """

            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": GEMENAI_API_KEY
            }
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }

            try:
                response = requests.post(GEMENAI_URL, headers=headers, json=payload)
                response.raise_for_status()
                gemeni_response = response.json()

                # extract the text response
                candidates = gemeni_response.get("candidates", [])
                if candidates:
                    text = candidates[0]["content"]["parts"][0]["text"]
                    results[player_id] = text
                else:
                    results[player_id] = "No recommendation returned."
            except requests.exceptions.RequestException as e:
                results[player_id] = f"Error calling Gemini API: {e}"

        return JSONResponse(content=results)

    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid JSON file"}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)