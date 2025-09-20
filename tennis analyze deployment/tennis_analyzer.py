import streamlit as st
from PIL import Image
import os
import base64
import requests
import json
import time
import re
import subprocess
import sys
from object_detection_Tracking import main_object_tracking
from action_recognition import run_pipeline
from processes_and_analysis import process_json_files

# Set page configuration
st.set_page_config(
    page_title="Tennis Match Analyzer",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# FastAPI endpoints
FASTAPI_URL = "http://localhost:8000/recommendations"  # For JSON analysis
FASTAPI_PROCESS_VIDEO = "http://localhost:8000/process-video"  # For video processing


# Install required dependencies from the GitHub repository
def install_dependencies():
    try:
        # Install requirements from the tennis-analysis project
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                               "https://raw.githubusercontent.com/AymanAbodala/Tennis-analysis-project/main/requirements.txt"])
        st.success("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        st.warning(f"Could not install dependencies automatically: {e}")
        st.info("Please manually install the requirements from the GitHub repository")


# Function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


# Function to add background image from local file
def add_bg_image_local():
    # Check if image exists
    if os.path.exists("Pickleball-Main.jpg"):
        encoded_string = get_image_base64("Pickleball-Main.jpg")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # Fallback to online image if local doesn't exist
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("https://images.unsplash.com/photo-1595231776511-ef4bbefc5a64");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )


# Function to send JSON to FastAPI and get recommendations
def get_recommendations_from_api(json_data):
    try:
        # Create a temporary file to send
        files = {"file": ("tennis_features.json", json.dumps(json_data), "application/json")}
        response = requests.post(FASTAPI_URL, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status code {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Failed to connect to API: {str(e)}"}


# Function to process video via FastAPI
def process_video(video_file=None, youtube_url=None):
    try:
        files = {}
        data = {}

        if video_file:
            files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
        elif youtube_url:
            data = {"youtube_url": youtube_url}

        response = requests.post(FASTAPI_PROCESS_VIDEO, files=files, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status code {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Failed to connect to API: {str(e)}"}


# Function to validate YouTube URL
def is_valid_youtube_url(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    match = re.match(youtube_regex, url)
    return bool(match)


# Function to clone and set up the tennis analysis project
def setup_tennis_analysis():
    try:
        # Clone the repository if it doesn't exist
        if not os.path.exists("Tennis-analysis-project"):
            subprocess.check_call(["git", "clone", "https://github.com/AymanAbodala/Tennis-analysis-project.git"])

        # Add the project to Python path
        sys.path.append("Tennis-analysis-project")

        st.success("Tennis analysis project setup successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to setup tennis analysis project: {e}")
        return False


# Custom CSS
def local_css():
    st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stApp {
        background-color: rgba(255, 255, 255, 0.2) !important;
        background-blend-mode: overlay;
    }

    .main-container {
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
    }

    .logo-container {
        display: flex;
        justify-content: space-between;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto 30px auto;
        padding: 20px;
    }

    .logo {
        width: 140px;
        height: 70px;
        background-color: white;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.2);
        padding: 10px;
    }

    .logo-img {
        max-width: 100%;
        max-height: 100%;
    }

    .logo span {
        color: #1a2a6c;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }

    .info-icon {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        transition: transform 0.3s ease;
    }

    .info-icon:hover {
        transform: scale(1.1);
    }

    .modal-content {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f);
        width: 90%;
        max-width: 600px;
        border-radius: 15px;
        padding: 30px;
        position: relative;
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.5);
        color: white;
    }

    .close-btn {
        position: absolute;
        top: 15px;
        right: 15px;
        font-size: 1.5rem;
        cursor: pointer;
        color: #fdbb2d;
    }

    .container {
        width: 100%;
        max-width: 1000px;
        display: flex;
        flex-direction: column;
        gap: 30px;
        margin: 100px auto 0 auto;
    }

    header {
        text-align: center;
        padding: 20px 0;
    }

    h1 {
        font-size: 2.8rem;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        color: white;
    }

    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        color: white;
    }

    .upload-section {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.5);
    }

    .upload-icon {
        font-size: 4rem;
        margin-bottom: 20px;
        color: #fdbb2d;
    }

    .upload-text {
        font-size: 1.5rem;
        margin-bottom: 25px;
        color: white;
    }

    .upload-btn {
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        border: none;
        padding: 15px 40px;
        font-size: 1.1rem;
        border-radius: 50px;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .upload-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    .upload-btn:active {
        transform: translateY(1px);
    }

    .smiley {
        display: block;
        margin-top: 15px;
        font-size: 1.5rem;
    }

    /* Results styling */
    .recommendation-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #fdbb2d;
        color: #000000; /* Black text color */
    }

    .player-title {
        color: #1a2a6c; /* Dark blue for player title */
        font-size: 1.5rem;
        margin-bottom: 10px;
    }

    /* Stats display */
    .stats-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        color: #000000;
    }

    .stats-title {
        color: #1a2a6c;
        font-size: 1.3rem;
        margin-bottom: 15px;
        border-bottom: 2px solid #fdbb2d;
        padding-bottom: 5px;
    }

    .stat-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        padding: 5px 0;
        border-bottom: 1px solid #eee;
    }

    .stat-label {
        font-weight: bold;
        color: #1a2a6c;
    }

    .stat-value {
        color: #333;
    }

    /* External Link Section */
    .external-link-section {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
    }

    .link-input {
        width: 100%;
        max-width: 500px;
        padding: 15px;
        border-radius: 50px;
        border: none;
        margin: 15px 0;
        background: rgba(255, 255, 255, 0.9);
        color: #1a2a6c;
        font-size: 1rem;
        text-align: center;
    }

    .link-input::placeholder {
        color: #666;
    }

    .link-btn {
        background: linear-gradient(to right, #00b4d8, #0077b6);
        border: none;
        padding: 15px 30px;
        font-size: 1.1rem;
        border-radius: 50px;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-top: 10px;
    }

    .link-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    .results-section {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 15px;
        padding: 30px;
        margin-top: 20px;
    }

    .results-title {
        font-size: 2rem;
        margin-bottom: 20px;
        color: #fdbb2d;
    }

    .sponsor {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00b4d8;
        margin-top: 30px;
    }

    .team-section {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 30px;
        margin-top: 40px;
    }

    .section-title {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 30px;
        color: #fdbb2d;
    }

    .team-members {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 25px;
    }

    .member {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        width: 220px;
        text-align: center;
        transition: transform 0.3s ease;
    }

    .member:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.25);
    }

    .member-img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto 15px;
        border: 3px solid #fdbb2d;
        background: #1a2a6c;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        color: #fdbb2d;
    }

    .member-name {
        font-size: 1.3rem;
        margin-bottom: 5px;
        color: white;
    }

    .member-role {
        color: #fdbb2d;
        margin-bottom: 10px;
        font-style: italic;
    }

    footer {
        margin-top: 50px;
        text-align: center;
        opacity: 0.7;
        font-size: 0.9rem;
        color: white;
    }

    /* Tabs styling */
    .tabs {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }

    .tab {
        padding: 10px 20px;
        margin: 0 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .tab.active {
        background: rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .tab:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    @media (max-width: 768px) {
        h1 {
            font-size: 2.2rem;
        }

        .logo {
            width: 110px;
            height: 55px;
        }

        .logo span {
            font-size: 1rem;
        }

        .upload-section {
            padding: 30px 20px;
        }

        .upload-text {
            font-size: 1.3rem;
        }

        .external-link-section {
            padding: 20px 15px;
        }

        .link-input {
            padding: 12px;
            font-size: 0.9rem;
        }

        .team-members {
            gap: 15px;
        }

        .member {
            width: 100%;
            max-width: 250px;
        }

        .tabs {
            flex-direction: column;
            align-items: center;
        }

        .tab {
            margin: 5px 0;
            width: 100%;
            max-width: 300px;
        }
    }

    /* Streamlit specific adjustments */
    .stButton > button {
        width: 100%;
    }

    .css-1d391kg {
        padding-top: 0;
    }

    /* Custom styling for Streamlit elements */
    .stFileUploader {
        width: 100%;
    }

    .stTextInput input {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1a2a6c !important;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# Main app
def main():
    add_bg_image_local()
    local_css()

    # Install dependencies and setup project
    if st.sidebar.button("Install Dependencies"):
        install_dependencies()

    if st.sidebar.button("Setup Tennis Analysis Project"):
        setup_tennis_analysis()

    # Logos - using local images if available
    nti_logo_html = ""
    mcit_logo_html = ""

    if os.path.exists("NTI-logo.png"):
        nti_encoded = get_image_base64("NTI-logo.png")
        nti_logo_html = f'<img src="data:image/png;base64,{nti_encoded}" class="logo-img" alt="NTI Logo">'
    else:
        nti_logo_html = '<span>NTI</span>'

    if os.path.exists("MCIT-logo.png"):
        mcit_encoded = get_image_base64("MCIT-logo.png")
        mcit_logo_html = f'<img src="data:image/png;base64,{mcit_encoded}" class="logo-img" alt="MCIT Logo">'
    else:
        mcit_logo_html = '<span>MCIT</span>'

    st.markdown(f"""
    <div class="logo-container">
        <div class="logo" id="NTI">
            {nti_logo_html}
        </div>
        <div class="logo" id="MCIT">
            {mcit_logo_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Info icon with JavaScript for modal
    st.markdown("""
    <div class="info-icon" id="infoIcon">
        ℹ
    </div>

    <div id="infoModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.8); z-index: 1001; align-items: center; justify-content: center;">
    </div>
    """, unsafe_allow_html=True)

    # Add JavaScript for modal functionality
    st.components.v1.html("""
    <script>
    function showModal() {
        document.getElementById("infoModal").style.display = "flex";
        document.getElementById("infoModal").innerHTML = `
            <div class="modal-content">
                <span class="close-btn" onclick="closeModal()">&times;</span>
                <h2>About Tennis Match Analyzer</h2>
                <p>This project is an advanced AI-powered tennis analysis system that provides players and coaches with detailed insights into match performance.</p>
                <br>
                <h3>Key Features:</h3>
                <ul>
                    <li>Analyze tennis matches from video files or YouTube links</li>
                    <li>Serve speed and accuracy tracking</li>
                    <li>Shot placement analysis</li>
                    <li>Movement and court coverage metrics</li>
                    <li>Stroke technique evaluation</li>
                    <li>Performance trends over time</li>
                </ul>
                <br>
                <p>The system uses computer vision and machine learning to analyze match footage and generate actionable insights for players of all levels.</p>
            </div>
        `;
    }

    function closeModal() {
        document.getElementById("infoModal").style.display = "none";
    }

    // Add event listeners
    document.getElementById("infoIcon").addEventListener("click", showModal);

    // Close modal when clicking outside content
    window.addEventListener("click", function(event) {
        const modal = document.getElementById("infoModal");
        if (event.target === modal) {
            closeModal();
        }
    });
    </script>
    """, height=0)

    # Main content
    st.markdown("""
    <div class="container">
        <header>
            <h1>Tennis Match Analyzer</h1>
            <p class="subtitle">Advanced AI-powered tennis performance analysis</p>
        </header>
    """, unsafe_allow_html=True)

    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'video'

    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Upload Video", "YouTube Link", "JSON Analysis"])

    with tab1:
        st.markdown("""
        <div class="upload-section">
            <div class="upload-icon">🎬</div>
            <p class="upload-text">Upload your tennis match video</p>
        </div>
        """, unsafe_allow_html=True)

        video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"], key="video_uploader",
                                      label_visibility="collapsed")

        if video_file is not None:
            st.video(video_file)

            if st.button("Analyze Video", key="analyze_video", use_container_width=True):
                st.session_state.processing = True
                with st.spinner("Processing video. This may take several minutes..."):
                    # Reset video file pointer to beginning
                    video_file.seek(0)
                    results = process_video(video_file=video_file)
                    st.session_state.results = results
                    st.session_state.processing = False

    with tab2:
        st.markdown("""
        <div class="external-link-section">
            <div class="upload-icon">📺</div>
            <p class="upload-text">Enter YouTube URL of tennis match</p>
        </div>
        """, unsafe_allow_html=True)

        youtube_url = st.text_input("YouTube URL", "", key="youtube_url",
                                    placeholder="https://www.youtube.com/watch?v=...",
                                    label_visibility="collapsed")

        if youtube_url:
            if is_valid_youtube_url(youtube_url):
                st.success("Valid YouTube URL detected!")

                if st.button("Analyze YouTube Video", key="analyze_youtube", use_container_width=True):
                    st.session_state.processing = True
                    with st.spinner("Processing YouTube video. This may take several minutes..."):
                        results = process_video(youtube_url=youtube_url)
                        st.session_state.results = results
                        st.session_state.processing = False
            else:
                st.error("Please enter a valid YouTube URL")

    with tab3:
        st.markdown("""
        <div class="upload-section">
            <div class="upload-icon">📁</div>
            <p class="upload-text">Upload your tennis features JSON file</p>
        </div>
        """, unsafe_allow_html=True)

        json_file = st.file_uploader("Upload JSON", type=["json"], key="json_uploader", label_visibility="collapsed")

        if json_file is not None:
            try:
                # Read and parse the JSON file
                json_data = json.load(json_file)
                st.session_state.analysis_data = json_data
                st.success("JSON file uploaded successfully!")

                # Display a button to analyze
                if st.button("Analyze Performance", key="analyze_json", use_container_width=True):
                    with st.spinner("Analyzing player performance..."):
                        # Send to FastAPI endpoint
                        results = get_recommendations_from_api(json_data)
                        st.session_state.results = results

            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please upload a valid JSON file.")

    # Display results if available
    if st.session_state.results:
        if "error" in st.session_state.results:
            st.error(f"Error: {st.session_state.results['error']}")
        else:
            st.markdown("""
            <div class="results-section">
                <h2 class="results-title">Performance Recommendations</h2>
            </div>
            """, unsafe_allow_html=True)

            # Display recommendations for each player
            for player_id, recommendation in st.session_state.results.items():
                if player_id.startswith("p"):
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h3 class="player-title">Player {player_id[1:]}</h3>
                        <p>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Display analysis data if available
    if st.session_state.analysis_data:
        st.markdown("""
        <div class="results-section">
            <h2 class="results-title">Match Statistics</h2>
        </div>
        """, unsafe_allow_html=True)

        # Display match summary
        if "match_summary" in st.session_state.analysis_data:
            st.markdown("""
            <div class="stats-container">
                <h3 class="stats-title">Match Summary</h3>
            </div>
            """, unsafe_allow_html=True)

            match_summary = st.session_state.analysis_data["match_summary"]
            if "ball" in match_summary:
                ball_stats = match_summary["ball"]
                st.markdown(f"""
                <div class="stats-container">
                    <h4 class="stats-title">Ball Statistics</h4>
                    <div class="stat-item">
                        <span class="stat-label">Average Speed:</span>
                        <span class="stat-value">{ball_stats.get('average_speed', 0):.2f} m/s</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Average Angle:</span>
                        <span class="stat-value">{ball_stats.get('average_angle', 0):.2f}°</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Display player statistics
        if "players" in st.session_state.analysis_data:
            for player_id, player_data in st.session_state.analysis_data["players"].items():
                st.markdown(f"""
                <div class="stats-container">
                    <h3 class="stats-title">Player {player_id[1:]} Statistics</h3>
                    <div class="stat-item">
                        <span class="stat-label">Distance Covered:</span>
                        <span class="stat-value">{player_data.get('distance_covered', 0):.2f} m</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Average Speed:</span>
                        <span class="stat-value">{player_data.get('average_speed', 0):.2f} m/s</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Serve Accuracy:</span>
                        <span class="stat-value">{player_data.get('serve_accuracy', 0) * 100:.2f}%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Unforced Errors:</span>
                        <span class="stat-value">{player_data.get('unforced_errors', 0)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Shots:</span>
                        <span class="stat-value">{player_data.get('total_shots', 0)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Team section - using Streamlit columns for proper layout
    st.markdown("""
    <div class="team-section">
        <h2 class="section-title">Our Team</h2>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for team members
    col1, col2, col3, col4, col5 = st.columns(5)

    # Check if images exist and create appropriate HTML
    ibrahim_img_html = ""
    ayman_img_html = ""

    if os.path.exists("ibrahim_yasser.jpg"):
        ibrahim_encoded = get_image_base64("ibrahim_yasser.jpg")
        ibrahim_img_html = f'<img src="data:image/jpg;base64,{ibrahim_encoded}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; margin: 0 auto 15px; border: 3px solid #fdbb2d;">'
    else:
        ibrahim_img_html = '<div class="member-img">👤</div>'

    if os.path.exists("ayman_abdallah.jpg"):
        ayman_encoded = get_image_base64("ayman_abdallah.jpg")
        ayman_img_html = f'<img src="data:image/jpg;base64,{ayman_encoded}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; margin: 0 auto 15px; border: 3px solid #fdbb2d;">'
    else:
        ayman_img_html = '<div class="member-img">👤</div>'

    if os.path.exists("A'laa_Omar.jpg"):
        Alaa_encoded = get_image_base64("A'laa_Omar.jpg")
        Alaa_img_html = f'<img src="data:image/jpg;base64,{Alaa_encoded}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; margin: 0 auto 15px; border: 3px solid #fdbb2d;">'
    else:
        Alaa_img_html = '<div class="member-img">👤</div>'

    if os.path.exists("Ahmed_Mohammed.jpg"):
        Ahmed_encoded = get_image_base64("Ahmed_Mohammed.jpg")
        Ahmed_img_html = f'<img src="data:image/jpg;base64,{Ahmed_encoded}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover; margin: 0 auto 15px; border: 3px solid #fdbb2d;">'
    else:
        Ahmed_img_html = '<div class="member-img">👤</div>'

    with col1:
        st.markdown(f"""
        <div class="member">
            {Ahmed_img_html}
            <h3 class="member-name">Ahmed Mohammed</h3>
            <p class="member-role">Developer</p>
            <p>Tracking</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="member">
            {ibrahim_img_html}
            <h3 class="member-name">Ibrahim Yasser</h3>
            <p class="member-role">Developer</p>
            <p>Deployment</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="member">
            {ayman_img_html}
            <h3 class="member-name">Ayman Abdallah</h3>
            <p class="member-role">Developer</p>
            <p>Object Detection</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="member">
            {Alaa_img_html}
            <h3 class="member-name">A'laa Omar</h3>
            <p class="member-role">Developer</p>
            <p>Data Processing</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="member">
            <div class="member-img">
                👤
            </div>
            <h3 class="member-name">A'laa Hegazy</h3>
            <p class="member-role">Developer</p>
            <p>Pose Estimation</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <footer>
        <p>© 2025 Tennis Match Analyzer | Privacy Policy | Terms of Service</p>
    </footer>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()