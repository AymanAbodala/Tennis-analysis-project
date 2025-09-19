import cv2
import yt_dlp

def read_video(video_path):
    """
    Reads a video from a file, returning all frames as a list of numpy arrays

    Parameters
    ----------
    video_path : str
        path to the video file

    Returns
    -------
    list
        list of frames, where each frame is a numpy array of shape (height, width, 3)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def convert_to_mp4(url):
    """
    Converts a YouTube video URL to a direct MP4 URL

    Parameters
    ----------
    url : str
        path to the YouTube video

    Returns
    -------
    str
        direct URL to the MP4 video
    """
    ydl_opts = {'format': 'best[ext=mp4]'}  
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        direct_url = info['url']

    return direct_url


def save_video(output_video_frames, output_video_path):
    """
    Saves a list of frames as a video to a file

    Parameters
    ----------
    output_video_frames : list
        list of frames, where each frame is a numpy array of shape (height, width, 3)
    output_video_path : str
        path to the output video file

    Returns
    -------
    None
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()