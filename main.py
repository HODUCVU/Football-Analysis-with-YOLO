from utils import read_video, save_video
from trackers import Tracker

def main():
    # Get frames from video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    # Tracker object
    model_path = 'models/best.pt'
    Trakcer = Trakcer(model_path)
    
    tracker = Tracker.get_object_tracks(video_frames)
    # Save video
    output_video_path = 'output_videos/output_video.avi'
    # save_video(video_frames, output_video_path)
    
if __name__ == '__main__':
    main() 