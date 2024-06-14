from utils import read_video, save_video
from trackers import Tracker

def main():
    # Get frames from video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    # Tracker object
    model_path = 'models/best.pt'
    trakcer = Tracker(model_path)
    stub_path = "stubs/track_stubs.pkl"
    tracks = trakcer.get_object_tracks(video_frames, read_from_stub=True, stub_path=stub_path)
    
    # Draw bbox for output frames
    video_frames_output = trakcer.draw_annotations(video_frames=video_frames, tracks=tracks)
    # Save video
    output_video_path = 'output_videos/output_video.avi'
    save_video(video_frames_output, output_video_path)
    
if __name__ == '__main__':
    main() 