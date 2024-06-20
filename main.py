from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssginer
import numpy as np
import argparse

def main(video_input_path='input_videos/08fd33_4.mp4', model_path = 'models/best.pt', output_video_path = 'output_videos/output_video.avi'):
    # Get frames from video
    video_frames = read_video(video_input_path)
    
    # Tracker object
    tracker = Tracker(model_path)
    
    stub_path = "stubs/track_stubs.pkl"
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True, 
                                       stub_path=stub_path)
    
    # Interpolate ball position
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])
    
    # Assign ball Aquisition
    player_id_ball_assigner = PlayerBallAssginer()
    
    # Ball control rate
    team_ball_control_rate = []
    
    # Assign player team with first frame
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frame=video_frames[frame_num], 
                                                 player_bbox=track['bbox'], 
                                                 player_id=player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team] # Assign team color to player

        # Assign player to ball
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assignerPlayer = player_id_ball_assigner.assign_player_to_ball(player_track, ball_bbox)
        
        if assignerPlayer is not None:
            tracks['players'][frame_num][assignerPlayer]['has_ball'] = True
            team_ball_control_rate.append(tracks['players'][frame_num][assignerPlayer]['team']) # 1 for team 1, 2 for team 2
        else:
            # Get last team ball control rate
            team_ball_control_rate.append(team_ball_control_rate[-1])     
            
    team_ball_control_rate = np.array(team_ball_control_rate)
    
    
    # Draw bbox for output frames
    video_frames_output = tracker.draw_annotations(video_frames=video_frames, tracks=tracks, team_ball_control_rate=team_ball_control_rate)
    # Save video
    save_video(video_frames_output, output_video_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video for tracking.')
    parser.add_argument('--video_input_path', type=str, default='input_videos/08fd33_4.mp4', help='Path to the input video file')
    parser.add_argument('--model_path', type=str, default='models/best.pt', help='Path to the model')
    parser.add_argument('--video_output_path', type=str, default='output_videos/output_video.avi', help='Path to the model')
    
    args = parser.parse_args()
    main(args.video_input_path, args.model_path, args.video_output_path)