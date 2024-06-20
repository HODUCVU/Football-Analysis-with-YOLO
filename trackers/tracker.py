from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')

from utils import get_center_of_bbox, get_bbox_width, get_bbox_height


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1:{"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
    
    def detect_frames(self, frames):
        # batch size
        bs = 20
        detections = []
        for i in range(0, len(frames), bs):
            detections_batch = self.model.predict(frames[i:i+bs], conf=0.1)
            detections += detections_batch
            # break
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Args:
            frames (_type_): _description_
            read_from_stub (bool, optional): _description_. Defaults to False.
            stub_path (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as F:
                tracks = pickle.load(F)
            return tracks
            
        
        detections = self.detect_frames(frames)
        tracks = {
            "players":[
                # {0:{'bbox':[0,1.1,2,3]}, 1:{'bbox':[0.1.1,2,3]}, ...},
                # {10:{'bbox':[0,1.1,2,3]}, 15:{'bbox':[0.1.1,2,3]}, ...},
            ],
            "referees":[],
            "ball":[]
        }
        for frame_num, detection in enumerate(detections):
            cls_name = detection.names #{0:player, 1:ball, ...}
            cls_name_inverse = {v:k for k, v in cls_name.items()} # {player:0. ball:1, ...}
            # print(cls_name_inverse)
            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Convert GoalKeeper to Player object
            for object_id, class_id in enumerate(detection_supervision.class_id):
                # print(object_id, class_id, sep=" | ")
                if cls_name[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_id] = cls_name_inverse["player"]
            # print(detection_supervision.class_id)
            # Track Object
            tracks_object_detection = self.tracker.update_with_detections(detection_supervision)
            # print(tracks_object_detection)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # tracks_object_detection -> [0] is bbox, [1] is mask, [2] is confidence, [3] is class_id, [4] is tracker_id, 
            # [5] is  data={'class_name': array(['player',..])}
            for frame_detection in tracks_object_detection:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_name_inverse["player"]:
                    tracks["players"][frame_num][track_id] = {'bbox':bbox}

                if cls_id == cls_name_inverse["referee"]:
                    tracks["referees"][frame_num][track_id] = {'bbox':bbox}
            # Track ball
            for frame_detecion in detection_supervision:
                bbox = frame_detecion[0].tolist()
                cls_id = frame_detecion[3]
                
                # only one ball
                if cls_id == cls_name_inverse["ball"]:
                    tracks["ball"][frame_num][1] = {'bbox':bbox}
                
            # print(tracks)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as F:
                pickle.dump(tracks, F)
                
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Args:
            frame (_type_): _description_
            bbox (_type_): _description_
            color (_type_): _description_
            track_id (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame, 
            center=(x_center,y2), 
            axes=(int(width), int(width*0.35)), 
            color=color,
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        # Draw track_id for player
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(x_center - rectangle_width//2) # move back to rectangle_width/2 from x_center
        x2_rect = int(x_center + rectangle_width//2)
        y1_rect = int((y2 - rectangle_height//2) + 10) # move to with "-"
        y2_rect = int((y2 + rectangle_height//2) + 10) # move down with "+"
        
        if track_id is not None:
            cv2.rectangle(frame,
                          (x1_rect, y1_rect),
                          (x2_rect, y2_rect),
                          color,
                          cv2.FILLED)
            # Set position id
            x1_id = x1_rect + 12
            if track_id >= 100:
                x1_id -= 10
            
            # image, text, point, font, size, color, thinkness
            cv2.putText(
                frame, 
                f"{track_id}", 
                (int(x1_id), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,0),
                2
            )
        return frame
    
    def draw_traingle(self, frame, bbox, color):
        """
        Args:
            frame (_type_): _description_
            bbox (_type_): _description_
            color (_type_): _description_

        Returns:
            _type_: _description_
        """
        y = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)
        
        traingle_points = np.array([
                            [x_center,y], 
                            [x_center - 10, y - 20], 
                            [x_center + 10, y - 20]
                        ])
        cv2.drawContours(frame, [traingle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [traingle_points], 0, (0, 0, 0), 2)
        
        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control_rate):
        # draw a semi-transparency rectangle
        overlap = frame.copy()
        cv2.rectangle(overlap, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlap, alpha, frame, 1 - alpha, 0, frame)
        
        team_ball_control_till_frame = team_ball_control_rate[:frame_num+1]
        # Get the number of team has ball control till frame
        team_1_ball_control = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_ball_control = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        team_1 = team_1_ball_control/(team_1_ball_control + team_2_ball_control)
        team_2 = 1 - team_1
        
        cv2.putText(
            frame, 
            f"Team 1 Ball control: {team_1*100:.2f}%", 
            (1400, 900), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,0,0), 3)
        cv2.putText(
            frame, 
            f"Team 2 Ball control: {team_2*100:.2f}%", 
            (1400, 950), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,0,0), 3)
        
        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control_rate):
        """
        Args:
            video_frames (_type_): _description_
            tracks (_type_): _description_

        Returns:
            _type_: _description_
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            # Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0,255,0))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player['bbox'], color)

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (255,255,100))
            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball['bbox'], (0,0,255))
                
            # Draw table team bal control rate
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control_rate)
            output_video_frames.append(frame)
            
        return output_video_frames