
import sys 
sys.path.append("../")

from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_bbox_height



class PlayerBallAssginer:
    def __init__(self):
        self.max_player_ball_distance = 70
        
    def distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def assign_player_to_ball(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        
        # find closest player to ball
        closest_player_id = None
        closest_player_ball_distance = sys.maxsize
        
        for player_id, player in players.items():
            
            player_bbox = player['bbox']
            
            # measure distance from y2 player to ball
            player_ball_distance_left = self.distance((player_bbox[0],player_bbox[-1]), ball_position)
            player_ball_distance_right = self.distance((player_bbox[2],player_bbox[-1]), ball_position)
            
            player_ball_distance = min(player_ball_distance_left, player_ball_distance_right)
            
            if player_ball_distance < closest_player_ball_distance and player_ball_distance < self.max_player_ball_distance:
                closest_player_id = player_id
                closest_player_ball_distance = player_ball_distance
        
        return closest_player_id
