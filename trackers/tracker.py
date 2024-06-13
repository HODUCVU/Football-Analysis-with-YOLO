from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        # batch size
        bs = 20
        detections = []
        for i in range(0, len(frames), bs):
            detections_batch = self.model.predict(frames[i:i+bs], conf=0.1)
            detections += detections_batch
            # break
        return detections
    
    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        
        for frame_num, detection in enumerate(detections):
            cls_name = detection.name #{0:player, 1:ball, ...}
            cls_name_inverse = {v:k for k, v in cls_name.item()} # {player:0. ball:1, ...}
            
            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # print(detection_supervision)
            