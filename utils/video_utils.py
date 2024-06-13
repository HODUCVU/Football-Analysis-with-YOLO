import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        # Wait for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    return frames
    