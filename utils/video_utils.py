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

def save_video(output_video_frames, output_video_path):
    # Get the video properties
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
