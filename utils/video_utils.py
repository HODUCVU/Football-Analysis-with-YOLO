import cv2

def read_video(video_path):
    # count_frame = 0
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        # if count_frame >= 10:
        #     break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        # count_frame += 1
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
