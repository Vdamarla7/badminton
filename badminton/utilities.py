import numpy as np
import cv2

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / frames_per_second
    ret, first_frame = cap.read()
    if ret:
        frame_shape = first_frame.shape
    else:
        frame_shape = None

    cap.release()
    return frame_count, frames_per_second, duration, frame_shape


def get_video_frames(video_path):
    frames = []
    frame_count, frames_per_second, duration, frame_shape = get_video_metadata(video_path)

    cap = cv2.VideoCapture(video_path)    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frame_count, frames_per_second, duration, frame_shape, frames

def find_bboxes_closest_to_center(bboxes, center, num_bboxes=2):
    # Calculate the midpoints of the bounding boxes
    midpoints = [(bbox[2] + bbox[0]) // 2 for bbox in bboxes]
    differences = [abs(midpoint - center) for midpoint in midpoints]
    # Get the indices of the two bounding boxes with the smallest differences
    closest_indices = np.argsort(differences)[:num_bboxes]

    # Get the bounding boxes
    closest_bboxes = bboxes[closest_indices]
    return closest_bboxes