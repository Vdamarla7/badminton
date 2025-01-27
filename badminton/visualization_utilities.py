import csv
import cv2
import imageio
import os
import numpy as np
from coco_keypoints import (
    COCO_KPTS_COLORS,
    COCO_SKELETON_INFO,
    create_keypoints_dict
)

from utilities import get_video_metadata, get_video_frames

def replace_mp4_with_csv(filename):
    """
    Removes the .mp4 extension (if present) and replaces it with .csv.
    Returns the new filename (doesn't rename any actual file on disk).
    """
    # Split filename into 'name' and 'ext'
    name, ext = os.path.splitext(filename)
    
    if ext.lower() == '.mp4':
        return f"{name}.csv"
    else:
        # If it's not an .mp4 file, return the original
        return filename
def list_video_files(folder_path, video_extensions=None):
    """
    Prints the directory name right above each video file,
    followed by the filename itself.
    
    :param folder_path: The starting directory to search within.
    :param video_extensions: A list of file extensions considered as video files.
    """
    if video_extensions is None:
        # Adjust or expand the list of extensions as needed
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv']

    # Walk through the folder and its subfolders
    filesList = []
    for root, dirs, files in os.walk(folder_path):
        print("hi")
        for file in files:
            # Check if the file has one of the desired video extensions
            if any(file.lower().endswith(ext) for ext in video_extensions):
                # Get the immediate folder name that contains the file
                parent_dir = os.path.basename(root)
                # Join the parent directory name and the file name using the OS separator
                relative_path = os.path.join(parent_dir, file)
                cap = cv2.VideoCapture(os.path.join(root, file))
                ret, frame = cap.read()
                if not ret:
                    continue
                filesList.append(relative_path)
                
    return(filesList)

def crop_image(img, bbox):
    x1, y1, x2, y2 = map(int, bbox[:4])
    return img[y1:y2, x1:x2]


def draw_boundingbox(img, bbox, color=(0, 255, 0), thickness=2):
    draw_img = img.copy()
    x1, y1, x2, y2 = map(int, bbox[:4])
    draw_img = cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
    return draw_img


def draw_keypoints(img, bbox, keypoints, confidence=0):
    x1, y1, x2, y2 = map(int, bbox[:4])
    bbox_width, bbox_height = x2 - x1, y2 - y1
    img_copy = img.copy()

    # Draw keypoints on the image
    for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
        if conf > confidence:  # Only draw confident keypoints
            x_coord = int(x * bbox_width / 192) + x1
            y_coord = int(y * bbox_height / 256) + y1
            cv2.circle(img_copy, (x_coord, y_coord), 3, COCO_KPTS_COLORS[i], -1)

    # Optionally draw skeleton
    for _, link_info in COCO_SKELETON_INFO.items():
        pt1_name, pt2_name = link_info['link']
        if pt1_name in keypoints and pt2_name in keypoints:
            pt1 = keypoints[pt1_name]
            pt2 = keypoints[pt2_name]
            if pt1[2] > confidence and pt2[2] > confidence:
                x1_coord = int(pt1[0] * bbox_width / 192) + x1
                y1_coord = int(pt1[1] * bbox_height / 256) + y1
                x2_coord = int(pt2[0] * bbox_width / 192) + x1
                y2_coord = int(pt2[1] * bbox_height / 256) + y1
                cv2.line(img_copy, (x1_coord, y1_coord), (x2_coord, y2_coord), COCO_KPTS_COLORS[i], 2)

    return img_copy



class VideoPoseDataset:
    def __init__(self, poses_path, video_path):
        self.poses_path = poses_path
        self.video_path = video_path
        self.frame_count, self.frames_per_second, self.duration, self.frame_shape = get_video_metadata(video_path)
        self.video_frames = None

        poses_data = []
        with open(poses_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header row if present
            for row in csvreader:
                poses_data.append(row)
        
        self.poses_data = poses_data

        self.playera = []
        self.playerb = []
        for row in poses_data:
            playera_bbox = list(map(float, row[1:5]))
            playerb_bbox = list(map(float, row[5:9]))
            playera_keypoints = create_keypoints_dict(list(map(float, row[8:59])))
            playerb_keypoints = create_keypoints_dict(list(map(float, row[59:110])))
            
            self.playera.append((playera_bbox, playera_keypoints))
            self.playerb.append((playerb_bbox, playerb_keypoints))

            
    def load_video_frames(self):
        if self.video_frames == None:
            self.video_frames = get_video_frames(self.video_path)[4]


    def __len__(self):
        return len(self.poses_data)


    def annotate_frame_with_poses(self, frame_idx, withBackground):
        self.load_video_frames()
        frame_width, frame_height = self.frame_shape[1], self.frame_shape[0]
        if withBackground:
            frame = self.video_frames[frame_idx]
        else:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Draw player A on frame[frame_idx]
        bbox_a, keypoints_a = self.playera[frame_idx]
        frame = draw_boundingbox(frame, bbox_a, color=(0,255,0))
        frame = draw_keypoints(frame, bbox_a, keypoints_a, confidence=0.0)
        
        # Draw player B on frame[frame_idx]
        bbox_b, keypoints_b = self.playerb[frame_idx]
        frame = draw_boundingbox(frame, bbox_b, color=(255,0,0))
        frame = draw_keypoints(frame, bbox_b, keypoints_b, confidence=0.0)
        return frame


    def annotate_video_with_poses(self, output_path, withBackground):
        
        # Get video properties
        self.load_video_frames()

        # Create video writer
        frame_width, frame_height = self.frame_shape[1], self.frame_shape[0]                
        
        with imageio.get_writer(output_path, fps=30) as writer:
            for i in range(len(self.video_frames)):
                frame = self.annotate_frame_with_poses(i, withBackground)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                print(frame)
                writer.append_data(frame)
          
def annotateData(dataPath, posePath, annotatedPath, withBackground):
    filesList = list_video_files(dataPath)
    for file in filesList:
        video_dataset = VideoPoseDataset(replace_mp4_with_csv(posePath+"/"+file), dataPath+"/"+file)
        # Try annotating and print the names of the files that it doesn't work on
        try:
            video_dataset.annotate_video_with_poses(annotatedPath+"/"+file, withBackground)
        except Exception as e:
            print(file)

annotateData("../PosesArchive/VideoBadminton_Dataset", "../PosesArchive/poses", "../PosesArchive/VideoBadminton_Dataset_Only_Poses",False)