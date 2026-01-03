import io
import csv
import cv2
import imageio
import os
import numpy as np
from .coco_keypoints import (
    COCO_KPTS_COLORS,
    COCO_SKELETON_INFO,
    create_keypoints_dict
)

from .poselet_classifier import classify_triplet

from .utilities import get_video_metadata, get_video_frames

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


def draw_boundingbox(img, bbox, color, thickness=2):
    draw_img = img.copy()
    x1, y1, x2, y2 = map(int, bbox[:4])
    draw_img = cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
    return draw_img


def draw_keypoints(img, bbox, keypoints, color, confidence=0):
    x1, y1, x2, y2 = map(int, bbox[:4])
    bbox_width, bbox_height = x2 - x1, y2 - y1
    img_copy = img.copy()

    # Draw keypoints on the image
    for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
        if conf > confidence:  # Only draw confident keypoints
            x_coord = int(x * bbox_width / 192) + x1
            y_coord = int(y * bbox_height / 256) + y1
            cv2.circle(img_copy, (x_coord, y_coord), 3, color, -1)

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
                cv2.line(img_copy, (x1_coord, y1_coord), (x2_coord, y2_coord), color, 2)

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
            playera_keypoints = create_keypoints_dict(list(map(float, row[9:60])))
            playerb_keypoints = create_keypoints_dict(list(map(float, row[60:])))
            
            self.playera.append((playera_bbox, playera_keypoints))
            self.playerb.append((playerb_bbox, playerb_keypoints))

    def load_video_frames(self):
        if self.video_frames == None:
            self.video_frames = get_video_frames(self.video_path)[4]

    def __len__(self):
        return len(self.poses_data)

    def annotate_frame_with_poses(self, frame_idx, include_background, include_bboxes, players):
        self.load_video_frames()
        frame_width, frame_height = self.frame_shape[1], self.frame_shape[0]
        if include_background:
            frame = self.video_frames[frame_idx]
        else:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Draw player A on frame[frame_idx]
        if 'green' in players:
            color = (0, 255, 0)
            bbox_a, keypoints_a = self.playera[frame_idx]
            if include_bboxes:
                frame = draw_boundingbox(frame, bbox_a, color=color)
            frame = draw_keypoints(frame, bbox_a, keypoints_a, color=color, confidence=0.0)

        # Draw player B on frame[frame_idx]
        if 'blue' in players:
            color = (255, 0, 0)
            bbox_b, keypoints_b = self.playerb[frame_idx]
            if include_bboxes:
                frame = draw_boundingbox(frame, bbox_b, color=color)
            frame = draw_keypoints(frame, bbox_b, keypoints_b, color=color, confidence=0.0)
        
        return frame

    def annotate_video_with_poses(self, output_path, include_background=True, 
                                  include_bboxes = True, players = ['green', 'blue']):
        # Get video properties
        self.load_video_frames()

        # Create video writer
        frame_width, frame_height = self.frame_shape[1], self.frame_shape[0]                
        
        with imageio.get_writer(output_path, fps=30) as writer:
            for i in range(len(self.video_frames)):
                frame = self.annotate_frame_with_poses(i, include_background, include_bboxes, players)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.append_data(frame)
   
    def get_poselets_for_player(self, player='green'):
        """
        Extract poslet features for the specified player ('green' or 'blue').

        :param player: 'green' for player A, 'blue' for player B
        :return: List of dictionaries containing poslet features for each frame
        """
        extracted_features = []
        count = 0
        if player == 'green':
            player = self.playera
        else:
            player = self.playerb
        
        # the multiply by -1 is for the y axis inversion in the image coordinates
        for r in player:
            row = r[1]
            new_row = {}
            left_ankle = (row['left_ankle'][0], -1 * row['left_ankle'][1])
            left_knee = (row['left_knee'][0], -1 * row['left_knee'][1])
            left_hip = (row['left_hip'][0], -1 * row['left_hip'][1])
            left_shoulder = (row['left_shoulder'][0], -1 * row['left_shoulder'][1])
            left_elbow = (row['left_elbow'][0], -1 * row['left_elbow'][1])
            left_wrist = (row['left_wrist'][0], -1 * row['left_wrist'][1])
            right_wrist = (row['right_wrist'][0], -1 * row['right_wrist'][1])
            right_elbow = (row['right_elbow'][0], -1 * row['right_elbow'][1])
            right_shoulder = (row['right_shoulder'][0], -1 * row['right_shoulder'][1])
            right_hip = (row['right_hip'][0], -1 * row['right_hip'][1])
            right_knee = (row['right_knee'][0], -1 * row['right_knee'][1])
            right_ankle = (row['right_ankle'][0], -1 * row['right_ankle'][1])

            left_arm_orientation = classify_triplet(left_shoulder, left_elbow, left_wrist)
            left_leg_orientation = classify_triplet(left_hip, left_knee, left_ankle)
            left_torso_orientation = classify_triplet(left_knee, left_hip, left_shoulder)
            
            # Classify the right arm, leg, and torso orientations    
            right_arm_orientation = classify_triplet(right_shoulder, right_elbow, right_wrist)
            right_leg_orientation = classify_triplet(right_hip, right_knee, right_ankle)
            right_torso_orientation = classify_triplet(right_knee, right_hip, right_shoulder)

            new_row['left_arm'] = left_arm_orientation
            new_row['left_leg'] = left_leg_orientation
            new_row['left_torso'] = left_torso_orientation
            new_row['right_arm'] = right_arm_orientation
            new_row['right_leg'] = right_leg_orientation
            new_row['right_torso'] = right_torso_orientation
            extracted_features.append(new_row)
            count +=1
            
        return extracted_features
   
   
    def get_shot_description_for_player(self, player='green'):
        """
        Generate a textual description of the player's pose for each frame.

        :param player: 'green' for player A, 'blue' for player B
        :return: List of strings describing the player's pose for each frame
        """

        pose_description = ""
        
        if player == 'green':
            player = self.playera
        else:
            player = self.playerb
            
        # This will need to be rewritten.
        # I need to extract the loaction based on court location and it will depend on which player is shooting.
        # for right now estimating based on shot_type

        shot_type = self.poses_path.split("/")[-2]
        
        
        
        Position: {FrontCourt, ServeLine, MidCourt, BackCourt}
    
        if shot_type == "00_Short_Serve":
            location = "ServeLine"
        elif shot_type == "13_Long_Serve":
            location = "ServeLine"
        elif shot_type == "14_Smash":
            location = "BackCourt or MidCourt"
        elif shot_type == "05_Drop":
            location = "FrontCourt"
        elif shot_type == "07_Transitional_Slice":
            location = "BackCourt"
        elif shot_type == "16_Rear_Court_Flat_Drive":
            location = "BackCourt"
        else:
            location = "MidCourt"

        poselets = self.get_poselets_for_player(player=player)
                # print extracted features to a string wiht headers and each rows as a cs

        output = io.StringIO()
        fieldnames = ['Frame', 'left_arm', 'left_leg', 'left_torso', 'right_arm', 'right_leg', 'right_torso']

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        # add a row number column

        for i, row in enumerate(poselets):
            row['Frame'] = i + 1
            writer.writerow(row)

        prompt = "Position: " + location + "\n" + output.getvalue()
        return prompt