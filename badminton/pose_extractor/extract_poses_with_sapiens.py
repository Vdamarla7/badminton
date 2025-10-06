import os
import csv

import torch
import argparse

from New.PosesArchive.coco_keypoints import (
    COCO_HEADERS,
    keypoints_to_list
)

from New.PosesArchive.detectors import SapiensPoseEstimation    
from file_utilities import ensure_file_system_in_correct_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files for pose estimation.")
    parser.add_argument('--data_location', nargs='+', help='Location of the dataset', required=True)
    parser.add_argument('--videos_subfolder', nargs='+', help='Location of the dataset', default='videos')
    parser.add_argument('--poses_subfolder', nargs='+', help='Location of the dataset', default='poses')
    parser.add_argument('--folders', nargs='+', help='List of folders to process', required=True)
    
    args = parser.parse_args()

    # Extract the arguments
    data_location = args.data_location[0]
    videos_subfolder = args.videos_subfolder
    poses_subfolder = args.poses_subfolder

    # Ensure the file system is in the correct state
    data_location, videos_location, poses_location  = ensure_file_system_in_correct_state(data_location, videos_subfolder, poses_subfolder)

    # Verify that the subfolders in args.folders exist under video_location
    for subfolder in args.folders:
        if not os.path.exists(videos_location + subfolder):
            raise FileNotFoundError(f"Subfolder {subfolder} does not exist under {videos_location}")
        
    # Initialize the pose estimator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_estimator = SapiensPoseEstimation()
    
    # Get all files in subfolders
    for subfolder in args.folders:
        if subfolder[-1] != '/':    
            subfolder = subfolder + '/'
            
        files = []
        for f in os.listdir(videos_location + subfolder):
            files.append([subfolder, f])

    print('Processing files: ' + str(len(files)))

    for f in files:
        mp4_file = videos_location + f[0] + f[1]
        csv_file = poses_location + f[0] + f[1].replace('.mp4', '.csv')

        # Check if the csv file already exists
        if os.path.exists(csv_file):
            print(f'{csv_file} already exists, skipping {mp4_file}')
            continue
        
        bboxes_list, closest_bboxes_list, keypoints_list = pose_estimator.process_video(mp4_file)
    
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(COCO_HEADERS)
            for i, bboxes, closest_bboxes, keypoints in enumerate(zip(bboxes_list, closest_bboxes_list, keypoints_list)):
                if len(closest_bboxes) == 2:
                    values = [i] + [closest_bboxes[0].tolist()] + [closest_bboxes[1].tolist()] + keypoints_to_list(keypoints[0]) + keypoints_to_list(keypoints[1])
                    file.writerow(values)

    # Clean up  the pose estimator
    del pose_estimator
    torch.cuda.empty_cache()
    
    
