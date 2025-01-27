import os
import argparse

def ensure_file_system_in_correct_state(data_location, videos_subfolder, poses_subfolder):
    # Make sure data_location ends in a '/' else add it
    if data_location[-1] != '/':    
        data_location = data_location + '/'
    
    # Make sure the videos folder exists
    if not os.path.exists(data_location + videos_subfolder):
        raise FileNotFoundError(f"Subfolder {videos_subfolder} does not exist under {data_location}")
    videos_location = data_location + videos_subfolder
    if videos_location[-1] != '/':    
        videos_location = videos_location + '/'
    
    # Make sure the poses folder exists
    if not os.path.exists(data_location + poses_subfolder):
        os.makedirs(data_location + '/' + poses_subfolder)
    poses_location = data_location + poses_subfolder
    if poses_location[-1] != '/':    
        poses_location = poses_location + '/'

    # Make sure there are no spaces in the folder names under videos location
    for subfolder in os.listdir(videos_location):       
        if ' ' in subfolder:
            new_subfolder = subfolder.replace(' ', '_')
            os.rename(videos_location + subfolder, videos_location + new_subfolder)
            print(f'Renamed {videos_location + subfolder} to {videos_location + new_subfolder}')

    # Make sure there is a corrsponding poses folder for each video folder
    for subfolder in os.listdir(videos_location):      
        if not os.path.exists(poses_location + subfolder):
            os.makedirs(poses_location + subfolder)
            
    return data_location, videos_location, poses_location

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files for pose estimation.")
    parser.add_argument('--data_location', nargs='+', help='Location of the dataset', required=True)
    parser.add_argument('--videos_subfolder', nargs='+', help='Location of the dataset', default='videos')
    parser.add_argument('--poses_subfolder', nargs='+', help='Location of the dataset', default='poses')
    
    args = parser.parse_args()

    # Extract the arguments
    data_location = args.data_location[0]
    videos_subfolder = args.videos_subfloder[0]
    poses_subfolder = args.poses_subfolder[0]

    # Ensure the file system is in the correct state
    data_location, videos_location, poses_location  = ensure_file_system_in_correct_state(data_location, videos_subfolder, poses_subfolder)

