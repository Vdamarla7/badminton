### How I created the dataset

Using the 2B parameter model we will: 
    find the objects
    the bounding boxes for the 2 people that have a center closest to the center of the court
    we will find the keypoints for each of the bounding boxes
        A will be the bounding box for the top person
        B will be the bounding box for the bottom person

The data structure is going to be: 
    Image name
    Image Location
    Frames[]
        Frame
            A bounding box
            B bounding box
            A keypoints
            B keypoints
    Analysis on the bounding boxes and the keypoints

Each line will include: 
    Frame
    ABBOX: 
    BBBOX: 
    AKEYPPINTS: 
    BKEYPOINTS: 


I use YOLO to find bounding boxes per frame, take the two largest boxes, run the Sapiens Pose model, and draw poses on the frame. The “two largest boxes” heuristic works well for VideoBadminton but may fail elsewhere (e.g., a non-player with a large box).


<img width="584" alt="Screenshot 2025-06-12 at 2 57 15 PM" src="https://github.com/user-attachments/assets/ba6224d1-72e0-4d8d-8294-a016e3f938cb" />

Code: [extract_poses_with_sapiens.py](https://github.com/Vdamarla7/badminton/blob/main/badminton/extract_poses_with_sapiens.py)  

**Future plans:** Clean mislabeled files, resolve exceptions, and train multiple shot-classification models on these poses.