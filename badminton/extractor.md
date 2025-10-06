### How I created the dataset
Code: [extract_poses_with_sapiens.py](https://github.com/Vdamarla7/badminton/blob/main/badminton/extract_poses_with_sapiens.py)  
I use YOLO to find bounding boxes per frame, take the two largest boxes, run the Sapiens Pose model, and draw poses on the frame. The “two largest boxes” heuristic works well for VideoBadminton but may fail elsewhere (e.g., a non-player with a large box).

Some videos could not be read or had frame-count anomalies—see `exceptions.txt`.

**Future plans:** Clean mislabeled files, resolve exceptions, and train multiple shot-classification models on these poses.

<img width="584" alt="Screenshot 2025-06-12 at 2 57 15 PM" src="https://github.com/user-attachments/assets/ba6224d1-72e0-4d8d-8294-a016e3f938cb" />
