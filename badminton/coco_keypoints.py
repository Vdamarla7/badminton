# the 17 key points from the COCO dataset

COCO_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

COCO_KPTS_COLORS = [
    [51, 153, 255],   # 0: nose
    [51, 153, 255],   # 1: left_eye
    [51, 153, 255],   # 2: right_eye
    [51, 153, 255],   # 3: left_ear
    [51, 153, 255],   # 4: right_ear
    [0, 255, 0],      # 5: left_shoulder
    [255, 128, 0],    # 6: right_shoulder
    [0, 255, 0],      # 7: left_elbow
    [255, 128, 0],    # 8: right_elbow
    [0, 255, 0],      # 9: left_wrist
    [255, 128, 0],    # 10: right_wrist
    [0, 255, 0],      # 11: left_hip
    [255, 128, 0],    # 12: right_hip
    [0, 255, 0],      # 13: left_knee
    [255, 128, 0],    # 14: right_knee
    [0, 255, 0],      # 15: left_ankle
    [255, 128, 0],    # 16: right_ankle
]

COCO_SKELETON_INFO = {
        0: dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2: dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3: dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4: dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5: dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6: dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7: dict(link=('left_shoulder', 'right_shoulder'), id=7, color=[51, 153, 255]),
        8: dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11: dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12: dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255])
    }

def keypoints_to_list(keypoints):
    return [coord for kp in COCO_KEYPOINTS for coord in keypoints[kp]]

def create_keypoints_dict(values, kps = COCO_KEYPOINTS):
    keypoints = {}
    for i, (name, x, y, conf) in enumerate(zip(kps, values[1::3], values[2::3], values[3::3])):
        keypoints[name] = (float(x), float(y), float(conf))
    return keypoints

COCO_HEADERS = ['frame', 
'abb_xmin', 'abb_ymin', 'abb_xmax', 'abb_ymax', 
'bbb_xmin', 'bbb_ymin', 'bbb_xmax', 'bbb_ymax',
'a_nose_x', 'a_nose_y', 'a_nose_confidence',             
'a_left_eye_x', 'a_left_eye_y', 'a_left_eye_confidence',
'a_right_eye_x', 'a_right_eye_y', 'a_right_eye_confidence',
'a_left_ear_x', 'a_left_ear_y', 'a_left_ear_confidence',
'a_right_ear_x', 'a_right_ear_y', 'a_right_ear_confidence',
'a_left_shoulder_x', 'a_left_shoulder_y', 'a_left_shoulder_confidence',
'a_right_shoulder_x', 'a_right_shoulder_y', 'a_right_shoulder_confidence',
'a_left_elbow_x', 'a_left_elbow_y', 'a_left_elbow_confidence',
'a_right_elbow_x', 'a_right_elbow_y', 'a_right_elbow_confidence',
'a_left_wrist_x', 'a_left_wrist_y', 'a_left_wrist_confidence',
'a_right_wrist_x', 'a_right_wrist_y', 'a_right_wrist_confidence',
'a_left_hip_x', 'a_left_hip_y', 'a_left_hip_confidence',
'a_right_hip_x', 'a_right_hip_y', 'a_right_hip_confidence',
'a_left_knee_x', 'a_left_knee_y', 'a_left_knee_confidence',
'a_right_knee_x', 'a_right_knee_y', 'a_right_knee_confidence',
'a_left_ankle_x', 'a_left_ankle_y', 'a_left_ankle_confidence',
'a_right_ankle_x', 'a_right_ankle_y', 'a_right_ankle_confidence',
'b_nose_x', 'b_nose_y', 'b_nose_confidence',             
'b_left_eye_x', 'b_left_eye_y', 'b_left_eye_confidence',
'b_right_eye_x', 'b_right_eye_y', 'b_right_eye_confidence',
'b_left_ear_x', 'b_left_ear_y', 'b_left_ear_confidence',
'b_right_ear_x', 'b_right_ear_y', 'b_right_ear_confidence',
'b_left_shoulder_x', 'b_left_shoulder_y', 'b_left_shoulder_confidence',
'b_right_shoulder_x', 'b_right_shoulder_y', 'b_right_shoulder_confidence',
'b_left_elbow_x', 'b_left_elbow_y', 'b_left_elbow_confidence',
'b_right_elbow_x', 'b_right_elbow_y', 'b_right_elbow_confidence',
'b_left_wrist_x', 'b_left_wrist_y', 'b_left_wrist_confidence',
'b_right_wrist_x', 'b_right_wrist_y', 'b_right_wrist_confidence',
'b_left_hip_x', 'b_left_hip_y', 'b_left_hip_confidence',
'b_right_hip_x', 'b_right_hip_y', 'b_right_hip_confidence',
'b_left_knee_x', 'b_left_knee_y', 'b_left_knee_confidence',
'b_right_knee_x', 'b_right_knee_y', 'b_right_knee_confidence',
'b_left_ankle_x', 'b_left_ankle_y', 'b_left_ankle_confidence',
'b_right_ankle_x', 'b_right_ankle_y', 'b_right_ankle_confidence']

"""
def keypoints_to_list(keypoints):
    l = []
    l.extend(keypoints['nose'])
    l.extend(keypoints['left_eye'])
    l.extend(keypoints['right_eye'])
    l.extend(keypoints['left_ear'])
    l.extend(keypoints['right_ear'])
    l.extend(keypoints['left_shoulder'])
    l.extend(keypoints['right_shoulder'])
    l.extend(keypoints['left_elbow'])
    l.extend(keypoints['right_elbow'])
    l.extend(keypoints['left_wrist'])
    l.extend(keypoints['right_wrist'])
    l.extend(keypoints['left_hip'])
    l.extend(keypoints['right_hip'])
    l.extend(keypoints['left_knee'])
    l.extend(keypoints['right_knee'])
    l.extend(keypoints['left_ankle'])
    l.extend(keypoints['right_ankle'])
    return l
"""