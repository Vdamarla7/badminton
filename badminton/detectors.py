import os

import requests
import time
import numpy as np

from typing import List
from tqdm import tqdm
from ultralytics import YOLO

from enum import Enum

from dataclasses import dataclass
from torchvision import transforms
import torch

from huggingface_hub import hf_hub_url
from New.PosesArchive.coco_keypoints import (COCO_KEYPOINTS)
from utilities import get_video_frames, find_bboxes_closest_to_center


class TaskType(Enum):
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"
    POSE = "pose"


class SapiensPoseEstimationType(Enum):
    COCO_POSE_ESTIMATION_1B = "./models/sapiens_1b_coco_best_coco_AP_821.pth"
    COCO_POSE_ESTIMATION_2B = "./models/sapiens_2b_coco_best_coco_AP_822_torchscript.pt2"
    

def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)


def download_hf_model(model_name: str, task_type: TaskType, model_dir: str = 'models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    path = model_dir + "/" + model_name
    if os.path.exists(path):
        return path

    print(f"Model {model_name} not found, downloading from Hugging Face Hub...")

    model_version = "_".join(model_name.split("_")[:2])
    repo_id = "facebook/sapiens"
    subdirectory = f"sapiens_lite_host/torchscript/{task_type.value}/checkpoints/{model_version}"

    # hf_hub_download(repo_id=repo_id, filename=model_name, subfolder=subdirectory, local_dir=model_dir)
    url = hf_hub_url(repo_id=repo_id, filename=model_name, subfolder=subdirectory)
    download(url, path)
    print("Model downloaded successfully to", path)

    return path


def create_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               transforms.Lambda(lambda x: x.unsqueeze(0))
                               ])

@dataclass
class DetectorConfig:
    model_path: str = "../models/yolov8m.pt"
    person_id: int = 0
    conf_thres: float = 0.25

class Detector:
    def __init__(self, config: DetectorConfig = DetectorConfig()):
        model_path = config.model_path
        if not model_path.endswith(".pt"):
            model_path = model_path.split(".")[0] + ".pt"
        self.model = YOLO(model_path)
        self.person_id = config.person_id
        self.conf_thres = config.conf_thres

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.detect(img)

    def detect(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        results = self.model(img, conf=self.conf_thres)
        detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)

        # Filter out only person
        person_detections = detections[detections[:, -1] == self.person_id]
        boxes = person_detections[:, :-2].astype(int)

        print(f"Detection inference took: {time.perf_counter() - start:.4f} seconds")
        return boxes


class SapiensPoseEstimation:
    def __init__(self,
                 type: SapiensPoseEstimationType = SapiensPoseEstimationType.COCO_POSE_ESTIMATION_2B,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        # Load the model
        self.device = device
        self.dtype = dtype
        path = '../models/sapiens_2b_coco_best_coco_AP_822_torchscript.pt2'
        self.model = torch.jit.load(path).eval().to(device).to(dtype)
        self.preprocessor = transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize((1024,768)),
                               transforms.ToTensor(),
                               #transforms.Normalize(mean=mean, std=std),
                               ])

        # Initialize the YOLO-based detector
        self.detector = Detector()

    
    def detect(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Detect persons in the image
        bboxes = self.detector.detect(img)
        closest_bboxes = find_bboxes_closest_to_center(bboxes, img.shape[1] // 2)
        
        # make sure the closest_bboxes are sorted by the bottom player (A) and then the top player(B)
        if closest_bboxes[0][1] > closest_bboxes[1][1]:
            closest_bboxes = [closest_bboxes[1], closest_bboxes[0]]        
        
        # Process the image and estimate the pose
        keypoints = self.estimate_pose(img, closest_bboxes)

        print(f"Pose estimation inference took: {time.perf_counter() - start:.4f} seconds")
        return bboxes, closest_bboxes, keypoints

    @torch.inference_mode()
    def estimate_pose(self, img: np.ndarray, bboxes: List[List[float]]) -> (np.ndarray, List[dict]):
        all_keypoints = []
        result_img = img.copy()

        for bbox in bboxes:
            cropped_img = self.crop_image(img, bbox)
            tensor = self.preprocessor(cropped_img).unsqueeze(0).to(self.device).to(self.dtype)

            heatmaps = self.model(tensor)
            keypoints = self.heatmaps_to_keypoints(heatmaps[0].cpu().numpy())
            all_keypoints.append(keypoints)

        return all_keypoints

    def crop_image(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        return img[y1:y2, x1:x2]


    def heatmaps_to_keypoints(self, heatmaps: np.ndarray) -> dict:
        keypoints = {}
        for i, name in enumerate(COCO_KEYPOINTS):
            if i < heatmaps.shape[0]:
                y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
                conf = heatmaps[i, y, x]
                keypoints[name] = (float(x), float(y), float(conf))
        return keypoints
    
    def process_video(self, input_file):
        _, _, _, _, frames = get_video_frames(input_file)
        print(f"There are {len(frames)} frames in {input_file}")
        
        count = 0
        bboxes_list = []
        closest_bboxes_list = []
        keypoints_list = []
        for img in frames:
            print("processing frame #:" + str(count + 1))
            start_time = time.perf_counter()
            bboxes, closest_bboxes, keypoints = self.detect(img)
            bboxes_list.append(bboxes)
            closest_bboxes_list.append(closest_bboxes)
            keypoints_list.append(keypoints)
            print(f"Time taken: {time.perf_counter() - start_time:.4f} seconds")
            count += 1

        return bboxes_list, closest_bboxes_list, keypoints_list