import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
from pathlib import Path
import albumentations as A #type: ignore
import os

class SignLanguageDataset(Dataset):
    """Dataset for sign language video classification"""
    
    def __init__(self, data_dir, split_file, num_frames=40, frame_size=112, is_training=True, class2idx=None):
        
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.is_training = is_training
        
        with open(split_file, 'r') as f:
            self.data = json.load(f)
        
        # Mapear las clases a índices
        if class2idx is None:
            self.class_names = sorted(list({item['label'] for item in self.data}))
            self.class2idx = {c: i for i, c in enumerate(self.class_names)}
        else:
            self.class2idx = class2idx
            # opcionalmente: build class_names from class2idx
            self.class_names = [None] * len(class2idx)
            for k, v in class2idx.items():
                self.class_names[v] = k

        if is_training:
            self.spatial_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.3),
                A.Resize(frame_size, frame_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.spatial_transform = A.Compose([
                A.Resize(frame_size, frame_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if 'label' in item:
            label_str = item['label']
            label_idx = self.class2idx[label_str]
        elif 'class' in item:
            label_str = item['class']
            label_idx = self.class2idx[label_str]
        elif 'class_id' in item:
            # si class_id ya es indice numérico, asegúrate que esté dentro del rango
            label_idx = int(item['class_id'])
        else:
            print(f"\n[ERROR] Unknown JSON format. Available keys: {list(item.keys())}")
            print(f"[ERROR] Sample item: {item}")
            raise KeyError(f"Could not find video path key in item. Available keys: {list(item.keys())}")
        
        video_path_str = video_path_str.replace('\\', '/')
        video_path = self.data_dir / video_path_str
        
        if 'label' in item:
            label_str = item['label']  # sigue siendo string
            label_idx = self.class2idx[label_str]  # convertir a entero
            

        elif 'class' in item:
            label = item['class']
        elif 'class_id' in item:
            label = item['class_id']
        else:
            raise KeyError(f"Could not find label key in item. Available keys: {list(item.keys())}")
        
        try:
            frames = self._load_video(video_path)
            frames = self._sample_frames(frames)
            frames = self._apply_transforms(frames)
            
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
            
            label = torch.tensor(label_idx, dtype=torch.long)
            
            return frames, label
        except Exception as e:
            print(f"\n[WARNING] Failed to load video: {video_path}")
            print(f"[WARNING] Error: {str(e)}")
            print(f"[WARNING] Skipping this sample and returning a dummy tensor")
            dummy_frames = torch.zeros((3, self.num_frames, self.frame_size, self.frame_size))
            label = torch.tensor(label, dtype=torch.long)
            return dummy_frames, label
    
    def _load_video(self, video_path):
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise ValueError(f"Video file does not exist: {video_path}")
        
        if video_path.stat().st_size == 0:
            raise ValueError(f"Video file is empty (0 bytes): {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        return frames
    
    def _sample_frames(self, frames):
        """
        Extract frames from the center of the video.
        If video has fewer frames than needed, pad by repeating the last frame.
        """
        num_frames_available = len(frames)
        
        if num_frames_available >= self.num_frames:
            center_idx = num_frames_available // 2
            start_idx = center_idx - (self.num_frames // 2)
            end_idx = start_idx + self.num_frames
            
            if start_idx < 0:
                start_idx = 0
                end_idx = self.num_frames
            elif end_idx > num_frames_available:
                end_idx = num_frames_available
                start_idx = end_idx - self.num_frames
            
            indices = list(range(start_idx, end_idx))
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames.copy()
            last_frame = frames[-1]
            
            frames_to_pad = self.num_frames - num_frames_available
            sampled_frames.extend([last_frame] * frames_to_pad)
        
        return sampled_frames
    
    def _apply_transforms(self, frames):
        transformed_frames = []
        
        for frame in frames:
            transformed = self.spatial_transform(image=frame)
            transformed_frames.append(transformed['image'])
        
        transformed_frames = np.stack(transformed_frames, axis=0)
        return transformed_frames
