import torch #type: ignore
from torch.utils.data import Dataset #type: ignore
import cv2
import numpy as np
import json
from pathlib import Path
import albumentations as A #type: ignore

class SignLanguageDataset(Dataset):
    """Dataset for sign language video classification"""
    
    def __init__(self, data_dir, split_file, num_frames=16, frame_size=112, is_training=True):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.is_training = is_training
        
        with open(split_file, 'r') as f:
            self.data = json.load(f)
        
        if is_training:
            self.spatial_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
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
        video_path = self.data_dir / item['video_path']
        label = item['label']
        
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)
        frames = self._apply_transforms(frames)
        
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        
        return frames, label
    
    def _load_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not load video: {video_path}")
        
        return frames
    
    def _sample_frames(self, frames):
        """
        Extract frames from the center of the video.
        If video has fewer frames than needed, pad by repeating the last frame.
        """
        num_frames_available = len(frames)
        
        if num_frames_available >= self.num_frames:
            # Calculate center position
            center_idx = num_frames_available // 2
            start_idx = center_idx - (self.num_frames // 2)
            end_idx = start_idx + self.num_frames
            
            # Ensure we don't go out of bounds
            if start_idx < 0:
                start_idx = 0
                end_idx = self.num_frames
            elif end_idx > num_frames_available:
                end_idx = num_frames_available
                start_idx = end_idx - self.num_frames
            
            # Extract center frames
            indices = list(range(start_idx, end_idx))
            sampled_frames = [frames[i] for i in indices]
        else:
            # Video is shorter than required frames
            # Use all available frames and pad with the last frame
            sampled_frames = frames.copy()
            last_frame = frames[-1]
            
            # Pad with last frame until we reach num_frames
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
