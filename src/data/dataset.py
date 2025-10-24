"""
Dataset para videos de ASL
"""
import torch #type: ignore
from torch.utils.data import Dataset #type: ignore
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

class ASLVideoDataset(Dataset):
    """Dataset para videos de lenguaje de señas"""
    
    def __init__(self,
                 video_paths: List[str],
                 labels: List[int],
                 frame_extractor,
                 augmenter=None,
                 use_augmentation: bool = False):
        """
        Args:
            video_paths: Lista de rutas a videos
            labels: Lista de etiquetas (índices de clases)
            frame_extractor: VideoFrameExtractor para procesar videos
            augmenter: FrameAugmenter para augmentación (opcional)
            use_augmentation: Si aplicar augmentación
        """
        self.video_paths = video_paths
        self.labels = labels
        self.frame_extractor = frame_extractor
        self.augmenter = augmenter
        self.use_augmentation = use_augmentation
        
        assert len(video_paths) == len(labels), \
            "Número de videos y labels debe ser igual"
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtiene un video y su label
        
        Returns:
            frames: Tensor de shape (num_frames, 3, H, W)
            label: Índice de clase
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extraer frames
        frames = self.frame_extractor.extract_frames(video_path)
        
        # Aplicar augmentación si está habilitada
        if self.use_augmentation and self.augmenter is not None:
            frames = self.augmenter(frames)
        
        return frames, label

class ASLPreExtractedDataset(Dataset):
    """Dataset para features pre-extraídas (más rápido para entrenamiento)"""
    
    def __init__(self,
                 features: torch.Tensor,
                 labels: torch.Tensor):
        """
        Args:
            features: Tensor de features (N, num_frames, feature_dim)
            labels: Tensor de labels (N,)
        """
        self.features = features
        self.labels = labels
        
        assert len(features) == len(labels), \
            "Número de features y labels debe ser igual"
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtiene features y label
        
        Returns:
            features: Tensor de shape (num_frames, feature_dim)
            label: Índice de clase
        """
        return self.features[idx], self.labels[idx]

class ASLLazyFeaturesDataset(Dataset):
    """
    Dataset que carga features .npy bajo demanda (lazy loading)
    Usa mucho menos memoria que cargar todo en memoria
    """
    
    def __init__(self,
                 feature_paths: List[str],
                 labels: List[int]):
        """
        Args:
            feature_paths: Lista de rutas a archivos .npy
            labels: Lista de etiquetas (índices de clases)
        """
        self.feature_paths = feature_paths
        self.labels = labels
        
        assert len(feature_paths) == len(labels), \
            "Número de feature_paths y labels debe ser igual"
    
    def __len__(self) -> int:
        return len(self.feature_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Carga features desde disco bajo demanda
        
        Returns:
            features: Tensor de shape (num_frames, feature_dim)
            label: Índice de clase
        """
        feature_path = self.feature_paths[idx]
        label = self.labels[idx]
        
        # Cargar features desde archivo .npy
        features = np.load(feature_path)  # Shape: (30, 2048)
        features = torch.from_numpy(features).float()
        
        return features, label

class ASLConsolidatedDataset(Dataset):
    """
    Dataset que usa features consolidadas con memory mapping
    Mucho más rápido que cargar archivos individuales
    """
    
    def __init__(self,
                 features_file: str,
                 labels_file: str,
                 indices: list = None):
        """
        Args:
            features_file: Path al archivo consolidado de features (.npy)
            labels_file: Path al archivo consolidado de labels (.npy)
            indices: Lista de índices a usar (para splits train/val/test)
        """
        # Usar memory mapping para no cargar todo en RAM
        self.features = np.load(features_file, mmap_mode='r')
        self.labels = np.load(labels_file, mmap_mode='r')
        
        # Si se proporcionan índices específicos, usarlos
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.labels)))
        
        print(f"✓ Dataset consolidado cargado:")
        print(f"  Total samples disponibles: {len(self.features)}")
        print(f"  Samples en este split: {len(self.indices)}")
        print(f"  Usando memory mapping (no carga en RAM)")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtiene features y label usando memory mapping
        
        Returns:
            features: Tensor de shape (num_frames, feature_dim)
            label: Índice de clase
        """
        actual_idx = self.indices[idx]
        
        # Memory mapping permite acceso rápido sin cargar todo
        features = self.features[actual_idx]  # Shape: (30, 2048)
        label = int(self.labels[actual_idx])
        
        # Convertir a tensor
        features = torch.from_numpy(features.copy()).float()
        
        return features, label
