"""
Procesamiento de videos: extracción de frames del centro del video
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import torch #type: ignore
from torchvision import transforms #type: ignore
from tqdm import tqdm #type: ignore

class VideoFrameExtractor:
    """Extrae frames del centro de videos, ignorando inicio y final"""
    
    def __init__(self, 
                 num_frames: int = 30,
                 frame_size: Tuple[int, int] = (224, 224),
                 skip_start_percent: float = 0.10,
                 skip_end_percent: float = 0.10,
                 padding_mode: str = "repeat_last",
                 min_frames_threshold: int = 10):
        """
        Args:
            num_frames: Número de frames a extraer
            frame_size: Tamaño de los frames (height, width)
            skip_start_percent: Porcentaje del inicio a ignorar
            skip_end_percent: Porcentaje del final a ignorar
            padding_mode: Modo de padding para videos cortos ("repeat_last", "zeros", "replicate")
            min_frames_threshold: Mínimo de frames para considerar un video válido
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.skip_start_percent = skip_start_percent
        self.skip_end_percent = skip_end_percent
        self.padding_mode = padding_mode
        self.min_frames_threshold = min_frames_threshold
        
        self.videos_with_padding = 0
        self.videos_too_short = 0
        
        # Transformaciones para normalización
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_frames(self, video_path: str) -> torch.Tensor:
        """
        Extrae frames del centro del video con manejo robusto de videos cortos
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            Tensor de shape (num_frames, 3, H, W) o None si el video es inválido
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.min_frames_threshold:
            self.videos_too_short += 1
            cap.release()
            return None
        
        if total_frames < self.num_frames * 1.5:
            # Video muy corto: usar todos los frames sin crop
            start_frame = 0
            end_frame = total_frames
            usable_frames = total_frames
        else:
            # Video normal: aplicar crop del 10% inicio/final
            start_frame = int(total_frames * self.skip_start_percent)
            end_frame = int(total_frames * (1 - self.skip_end_percent))
            usable_frames = end_frame - start_frame
        
        if usable_frames >= self.num_frames:
            # Suficientes frames: seleccionar uniformemente
            frame_indices = np.linspace(start_frame, end_frame - 1, 
                                       self.num_frames, dtype=int)
        else:
            # Pocos frames: usar todos los disponibles
            frame_indices = np.arange(start_frame, end_frame, dtype=int)
            self.videos_with_padding += 1
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                if len(frames) > 0 and self.padding_mode == "repeat_last":
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, *self.frame_size))
                continue
            
            # Convertir BGR a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Aplicar transformaciones
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)
        
        cap.release()
        
        if len(frames) < self.num_frames:
            frames = self._pad_frames(frames)
        
        # Stack frames: (num_frames, 3, H, W)
        frames_tensor = torch.stack(frames[:self.num_frames])
        
        return frames_tensor
    
    def _pad_frames(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Aplica padding a la lista de frames según el modo configurado
        
        Args:
            frames: Lista de frames existentes
            
        Returns:
            Lista de frames con padding hasta num_frames
        """
        if len(frames) == 0:
            # No hay frames válidos: crear frames negros
            return [torch.zeros(3, *self.frame_size) for _ in range(self.num_frames)]
        
        padded_frames = frames.copy()
        frames_needed = self.num_frames - len(frames)
        
        if self.padding_mode == "repeat_last":
            # Repetir el último frame
            last_frame = frames[-1]
            padded_frames.extend([last_frame.clone() for _ in range(frames_needed)])
            
        elif self.padding_mode == "zeros":
            # Rellenar con frames negros
            padded_frames.extend([torch.zeros(3, *self.frame_size) for _ in range(frames_needed)])
            
        elif self.padding_mode == "replicate":
            # Replicar frames existentes cíclicamente
            for i in range(frames_needed):
                padded_frames.append(frames[i % len(frames)].clone())
        
        return padded_frames
    
    def get_stats(self) -> dict:
        """Retorna estadísticas de procesamiento"""
        return {
            'videos_with_padding': self.videos_with_padding,
            'videos_too_short': self.videos_too_short
        }

    def extract_frames_batch(self, 
                            video_paths: List[str], 
                            output_dir: Path = None,
                            save_to_disk: bool = False) -> List[torch.Tensor]:
        """
        Extrae frames de múltiples videos
        
        Args:
            video_paths: Lista de rutas a videos
            output_dir: Directorio para guardar frames (si save_to_disk=True)
            save_to_disk: Si guardar frames a disco
            
        Returns:
            Lista de tensores de frames
        """
        all_frames = []
        
        for video_path in tqdm(video_paths, desc="Extrayendo frames"):
            try:
                frames = self.extract_frames(video_path)
                if frames is not None:
                    all_frames.append(frames)
                    
                    if save_to_disk and output_dir:
                        video_name = Path(video_path).stem
                        output_path = output_dir / f"{video_name}.pt"
                        torch.save(frames, output_path)
                    
            except Exception as e:
                print(f"Error procesando {video_path}: {e}")
                # Agregar tensor vacío en caso de error
                all_frames.append(torch.zeros(self.num_frames, 3, *self.frame_size))
        
        return all_frames


class FrameAugmenter:
    """Augmentación de frames para entrenamiento"""
    
    def __init__(self, 
                 horizontal_flip_prob: float = 0.5,
                 rotation_degrees: float = 10,
                 color_jitter: float = 0.2):
        """
        Args:
            horizontal_flip_prob: Probabilidad de flip horizontal
            rotation_degrees: Grados máximos de rotación
            color_jitter: Intensidad de color jitter
        """
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter/2
            )
        ])
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Aplica augmentación a todos los frames
        
        Args:
            frames: Tensor de shape (num_frames, 3, H, W)
            
        Returns:
            Frames augmentados
        """
        # Aplicar la misma augmentación a todos los frames del video
        augmented_frames = []
        for frame in frames:
            # Convertir a PIL para augmentación
            frame_pil = transforms.ToPILImage()(frame)
            augmented_pil = self.augment(frame_pil)
            augmented_tensor = transforms.ToTensor()(augmented_pil)
            augmented_frames.append(augmented_tensor)
        
        return torch.stack(augmented_frames)
