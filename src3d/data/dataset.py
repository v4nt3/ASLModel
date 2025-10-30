# dataset.py -- parche: SignLanguageDataset robusto y reusable

import json
import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, split_file, num_frames=16, frame_size=112,
                 is_training=True, class2idx: dict = None, transform=None):
        """
        data_dir: carpeta base con secuencias/frames (si aplica)
        split_file: json con la lista de muestras
        class2idx: mapping global opcional {class_name: idx}. Si None, se construye desde el split_file.
        """
        self.data_dir = Path(data_dir)
        with open(split_file, 'r') as f:
            self.data = json.load(f)

        # Construir class2idx si no fue provisto
        if class2idx is None:
            # buscamos nombres de clase en el JSON (intentar varias claves comunes)
            names = set()
            for item in self.data:
                if 'label' in item:
                    names.add(item['label'])
                elif 'class' in item:
                    names.add(item['class'])
                elif 'class_name' in item:
                    names.add(item['class_name'])
                # si hay class_id numérico no lo agregamos aquí
            self.class_names = sorted(list(names))
            self.class2idx = {c: i for i, c in enumerate(self.class_names)}
        else:
            self.class2idx = class2idx
            # construir class_names (opcional, para inspección)
            inv = {v: k for k, v in class2idx.items()}
            self.class_names = [inv[i] if i in inv else None for i in range(len(class2idx))]

        self.num_frames = num_frames
        self.frame_size = frame_size
        self.is_training = is_training
        self.transform = transform

        # transforms por defecto (puedes ajustar)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),            # -> [C, H, W], float [0,1]
                transforms.Resize((frame_size, frame_size))  # si usas PIL frames
            ])

    def __len__(self):
        return len(self.data)

    def _get_label_idx(self, item):
        """Obtiene un índice de etiqueta robusto a varias claves en el JSON."""
        # casos en que el JSON guarda el nombre de la clase
        if 'label' in item:
            label_name = item['label']
            if label_name in self.class2idx:
                return self.class2idx[label_name]
            else:
                raise KeyError(f"Label name '{label_name}' not found in class2idx mapping.")
        if 'class' in item:
            label_name = item['class']
            if label_name in self.class2idx:
                return self.class2idx[label_name]
            else:
                raise KeyError(f"Class name '{label_name}' not found in class2idx mapping.")
        # si el JSON guarda un índice numérico
        if 'class_id' in item:
            try:
                return int(item['class_id'])
            except:
                raise ValueError("class_id exists but cannot be cast to int.")
        if 'label_id' in item:
            return int(item['label_id'])

        raise KeyError("No label found for item. Expected one of ['label','class','class_id','label_id'].")

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- Cargar frames/ vídeo ---
        # Este bloque asume que el JSON tiene una ruta en 'video_path' o 'frames'
        # Ajusta según tu dataset real.
        if 'path' in item:
            video_path = self.data_dir / item['path']
            # aquí deberías implementar la lectura real de frames (cv2, decodificador, etc.)
            frames = self._read_video_frames(video_path)
        elif 'frames' in item:
            # 'frames' puede ser una lista de paths relativos
            frame_paths = [self.data_dir / p for p in item['frames']]
            frames = [self._read_frame(p) for p in frame_paths]
        else:
            # fallback: si el JSON tiene 'frames_dir' con una carpeta de imágenes
            if 'frames_dir' in item:
                frames_dir = self.data_dir / item['frames_dir']
                frame_files = sorted([frames_dir / f for f in os.listdir(frames_dir)])
                frames = [self._read_frame(p) for p in frame_files]
            else:
                raise KeyError("No video_path/frames/frames_dir found in item JSON.")

        # frames -> numpy array shape (T, H, W, C) o lista de PIL images
        # reducir/expandir a self.num_frames
        frames = self._temporal_sample(frames, self.num_frames)

        # aplicar transform individual por frame y apilar en tensor  [C, T, H, W]
        processed = []
        for f in frames:
            if isinstance(f, np.ndarray):
                # convertir a PIL si tu transform espera PIL, o convertir directamente
                f_t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0  # [C,H,W]
            else:
                f_t = self.transform(f)  # assume transform -> tensor [C,H,W]
            processed.append(f_t)

        # apilar: list de [C,H,W] -> tensor [C, T, H, W]
        video_tensor = torch.stack(processed, dim=1)  # dim=1 -> time dimension

        # obtener etiqueta
        label_idx = self._get_label_idx(item)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        # Comprobaciones rápidas (útiles para debug, coméntalas en producción)
        # assert video_tensor.shape[1] == self.num_frames

        return video_tensor, label_tensor

    # --- helper functions que debes adaptar a tu flujo concreto ---
    def _read_frame(self, path):
        # ejemplo simple usando PIL
        from PIL import Image
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            return self.transform(img)
        return img

    def _read_video_frames(self, video_path):
        # Implementa lectura real (cv2.VideoCapture o decodificador)
        # aquí devolvemos una lista vacía como placeholder para que no falle
        # ----> Sustituye por tu lector real
        raise NotImplementedError("Implement _read_video_frames with cv2 or decord according to your dataset.")

    def _temporal_sample(self, frames, num_frames):
        """
        frames: lista de PIL o numpy arrays
        Devuelve exactamente num_frames seleccionados (padding o recorte).
        """
        T = len(frames)
        if T == 0:
            raise ValueError("Sequence has zero frames.")
        if T == num_frames:
            return frames
        if T > num_frames:
            # sample uniformemente
            indices = np.linspace(0, T - 1, num_frames).astype(int)
            return [frames[i] for i in indices]
        else:
            # si hay menos frames, repetir últimos hasta completar
            reps = frames.copy()
            while len(reps) < num_frames:
                reps.append(frames[-1])
            return reps[:num_frames]
