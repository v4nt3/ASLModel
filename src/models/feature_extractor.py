"""
Feature extractor usando ResNet101 pre-entrenado
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import models # type: ignore

class ResNet50FeatureExtractor(nn.Module):
    """Extrae features espaciales de frames usando ResNet50"""
    
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        """
        Args:
            pretrained: Usar pesos pre-entrenados de ImageNet
            freeze: Congelar pesos de ResNet (no entrenar)
        """
        super().__init__()

        resnet = models.resnet50(pretrained=pretrained)

        # Remover la capa de clasificación final
        # ResNet50 tiene: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        # Queremos todo excepto fc
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Dimensión de salida de ResNet50
        self.feature_dim = 2048
        
        # Congelar pesos si se requiere
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrae features de un batch de frames
        
        Args:
            x: Tensor de shape (batch_size, 3, H, W)
        
        Returns:
            features: Tensor de shape (batch_size, 2048)
        """
        with torch.set_grad_enabled(self.training):
            features = self.features(x)  # (B, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (B, 2048)
        
        return features
    
    def extract_video_features(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Extrae features de todos los frames de un video
        
        Args:
            video_frames: Tensor de shape (batch_size, num_frames, 3, H, W)
        
        Returns:
            features: Tensor de shape (batch_size, num_frames, 2048)
        """
        batch_size, num_frames, C, H, W = video_frames.shape
        
        # Reshape para procesar todos los frames a la vez
        frames_flat = video_frames.view(batch_size * num_frames, C, H, W)
        
        # Extraer features
        features_flat = self.forward(frames_flat)  # (B*T, 2048)
        
        # Reshape de vuelta
        features = features_flat.view(batch_size, num_frames, self.feature_dim)
        
        return features
