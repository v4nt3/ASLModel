"""
Modelo completo para clasificación de ASL
"""
import torch #type: ignore
import torch.nn as nn #type: ignore
from src.models.feature_extractor import ResNet50FeatureExtractor
from src.models.temporal_model import TemporalLSTM, TemporalAttentionLSTM

class ASLClassifier(nn.Module):
    """Clasificador completo: ResNet50 + LSTM + Classifier"""
    
    def __init__(self,
                 num_classes: int = 2288,
                 resnet_pretrained: bool = True,
                 freeze_backbone: bool = True,
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.3,
                 lstm_bidirectional: bool = True,
                 use_attention: bool = True,
                 classifier_hidden_dims: list = [1024, 512],
                 classifier_dropout: float = 0.5,
                 resnet_freeze: bool = None):
        """
        Args:
            num_classes: Número de clases a predecir
            resnet_pretrained: Usar ResNet pre-entrenado
            freeze_backbone: Congelar pesos de ResNet (nuevo nombre preferido)
            lstm_hidden_size: Tamaño hidden de LSTM
            lstm_num_layers: Número de capas LSTM
            lstm_dropout: Dropout en LSTM
            lstm_bidirectional: LSTM bidireccional
            use_attention: Usar attention mechanism
            classifier_hidden_dims: Dimensiones de capas ocultas del clasificador
            classifier_dropout: Dropout en clasificador
            resnet_freeze: (Deprecated) Usar freeze_backbone en su lugar
        """
        super().__init__()
        
        if resnet_freeze is not None:
            freeze_backbone = resnet_freeze
        
        # Feature extractor (ResNet50)
        self.feature_extractor = ResNet50FeatureExtractor(
            pretrained=resnet_pretrained,
            freeze=freeze_backbone
        )
        
        # Temporal model (LSTM)
        if use_attention:
            self.temporal_model = TemporalAttentionLSTM(
                input_dim=self.feature_extractor.feature_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                bidirectional=lstm_bidirectional
            )
        else:
            self.temporal_model = TemporalLSTM(
                input_dim=self.feature_extractor.feature_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                bidirectional=lstm_bidirectional
            )
        
        # Classifier head
        layers = []
        input_dim = self.temporal_model.output_dim
        
        for hidden_dim in classifier_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(classifier_dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        self.num_classes = num_classes
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo - acepta tanto frames de video como features pre-extraídas
        
        Args:
            input_data: Puede ser:
                - Video frames: (batch_size, num_frames, 3, H, W) - 5D
                - Features pre-extraídas: (batch_size, num_frames, feature_dim) - 3D
        
        Returns:
            logits: Tensor de shape (batch_size, num_classes)
        """
        if input_data.dim() == 3:
            # Features pre-extraídas (batch_size, num_frames, feature_dim)
            spatial_features = input_data
        elif input_data.dim() == 5:
            # Video frames (batch_size, num_frames, C, H, W)
            spatial_features = self.feature_extractor.extract_video_features(input_data)
        else:
            raise ValueError(
                f"Input debe ser 3D (features pre-extraídas) o 5D (video frames). "
                f"Recibido: {input_data.dim()}D con shape {input_data.shape}"
            )
        
        # Modelado temporal
        temporal_features = self.temporal_model(spatial_features)
        # (batch_size, lstm_output_dim)
        
        # Clasificación
        logits = self.classifier(temporal_features)
        # (batch_size, num_classes)
        
        return logits
    
    def extract_features(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Extrae features sin clasificar (útil para visualización)
        
        Args:
            input_data: Video frames (batch_size, num_frames, 3, H, W) o 
                       features pre-extraídas (batch_size, num_frames, feature_dim)
        
        Returns:
            features: Tensor de shape (batch_size, lstm_output_dim)
        """
        if input_data.dim() == 3:
            spatial_features = input_data
        elif input_data.dim() == 5:
            spatial_features = self.feature_extractor.extract_video_features(input_data)
        else:
            raise ValueError(f"Input inválido con shape {input_data.shape}")
        
        temporal_features = self.temporal_model(spatial_features)
        return temporal_features
    
    def unfreeze_backbone(self):
        """Descongelar el backbone de ResNet para fine-tuning"""
        for param in self.feature_extractor.features.parameters():
            param.requires_grad = True
        self.feature_extractor.features.train()
    
    def freeze_backbone(self):
        """Congelar el backbone de ResNet"""
        for param in self.feature_extractor.features.parameters():
            param.requires_grad = False
        self.feature_extractor.features.eval()

class ASLClassifierPreExtracted(nn.Module):
    """Clasificador para features pre-extraídas (más rápido para entrenar)"""
    
    def __init__(self,
                 num_classes: int = 2288,
                 feature_dim: int = 2048,
                 refine_dim: int = 512, 
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.4,
                 lstm_bidirectional: bool = True,
                 use_attention: bool = True,
                 classifier_hidden_dims: list = [1024, 512],
                 classifier_dropout: float = 0.6):
        """
        Args:
            num_classes: Número de clases
            feature_dim: Dimensión de features de entrada
            lstm_hidden_size: Tamaño hidden de LSTM
            lstm_num_layers: Número de capas LSTM
            lstm_dropout: Dropout en LSTM
            lstm_bidirectional: LSTM bidireccional
            use_attention: Usar attention mechanism
            classifier_hidden_dims: Dimensiones de capas ocultas
            classifier_dropout: Dropout en clasificador
        """
        super().__init__()

                # ======== Refinador CNN (nuevo bloque) ========
        # Paso 1: Proyectar features a espacio menor
        self.projection = nn.Linear(feature_dim, refine_dim)

        # Paso 2: Mini CNN que aprende relaciones espaciales implícitas
        # Suponemos que refine_dim = 512 → reshape como (8x8x8)
        self.cnn_refiner = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        if use_attention:
            self.temporal_model = TemporalAttentionLSTM(
                input_dim=refine_dim,  # <--- 512 en lugar de 2048
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                bidirectional=lstm_bidirectional
            )
        else:
            self.temporal_model = TemporalLSTM(
                input_dim=refine_dim,  # <--- 512 en lugar de 2048
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                bidirectional=lstm_bidirectional
            )

        
        # Classifier
        layers = []
        input_dim = self.temporal_model.output_dim
        
        for hidden_dim in classifier_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(classifier_dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        self.num_classes = num_classes
        self.refine_dim = refine_dim
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con features pre-extraídas
        
        Args:
            features: Tensor de shape (batch_size, num_frames, feature_dim)
        
        Returns:
            logits: Tensor de shape (batch_size, num_classes)
        """
        B, T, D = features.shape
        # Proyección inicial
        x = self.projection(features)  # (B, T, refine_dim)

        # CNN refinement
        x = x.view(B*T, 8, 8, 8).permute(0, 3, 1, 2)  # [B*T, C, H, W]
        x = self.cnn_refiner(x)
        x = torch.flatten(x, 1)  # [B*T, refine_dim]

        # Volver a secuencia temporal
        x = x.view(B, T, -1)

        # LSTM temporal + atención
        temporal_features = self.temporal_model(x)

        # Clasificador
        logits = self.classifier(temporal_features)
        return logits
