import torch #type: ignore
from pathlib import Path

class Config:
    #  RUTAS DEL PROYECTO 
    PROJECT_ROOT = Path(__file__).parent.parent
    """
Configuración centralizada para el proyecto de reconocimiento ASL
Todas las rutas y parámetros están definidos aquí para evitar argumentos en línea de comandos
"""
import torch #type: ignore
from pathlib import Path

class Config:
    #  RUTAS DEL PROYECTO 
    PROJECT_ROOT = Path(__file__).parent.parent
    
    DATA_ROOT = PROJECT_ROOT / "data"
    VIDEOS_DIR = DATA_ROOT / "dataset"  # Directorio con videos originales
    FRAMES_DIR = DATA_ROOT / "frames"  # Frames extraídos
    FEATURES_DIR = DATA_ROOT / "features101"  # Features pre-extraídas
    
    # Directorios de salida
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    LOGS_DIR = PROJECT_ROOT / "logs"
    VISUALIZATIONS_DIR = PROJECT_ROOT / "outputs"
    
    # Crear directorios si no existen
    for dir_path in [FRAMES_DIR, FEATURES_DIR, CHECKPOINTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    #  PARÁMETROS DE VIDEO Y FRAMES     
    NUM_FRAMES = 30  # Número de frames a extraer por video    
    FRAME_SIZE = (224, 224)  # Tamaño para ResNet101   
    CROP_PERCENTAGE = 0.10  # Ignorar 10% del inicio y final (tomar solo el centro)        
    PADDING_MODE = "repeat_last"  # "repeat_last", "zeros", "replicate"    
    MIN_FRAMES_THRESHOLD = 10  # Mínimo de frames para considerar un video válido       
    
    # ARQUITECTURA DEL MODELO     
    # Feature Extractor (ResNet)    
    RESNET_MODEL = "resnet101"  # Modelo de feature extraction    
    RESNET_PRETRAINED = True    
    FEATURE_DIM = 2048  # Dimensión de features de ResNet101  
    FREEZE_BACKBONE = True  # Congelar ResNet durante entrenamiento        
    
    
    # LSTM parameters    
    LSTM_HIDDEN_SIZE = 512    
    LSTM_NUM_LAYERS = 2 
    LSTM_DROPOUT = 0.3    
    LSTM_BIDIRECTIONAL = True    
    USE_ATTENTION = True  

    # Clasificador    
    NUM_CLASSES = 2288  # ASL Citizen dataset   
    CLASSIFIER_HIDDEN_DIMS = [1024, 512]    
    CLASSIFIER_DROPOUT = 0.5        
    
    #  HIPERPARÁMETROS DE ENTRENAMIENTO     
    BATCH_SIZE = 64    
    NUM_EPOCHS = 200    
    LEARNING_RATE = 3e-4 # Reducido para evitar inestabilidad    
    WEIGHT_DECAY = 1e-4  # Learning rate scheduler    
    SCHEDULER_TYPE = "plateau"  # "cosine", "step", "plateau" - plateau es más estable    
    WARMUP_EPOCHS = 5    
    SCHEDULER_T0 = 10  # Para CosineAnnealingWarmRestarts    
    SCHEDULER_TMULT = 2   
    MIN_LR = 1e-6  
    
    # Optimizer    
    OPTIMIZER = "adam"  # "adam", "adamw", "sgd"       
    
    #  DIVISIÓN DE DATOS     
    TRAIN_SPLIT = 0.7    
    VAL_SPLIT = 0.15    
    TEST_SPLIT = 0.15    
    RANDOM_SEED = 42       
    
    #  AUGMENTATION     
    USE_AUGMENTATION = True    
    HORIZONTAL_FLIP_PROB = 0.5    
    ROTATION_DEGREES = 10    
    COLOR_JITTER = 0.2        
    
    #  CONFIGURACIÓN DE ENTRENAMIENTO     
    GRADIENT_CLIP = 1.0    
    LABEL_SMOOTHING = 0.1  # reducir overfitting    
    EARLY_STOPPING_PATIENCE = 10    
    USE_AMP = True  # Automatic Mixed Precision  
    GRADIENT_ACCUMULATION_STEPS = 1         
    
    #  DEVICE Y WORKERS     
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    NUM_WORKERS = 0     
    PIN_MEMORY = True    
    PREFETCH_FACTOR = 4  # Pre-cargar 4 batches por worker    
    PERSISTENT_WORKERS = True  # No recrear workers entre épocas       
    USE_TORCH_COMPILE = False  # Deshabilitado - requiere triton que no es compatible con esta versión        
    USE_CONSOLIDATED_FEATURES = True  # Usar features consolidadas (mucho más rápido)        
    
    #  LOGGING Y CHECKPOINTS     
    LOG_INTERVAL = 10  # Log cada N batches    
    SAVE_INTERVAL = 5  # Guardar checkpoint cada N épocas       
    
    #  VISUALIZACIÓN     
    PLOT_INTERVAL = 1  # Actualizar plots cada N épocas    
    TOP_K = [1, 5, 10]  # Top-K accuracy a calcular

    @classmethod
    def print_config(cls):
        """Imprime la configuración actual"""
        print("=" * 80)
        print(" " * 25 + "CONFIGURACIÓN DEL MODELO ASL")
        print("=" * 80)
        print("\n[RUTAS]")
        print(f"  Videos Dir: {cls.VIDEOS_DIR}")
        print(f"  Frames Dir: {cls.FRAMES_DIR}")
        print(f"  Features Dir: {cls.FEATURES_DIR}")
        print(f"  Checkpoints Dir: {cls.CHECKPOINTS_DIR}")
        print(f"  Visualizations Dir: {cls.VISUALIZATIONS_DIR}")
        print("\n[MODELO]")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Feature Extractor: {cls.RESNET_MODEL}")
        print(f"  LSTM Hidden Size: {cls.LSTM_HIDDEN_SIZE}")
        print(f"  LSTM Layers: {cls.LSTM_NUM_LAYERS}")
        print(f"  Bidirectional: {cls.LSTM_BIDIRECTIONAL}")
        print(f"  Attention: {cls.USE_ATTENTION}")
        print(f"  Freeze Backbone: {cls.FREEZE_BACKBONE}")
        print("\n[ENTRENAMIENTO]")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Frames por Video: {cls.NUM_FRAMES}")
        print(f"  Número de Clases: {cls.NUM_CLASSES}")
        print("\n[OPTIMIZACIONES GPU]")
        print(f"  Mixed Precision (AMP): {cls.USE_AMP}")
        print(f"  Gradient Accumulation: {cls.GRADIENT_ACCUMULATION_STEPS} steps")
        print(f"  Num Workers: {cls.NUM_WORKERS}")
        print(f"  Prefetch Factor: {cls.PREFETCH_FACTOR}")
        print(f"  Persistent Workers: {cls.PERSISTENT_WORKERS}")
        print(f"  Torch Compile: {cls.USE_TORCH_COMPILE}")
        print(f"  Consolidated Features: {cls.USE_CONSOLIDATED_FEATURES}")
        print("=" * 80)
    
    @classmethod
    def to_dict(cls):
        """Convierte la configuración a diccionario"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Carga configuración desde diccionario"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
        return cls
