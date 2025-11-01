"""
Centralized configuration file for the 3D CNN Sign Language Recognition pipeline
Modify all paths and hyperparameters here
"""
from pathlib import Path

class Config:
    # PATHS
    # Data paths
    DATA_DIR = Path("data/dataset")  # Directory containing video files organized by class
    OUTPUT_DIR = Path("data")  # Where to save train.json, val.json, test.json
    
    # Model paths
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")
    PLOTS_DIR = Path("plots")
    
    # DATA PARAMETERS
    # Dataset split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15  # Remaining after train and val
    
    # Video processing
    NUM_FRAMES = 40  # Number of frames to extract from center of each video
    FRAME_SIZE = 112  # Spatial size (height and width)
    NUM_CLASSES = 2303  # Total number of sign language classes
    
    # Motion detection parameters
    MOTION_THRESHOLD = 0.02  # Umbral para detectar inicio de movimiento (0.01-0.05)
    SKIP_INITIAL_FRAMES = 0.10  # Porcentaje de frames iniciales a saltar (0.0-0.3)
    
    # MODEL PARAMETERS
    # Model architecture: 'c3d', 'r3d', or 'i3d'
    MODEL_ARCH = 'i3d'
    
    # Model hyperparameters
    DROPOUT = 0.5
    
    # TRAINING PARAMETERS
    # Training settings - Adjusted for 15GB VRAM
    BATCH_SIZE = 8  # Reduced from 8 to fit in 15GB VRAM with 3D CNN
    ACCUMULATION_STEPS = 2  # Gradient accumulation to simulate batch_size=8
    NUM_EPOCHS = 150
    NUM_WORKERS = 10  # DataLoader workers (adjust based on CPU cores)
    
    # Optimizer settings
    LEARNING_RATE = 0.01  # Increased 10x for better learning
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9  # For SGD
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'  # Changed from 'plateau' to 'cosine' for better convergence
    SCHEDULER_PATIENCE = 5  # Reduced from 10 for faster adaptation (only for plateau)
    SCHEDULER_FACTOR = 0.5  # For ReduceLROnPlateau
    MIN_LR = 1e-6
    
    USE_WARMUP = True
    WARMUP_EPOCHS = 5  # Gradually increase LR for first 5 epochs
    WARMUP_START_LR = 1e-4  # Start with lower LR during warmup
    
    # CALLBACKS
    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Model checkpointing
    SAVE_BEST_ONLY = False  # If True, only save best model
    SAVE_FREQUENCY = 10  # Save checkpoint every N epochs
    
    # TRAINING OPTIONS
    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision for faster training and lower memory
    
    # Class imbalance handling
    USE_CLASS_WEIGHTS = False
    
    # Gradient clipping
    USE_GRAD_CLIP = True
    GRAD_CLIP_VALUE = 1.0
    
    # EVALUATION PARAMETERS
    # Top-K accuracy
    TOP_K = [1, 5, 10]
    
    # Visualization
    PLOT_DPI = 200
    
    # ROC/AUC settings
    MAX_CLASSES_FOR_ROC = 50  # Only plot ROC for top N classes 
    
    # Confusion matrix
    CONFUSION_MATRIX_NORMALIZE = True
    
    # DEVICE
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Model Architecture: {cls.MODEL_ARCH.upper()}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print(f"Number of Frames: {cls.NUM_FRAMES}")
        print(f"Motion Detection: Threshold={cls.MOTION_THRESHOLD}, Skip Initial={cls.SKIP_INITIAL_FRAMES*100:.0f}%")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Gradient Accumulation Steps: {cls.ACCUMULATION_STEPS}")
        print(f"Effective Batch Size: {cls.BATCH_SIZE * cls.ACCUMULATION_STEPS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        if cls.USE_WARMUP:
            print(f"Warmup: {cls.WARMUP_EPOCHS} epochs (start LR: {cls.WARMUP_START_LR})")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision: {cls.USE_AMP}")
        print(f"Early Stopping: {cls.USE_EARLY_STOPPING} (patience={cls.EARLY_STOPPING_PATIENCE})")
        print(f"LR Scheduler: {cls.USE_SCHEDULER} (type={cls.SCHEDULER_TYPE})")
        print(f"Class Weights: {cls.USE_CLASS_WEIGHTS}")
        print("=" * 60)
