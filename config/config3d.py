"""
Centralized configuration file for the 3D CNN Sign Language Recognition pipeline
Modify all paths and hyperparameters here
"""
from pathlib import Path

class Config:
    # PATHS
    # Data paths
    DATA_DIR = Path("data/dataset") 
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
    NUM_FRAMES = 40
    FRAME_SIZE = 112
    NUM_CLASSES = 2288
    
    # MODEL PARAMETERS
    # Model architecture: 'c3d', 'r3d', or 'i3d'
    MODEL_ARCH = 'c3d'
    
    # Model hyperparameters
    DROPOUT = 0.5
    
    # TRAINING PARAMETERS
    # Training settings
    BATCH_SIZE = 8
    NUM_EPOCHS = 150
    NUM_WORKERS = 4  # DataLoader workers
    
    # Optimizer settings
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9  # For SGD
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'plateau'  # 'plateau' or 'cosine'
    SCHEDULER_PATIENCE = 10  # For ReduceLROnPlateau
    SCHEDULER_FACTOR = 0.5  # For ReduceLROnPlateau
    MIN_LR = 1e-6
    
    # CALLBACKS
    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Model checkpointing
    SAVE_BEST_ONLY = False  # If True, only save best model
    SAVE_FREQUENCY = 10  # Save checkpoint every N epochs
    
    # TRAINING OPTIONS
    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision for faster training
    
    # Class imbalance handling
    USE_CLASS_WEIGHTS = True
    
    # Gradient clipping
    USE_GRAD_CLIP = True
    GRAD_CLIP_VALUE = 1.0
    
    # EVALUATION PARAMETERS
    # Top-K accuracy
    TOP_K = [1, 5, 10]
    
    # Visualization
    PLOT_DPI = 200
    
    # ROC/AUC settings
    MAX_CLASSES_FOR_ROC = 50  # Only plot ROC for top N classes (too many classes = slow)
    
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
        print("CONFIGURATION")
        print(f"Model Architecture: {cls.MODEL_ARCH.upper()}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print(f"Number of Frames: {cls.NUM_FRAMES}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision: {cls.USE_AMP}")
        print(f"Early Stopping: {cls.USE_EARLY_STOPPING} (patience={cls.EARLY_STOPPING_PATIENCE})")
        print(f"LR Scheduler: {cls.USE_SCHEDULER} (type={cls.SCHEDULER_TYPE})")
        print(f"Class Weights: {cls.USE_CLASS_WEIGHTS}")
