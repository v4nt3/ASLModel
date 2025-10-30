"""
Main training script for 3D CNN Sign Language Recognition
All configuration is in config.py - just run this script directly!
"""
import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path

from config.config3d import Config
from src3d.models import get_model
from src3d.data.dataset import SignLanguageDataset
from src3d.training.trainer import Trainer
from src3d.training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from src3d.utils.metrics import calculate_class_weights
from src3d.utils.visualization import plot_training_history

def main():
    # Create directories
    Config.create_directories()
    Config.print_config()
    
    # Set device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create datasets
    print("Loading datasets")

    train_dataset = SignLanguageDataset(
        data_dir=Config.DATA_DIR,
        split_file=Config.OUTPUT_DIR / 'train.json',
        num_frames=Config.NUM_FRAMES,
        frame_size=Config.FRAME_SIZE,
        is_training=True,
        class2idx=None,
        motion_threshold=Config.MOTION_THRESHOLD,
        skip_initial_frames=Config.SKIP_INITIAL_FRAMES
    )

    global_class2idx = train_dataset.class2idx
    num_classes = len(global_class2idx)
    print(f"Detected {num_classes} unique classes in training set")

    val_dataset = SignLanguageDataset(
        data_dir=Config.DATA_DIR,
        split_file=Config.OUTPUT_DIR / 'val.json',
        num_frames=Config.NUM_FRAMES,
        frame_size=Config.FRAME_SIZE,
        is_training=False,
        class2idx=global_class2idx,
        motion_threshold=Config.MOTION_THRESHOLD,
        skip_initial_frames=Config.SKIP_INITIAL_FRAMES
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    print(f"Creating {Config.MODEL_ARCH.upper()} model")
    
    model = get_model(
        arch=Config.MODEL_ARCH,
        num_classes=num_classes,
        dropout=Config.DROPOUT
    )

    model = model.to(device)
    
    if Config.USE_CLASS_WEIGHTS:
        print("\nCalculating class weights")
        class_weights = calculate_class_weights(
            Config.OUTPUT_DIR / 'train.json',
            global_class2idx  # pasar el mapping correcto
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Create callbacks
    callbacks = {}
    
    if Config.USE_EARLY_STOPPING:
        callbacks['early_stopping'] = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            mode='max'
        )
    
    callbacks['checkpoint'] = ModelCheckpoint(
        checkpoint_dir=Config.CHECKPOINT_DIR,
        save_best_only=Config.SAVE_BEST_ONLY,
        mode='max'
    )
    
    scheduler_callback = None
    if Config.USE_SCHEDULER:
        scheduler_callback = LearningRateScheduler(
            optimizer,
            scheduler_type=Config.SCHEDULER_TYPE,
            mode='max',
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
            T_max=Config.NUM_EPOCHS
        )
        callbacks['scheduler'] = scheduler_callback
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=Config.USE_AMP,
        accumulation_steps=Config.ACCUMULATION_STEPS,
        grad_clip=Config.GRAD_CLIP_VALUE if Config.USE_GRAD_CLIP else None
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_acc = 0.0

    print("\nVerifying data shapes...")
    videos, labels = next(iter(train_loader))
    print(f"  Video batch shape: {videos.shape}")  # expected (B, C, T, H, W)
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels in batch: {torch.unique(labels).tolist()}")
    print(f"  Label range: [{labels.min().item()}, {labels.max().item()}]")
    assert num_classes > int(labels.max().item()), f"num_classes ({num_classes}) must be > max_label ({labels.max().item()})"
    print("  ✓ Data verification passed\n")

    
    # Training loop
    print("Starting training\n")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train - NO pasar scheduler aquí (se maneja después)
        train_loss, train_acc = trainer.train_one_epoch(train_loader, epoch, scheduler=None)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)        
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if Config.USE_SCHEDULER:
            current_lr = scheduler_callback.get_last_lr()
            history['lr'].append(current_lr)
        else:
            history['lr'].append(Config.LEARNING_RATE)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        if Config.USE_SCHEDULER:
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'history': history,
            'config': {
                'model_arch': Config.MODEL_ARCH,
                'num_classes': num_classes,  # usar num_classes detectado
                'num_frames': Config.NUM_FRAMES,
                'frame_size': Config.FRAME_SIZE
            }
        }
        
        if Config.USE_SCHEDULER:
            state['scheduler_state_dict'] = scheduler_callback.scheduler.state_dict()
        
        callbacks['checkpoint'].save(state, val_acc, epoch)
        
        # Update learning rate scheduler
        if Config.USE_SCHEDULER:
            scheduler_callback.step(val_acc)
        
        # Check early stopping
        if Config.USE_EARLY_STOPPING:
            if callbacks['early_stopping'](val_acc):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
    
    
    print("GENERATING FINAL VISUALIZATIONS")
    
    
    print("\nGenerating training history with overfitting analysis")
    plot_training_history(history, Config.PLOTS_DIR)
    
    # Save final history
    with open(Config.RESULTS_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"TRAINING COMPLETED!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {Config.CHECKPOINT_DIR}")
    print(f"Plots saved to: {Config.PLOTS_DIR}")
    print(f"\nTo generate comprehensive evaluation metrics:")
    print(f"  - ROC curves with AUC scores")
    print(f"  - Confusion matrices")
    print(f"  - Precision-Recall curves")
    print(f"  - Top-K accuracy analysis")
    print(f"  - Per-class performance")
    print(f"\nRun: python scripts/evaluate.py")

if __name__ == '__main__':
    main()
