"""
Training callbacks for monitoring and controlling the training process
"""
import numpy as np
from pathlib import Path
import torch #type: ignore

class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    """
    def __init__(self, patience=20, min_delta=0.001, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for metrics like loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score):
        """
        Check if training should stop
        
        Args:
            current_score: Current validation metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nStopping training after {self.counter} epochs without improvement")
                return True
        
        return False
    
    def reset(self):
        """Reset the early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """
    Save model checkpoints during training
    """
    def __init__(self, checkpoint_dir, save_best_only=False, mode='max'):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: If True, only save when metric improves
            mode: 'max' for metrics like accuracy, 'min' for metrics like loss
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score = None
        
    def is_better(self, current_score):
        """Check if current score is better than best score"""
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return current_score > self.best_score
        else:
            return current_score < self.best_score
    
    def save(self, state, current_score, epoch):
        """
        Save checkpoint if conditions are met
        
        Args:
            state: Dictionary containing model state and training info
            current_score: Current validation metric value
            epoch: Current epoch number
            
        Returns:
            True if checkpoint was saved, False otherwise
        """
        is_best = self.is_better(current_score)
        
        if is_best:
            self.best_score = current_score
        
        # Save latest checkpoint
        if not self.save_best_only:
            latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
            
            torch.save(state, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            
            torch.save(state, best_path)
            print(f" Saved best model with score: {current_score:.4f}")
            return True
        
        return False


class LearningRateScheduler:
    """
    Wrapper for PyTorch learning rate schedulers
    """
    def __init__(self, optimizer, scheduler_type='plateau', **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('plateau', 'cosine', 'step')
            **kwargs: Additional arguments for the scheduler
        """
        import torch.optim as optim #type: ignore
        
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'max'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                min_lr=kwargs.get('min_lr', 1e-6)
                
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric=None):
        """
        Update learning rate
        
        Args:
            metric: Validation metric (required for ReduceLROnPlateau)
        """
        if self.scheduler_type == 'plateau':
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires a metric value")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0]
