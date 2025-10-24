import torch #type: ignore
import torch.nn as nn #type: ignore
from torch.utils.data import DataLoader #type: ignore
from torch.optim import Optimizer #type: ignore
from torch.optim.lr_scheduler import _LRScheduler #type: ignore
from typing import Dict, Optional, Callable
from tqdm import tqdm #type: ignore
import numpy as np
from pathlib import Path

from .metrics import MetricsCalculator, MetricsTracker


class Trainer:
    """Trainer class for ASL recognition model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cuda',
        num_classes: int = 2288,
        class_names: list = None,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip
        
        # Initialize metrics
        self.metrics_tracker = MetricsTracker()
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        print(f"Trainer initialized on device: {device}")
        print(f"Mixed precision training: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {train_loader.batch_size * self.gradient_accumulation_steps}")
        print(f"Gradient clipping: {self.gradient_clip}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        metrics_calc = MetricsCalculator(self.num_classes, self.class_names)
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            _, predicted = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)
            
            metrics_calc.update(predicted, labels, probs)
            running_loss += loss.item() * self.gradient_accumulation_steps
            
            if batch_idx % self.log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = metrics_calc.compute_accuracy(k=1)
        epoch_top5 = metrics_calc.compute_accuracy(k=5)
        
        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'train_top5': epoch_top5
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        metrics_calc = MetricsCalculator(self.num_classes, self.class_names)
        running_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for features, labels in pbar:
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)
            
            metrics_calc.update(predicted, labels, probs)
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = metrics_calc.compute_accuracy(k=1)
        epoch_top5 = metrics_calc.compute_accuracy(k=5)
        
        return {
            'val_loss': epoch_loss,
            'val_acc': epoch_acc,
            'val_top5': epoch_top5
        }
    
    def train(
        self, 
        num_epochs: int, 
        early_stopping_patience: int = 10,
        on_epoch_start: Optional[Callable[[int], None]] = None,
        on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None
    ):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
            on_epoch_start: Optional callback called at the start of each epoch with epoch number
            on_epoch_end: Optional callback called at the end of each epoch with epoch number and metrics
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            if on_epoch_start is not None:
                on_epoch_start(epoch)
            
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = self.validate(epoch)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_metrics = {**train_metrics, **val_metrics, 'learning_rate': current_lr}
            self.metrics_tracker.update(epoch_metrics)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']:.2f}% | Train Top-5: {train_metrics['train_top5']:.2f}%")
            print(f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.2f}% | Val Top-5: {val_metrics['val_top5']:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_metrics)
            
            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ New best model saved! Val Acc: {self.best_val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        
        self.metrics_tracker.save(self.checkpoint_dir / 'training_history.npy')
    
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Test the model on test set"""
        print("\nEvaluating on test set")
        
        best_checkpoint = self.checkpoint_dir / 'best_model.pth'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        
        metrics_calc = MetricsCalculator(self.num_classes, self.class_names)
        running_loss = 0.0
        
        pbar = tqdm(self.test_loader, desc="Testing")
        
        for features, labels in pbar:
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)
            
            metrics_calc.update(predicted, labels, probs)
            running_loss += loss.item()
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = metrics_calc.compute_accuracy(k=1)
        test_top5 = metrics_calc.compute_accuracy(k=5)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.2f}%")
        print(f"Test Top-5: {test_top5:.2f}%")
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_top5': test_top5,
            'metrics_calculator': metrics_calc
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics_history': self.metrics_tracker.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            filepath = self.checkpoint_dir / 'best_model.pth'
        else:
            filepath = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.metrics_tracker.history = checkpoint.get('metrics_history', {})
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
