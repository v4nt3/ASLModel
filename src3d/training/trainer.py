"""
Training logic for 3D CNN models
"""
import torch #type: ignore
import torch.nn as nn #type: ignore
from torch.amp import autocast, GradScaler #type: ignore
from tqdm import tqdm #type: ignore
import time

from src3d.utils.metrics import AverageMeter, accuracy

class Trainer:
    """Handles model training and validation"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, use_amp=True, grad_clip=None):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            use_amp: Use automatic mixed precision
            grad_clip: Gradient clipping value (None to disable)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler('cuda') if use_amp else None
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for videos, labels in pbar:
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            # Measure accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            # Update metrics
            batch_size = videos.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'top1': f'{top1.avg:.2f}%',
                'top5': f'{top5.avg:.2f}%'
            })
        
        return losses.avg, top1.avg, top5.avg
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
                
                # Measure accuracy
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                
                # Update metrics
                batch_size = videos.size(0)
                losses.update(loss.item(), batch_size)
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'top1': f'{top1.avg:.2f}%',
                    'top5': f'{top5.avg:.2f}%'
                })
        
        return losses.avg, top1.avg, top5.avg
