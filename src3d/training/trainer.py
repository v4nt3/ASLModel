# trainer.py -- Trainer integrado con metrics.py
import torch
import torch.nn as nn
from torch import amp
from torch.cuda.amp import autocast, GradScaler
from src3d.utils.metrics import AverageMeter, accuracy  # asumimos que metrics.py está en la misma carpeta
import time
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, device, use_amp=True,
                 accumulation_steps=1, grad_clip=None, log_every=100):
        
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
         
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and (device.type == 'cuda')
        self.scaler = GradScaler() if self.use_amp else None
        self.accumulation_steps = max(1, accumulation_steps)
        self.grad_clip = grad_clip
        self.log_every = log_every

    def train_one_epoch(self, dataloader, epoch_idx, scheduler=None):
        self.model.train()
        losses = AverageMeter()
        top1_meter = AverageMeter()
        total_samples = 0

        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.optimizer.zero_grad()

        start = time.time()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                        desc=f"Epoch {epoch_idx} [Train]", leave=True, dynamic_ncols=True)

        for batch_idx, (videos, labels) in enumerate(dataloader):
            total_norm = 0.0
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # forward (AMP-aware)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)


            # normalizar si se usan pasos de acumulación
            loss = loss / self.accumulation_steps

            # backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # solo hacer step cada accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # unscale antes de clippear si usamos AMP
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # debug: calcular grad norm
                
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                total_norm = total_norm ** 0.5

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if scheduler is not None:
                    # scheduler.step() puede hacerse por epoch o por step según tu política
                    pass

            # métricas
            # accuracy devuelve porcentaje (ej: 12.34)
            prec1 = accuracy(outputs.detach(), labels, topk=(1,))[0].item()
            batch_size = labels.size(0)
            losses.update(loss.item() * self.accumulation_steps, batch_size)
            top1_meter.update(prec1, batch_size)
            total_samples += batch_size

            # logging por batch
            progress_bar.update(1)
            progress_bar.set_postfix({
            "loss": f"{losses.avg:.4f}",
            "top1": f"{top1_meter.avg:.2f}%",
            "grad_norm": f"{total_norm:.4f}"
        })
            
        epoch_loss = losses.avg
        epoch_top1 = top1_meter.avg / 100.0  # convertir a fracción [0,1]
        return epoch_loss, epoch_top1

    def validate(self, dataloader):
        self.model.eval()

        losses = AverageMeter()
        top1_meter = AverageMeter()
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        with torch.no_grad():
            for videos, labels in dataloader:
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels)


                prec1 = accuracy(outputs, labels, topk=(1,))[0].item()
                batch_size = labels.size(0)
                losses.update(loss.item(), batch_size)
                top1_meter.update(prec1, batch_size)

        epoch_loss = losses.avg
        epoch_top1 = top1_meter.avg / 100.0
        return epoch_loss, epoch_top1
