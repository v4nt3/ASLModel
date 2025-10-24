import torch #type: ignore
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


class MetricsCalculator:
    """Calculate and track training metrics"""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, probabilities: torch.Tensor = None):
        """Update metrics with batch results"""
        self.all_preds.extend(predictions.cpu().detach().numpy())
        self.all_labels.extend(labels.cpu().detach().numpy())
        if probabilities is not None:
            self.all_probs.extend(probabilities.cpu().detach().numpy())
    
    def compute_accuracy(self, k: int = 1) -> float:
        """Compute top-k accuracy"""
        if k == 1:
            correct = sum(p == l for p, l in zip(self.all_preds, self.all_labels))
            return 100.0 * correct / len(self.all_labels)
        else:
            # Top-k accuracy
            if not self.all_probs:
                raise ValueError("Probabilities needed for top-k accuracy")
            
            probs = np.array(self.all_probs)
            labels = np.array(self.all_labels)
            top_k_preds = np.argsort(probs, axis=1)[:, -k:]
            
            correct = sum(label in preds for label, preds in zip(labels, top_k_preds))
            return 100.0 * correct / len(labels)
    
    def compute_per_class_accuracy(self) -> Dict[str, float]:
        """Compute accuracy per class"""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        per_class_acc = {}
        for i in range(self.num_classes):
            mask = labels == i
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean() * 100
                per_class_acc[self.class_names[i]] = acc
        
        return per_class_acc
    
    def compute_confusion_matrix(self, normalize: bool = True) -> np.ndarray:
        """Compute confusion matrix"""
        cm = confusion_matrix(self.all_labels, self.all_preds)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        return classification_report(
            self.all_labels, 
            self.all_preds, 
            target_names=self.class_names,
            zero_division=0
        )
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        summary = {
            'top1_accuracy': self.compute_accuracy(k=1),
            'top5_accuracy': self.compute_accuracy(k=5) if self.all_probs else 0.0,
        }
        
        # Add average per-class accuracy
        per_class_acc = self.compute_per_class_accuracy()
        if per_class_acc:
            summary['avg_per_class_accuracy'] = np.mean(list(per_class_acc.values()))
        
        return summary


class MetricsTracker:
    """Track metrics across epochs"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_top5': [],
            'val_top5': [],
            'learning_rate': []
        }
    
    def update(self, epoch_metrics: Dict[str, float]):
        """Update history with epoch metrics"""
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_acc') -> Tuple[int, float]:
        """Get epoch with best metric"""
        if metric not in self.history or not self.history[metric]:
            return -1, 0.0
        
        values = self.history[metric]
        best_idx = np.argmax(values)
        return best_idx, values[best_idx]
    
    def save(self, filepath: str):
        """Save history to file"""
        np.save(filepath, self.history)
    
    def load(self, filepath: str):
        """Load history from file"""
        self.history = np.load(filepath, allow_pickle=True).item()
