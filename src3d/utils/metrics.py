"""
Utility functions for metrics and evaluation
"""
import torch #type: ignore
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def calculate_class_weights(split_file, num_classes):
    """Calculate class weights for imbalanced dataset"""
    import json
    from collections import Counter
    
    with open(split_file, 'r') as f:
        data = json.load(f)
    
    class_counts = Counter([item['label'] for item in data])
    
    total_samples = len(data)
    weights = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 1)
        weights[class_idx] = total_samples / (num_classes * count)
    
    weights = weights / weights.sum() * num_classes
    
    print(f"Class weight statistics:")
    print(f"  Min weight: {weights.min():.4f}")
    print(f"  Max weight: {weights.max():.4f}")
    print(f"  Mean weight: {weights.mean():.4f}")
    
    return weights
