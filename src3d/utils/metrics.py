"""
Utility functions for metrics and evaluation
"""
import torch #type: ignore
import numpy as np
import json
from collections import Counter

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
    """Computes the accuracy over the k top predictions (returns percentages)"""
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

def calculate_class_weights(split_file, class2idx):
    """
    Calculate class weights for imbalanced dataset.

    split_file: path to JSON split (list of items). Each item should contain a label
                as either 'label' (preferred) or 'class' or 'class_id'.
    class2idx: dict mapping class_name -> idx (global mapping derived from train split)

    Returns: numpy array of shape (num_classes,) with weights (dtype float32)
             and also prints some stats.
    """
    with open(split_file, 'r') as f:
        data = json.load(f)

    # Contar usando nombres de clase si existen, si no intentar usar índices
    name_counts = Counter()
    idx_counts = Counter()
    for item in data:
        if 'label' in item:
            name_counts[item['label']] += 1
        elif 'class' in item:
            name_counts[item['class']] += 1
        elif 'class_name' in item:
            name_counts[item['class_name']] += 1
        elif 'class_id' in item:
            try:
                idx_counts[int(item['class_id'])] += 1
            except:
                pass
        elif 'label_id' in item:
            try:
                idx_counts[int(item['label_id'])] += 1
            except:
                pass

    num_classes = len(class2idx)
    total_samples = len(data)
    weights = np.zeros(num_classes, dtype=np.float32)

    # Priorizar conteo por nombre de clase (si hay mapping)
    if len(name_counts) > 0:
        # mapear name_counts -> idx_counts via class2idx
        for name, count in name_counts.items():
            if name in class2idx:
                idx = class2idx[name]
                idx_counts[idx] = count
            else:
                # clase en split no encontrada en class2idx: la ignoramos o sacamos advertencia
                print(f"[WARN] class name '{name}' in {split_file} not present in provided class2idx mapping.")
    # ahora idx_counts puede tener conteos (si el JSON daba índices) o haber sido rellenado
    for cls_idx in range(num_classes):
        count = idx_counts.get(cls_idx, 0)
        if count <= 0:
            # evitar división por cero: asignar una pequeña frecuencia (suavizado)
            count = 1
        weights[cls_idx] = total_samples / (num_classes * float(count))

    # Normalizar para que la suma sea igual a num_classes (opcional, facilita interpretación)
    weights = weights / weights.sum() * num_classes

    print(f"Class weight statistics:")
    print(f"  Min weight: {weights.min():.6f}")
    print(f"  Max weight: {weights.max():.6f}")
    print(f"  Mean weight: {weights.mean():.6f}")

    return weights
