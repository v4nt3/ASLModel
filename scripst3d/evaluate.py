"""
Comprehensive evaluation script for 3D CNN Sign Language Recognition
Generates ROC curves, AUC, confusion matrices, and detailed metrics
All configuration is in config.py
"""
import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from models import get_model
from data.dataset import SignLanguageDataset
from src3d.utils.metrics import AverageMeter, accuracy
from src3d.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_top_k_accuracy
)

def evaluate_model(model, test_loader, device, num_classes):
    """Comprehensive model evaluation with all metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()
    
    print("\nEvaluating model")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluation')
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            
            # Calculate top-k accuracies
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc10 = accuracy(outputs, labels, topk=(1, 10))[0]
            
            top1.update(acc1.item(), videos.size(0))
            top5.update(acc5.item(), videos.size(0))
            top10.update(acc10.item(), videos.size(0))
            
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({
                'top1': f'{top1.avg:.2f}%',
                'top5': f'{top5.avg:.2f}%',
                'top10': f'{top10.avg:.2f}%'
            })
    
    return {
        'top1_acc': top1.avg,
        'top5_acc': top5.avg,
        'top10_acc': top10.avg,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }

def analyze_per_class_performance(labels, predictions, probabilities, num_classes, save_path):
    """Analyze and save per-class performance metrics"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=range(num_classes), average=None, zero_division=0
    )
    
    # Create per-class report
    per_class_stats = []
    for i in range(num_classes):
        per_class_stats.append({
            'class': i,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        })
    
    # Sort by F1 score
    per_class_stats_sorted = sorted(per_class_stats, key=lambda x: x['f1'], reverse=True)
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(per_class_stats_sorted, f, indent=2)
    
    # Print statistics
    
    print("PER-CLASS PERFORMANCE ANALYSIS")
    
    
    # Best performing classes
    print("\nTop 10 best performing classes:")
    for i, stats in enumerate(per_class_stats_sorted[:10], 1):
        print(f"{i}. Class {stats['class']}: F1={stats['f1']:.3f}, Support={stats['support']}")
    
    # Worst performing classes
    print("\nTop 10 worst performing classes:")
    worst_classes = sorted(per_class_stats, key=lambda x: x['f1'])[:10]
    for i, stats in enumerate(worst_classes, 1):
        print(f"{i}. Class {stats['class']}: F1={stats['f1']:.3f}, Support={stats['support']}")
    
    print(f"\nSaved detailed per-class analysis to {save_path}")
    
    return per_class_stats

def main():
    # Create directories
    Config.create_directories()
    
    # Set device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Config.CHECKPOINT_DIR / 'checkpoint_best.pth'
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python scripts/train.py")
        return
    
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    print(f"Creating {checkpoint['config']['model_arch'].upper()} model")
    model = get_model(
        arch=checkpoint['config']['model_arch'],
        num_classes=checkpoint['config']['num_classes'],
        dropout=Config.DROPOUT
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    
    # Create test dataset
    print("\nLoading test dataset")
    test_dataset = SignLanguageDataset(
        data_dir=Config.DATA_DIR,
        split_file=Config.OUTPUT_DIR / 'test.json',
        num_frames=Config.NUM_FRAMES,
        frame_size=Config.FRAME_SIZE,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, Config.NUM_CLASSES)
    
    # Print results
    
    print("EVALUATION RESULTS")
    
    print(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_acc']:.2f}%")
    print(f"Top-10 Accuracy: {results['top10_acc']:.2f}%")
    
    
    # Save results summary
    results_summary = {
        'top1_accuracy': results['top1_acc'],
        'top5_accuracy': results['top5_acc'],
        'top10_accuracy': results['top10_acc'],
        'model_arch': checkpoint['config']['model_arch'],
        'num_classes': Config.NUM_CLASSES,
        'test_samples': len(test_dataset),
        'epoch': checkpoint['epoch']
    }
    
    with open(Config.RESULTS_DIR / 'test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    
    print("GENERATING VISUALIZATIONS")
    
    
    # 1. Confusion Matrix
    print("\n1. Generating confusion matrices")
    cm = confusion_matrix(results['labels'], results['predictions'])
    plot_confusion_matrix(
        cm, 
        results['labels'],
        results['predictions'],
        Config.PLOTS_DIR,
        top_k=100
    )
    
    # 2. ROC Curves and AUC
    print("\n2. Generating ROC curves and AUC metrics")
    plot_roc_curves(
        results['labels'],
        results['probabilities'],
        Config.NUM_CLASSES,
        Config.PLOTS_DIR,
        max_classes=Config.MAX_CLASSES_FOR_ROC
    )
    
    # 3. Precision-Recall Curves
    print("\n3. Generating Precision-Recall curves")
    plot_precision_recall_curves(
        results['labels'],
        results['probabilities'],
        Config.NUM_CLASSES,
        Config.PLOTS_DIR,
        max_classes=Config.MAX_CLASSES_FOR_ROC
    )
    
    # 4. Top-K Accuracy Analysis
    print("\n4. Generating Top-K accuracy analysis")
    plot_top_k_accuracy(
        results['labels'],
        results['probabilities'],
        Config.PLOTS_DIR,
        k_values=[1, 5, 10, 20, 50, 100]
    )
    
    # 5. Per-class Performance Analysis
    print("\n5. Analyzing per-class performance")
    per_class_stats = analyze_per_class_performance(
        results['labels'],
        results['predictions'],
        results['probabilities'],
        Config.NUM_CLASSES,
        Config.RESULTS_DIR / 'per_class_performance.json'
    )

    
    
    print("EVALUATION COMPLETE")
    
    print(f"Results saved to: {Config.RESULTS_DIR}")
    print(f"Plots saved to: {Config.PLOTS_DIR}")
    print("\nGenerated visualizations:")
    print("  - Confusion matrices (normalized and raw)")
    print("  - ROC curves with AUC scores")
    print("  - Precision-Recall curves")
    print("  - Top-K accuracy analysis")
    print("  - Per-class performance analysis")

if __name__ == '__main__':
    main()
