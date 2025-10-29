import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import torch
from pathlib import Path
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 10


def plot_training_history(history, save_dir):
    """
    Plot training and validation loss/accuracy over epochs.
    Detects overfitting by showing the gap between train and val metrics.
    """
    print("\n Generating training history plot...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overfitting detection - Loss gap
    loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 0].plot(epochs, loss_gap, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].fill_between(epochs, 0, loss_gap, where=(loss_gap > 0), alpha=0.3, color='red', label='Overfitting')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Gap (Val - Train)')
    axes[1, 0].set_title('Overfitting Detection (Loss)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting detection - Accuracy gap
    acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(epochs, acc_gap, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].fill_between(epochs, 0, acc_gap, where=(acc_gap > 0), alpha=0.3, color='red', label='Overfitting')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap (Train - Val)')
    axes[1, 1].set_title('Overfitting Detection (Accuracy)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', bbox_inches='tight')
    plt.close()
    
    print(f" ✓ Training history plot saved to {save_dir / 'training_history.png'}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, normalize=False, top_k=50):
    """
    Plot confusion matrix. For large number of classes, shows top-k most confused classes.
    """
    matrix_type = "normalized" if normalize else "raw"
    print(f"\n Generating {matrix_type} confusion matrix...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        filename = 'confusion_matrix_normalized.png'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        filename = 'confusion_matrix.png'
        fmt = 'd'
    
    # For large number of classes, show only top-k most confused
    if len(class_names) > top_k:
        # Calculate confusion (off-diagonal sum for each class)
        confusion_per_class = cm.sum(axis=1) - np.diag(cm)
        top_confused_indices = np.argsort(confusion_per_class)[-top_k:]
        
        cm_subset = cm[np.ix_(top_confused_indices, top_confused_indices)]
        class_names_subset = [class_names[i] for i in top_confused_indices]
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_subset, annot=False, fmt=fmt, cmap='Blues', 
                    xticklabels=class_names_subset, yticklabels=class_names_subset,
                    cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        plt.title(f'{title} (Top {top_k} Most Confused Classes)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90, ha='right', fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
    else:
        plt.figure(figsize=(max(12, len(class_names) * 0.3), max(10, len(class_names) * 0.3)))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_dir / filename, bbox_inches='tight')
    plt.close()
    
    print(f" ✓ Confusion matrix saved to {save_dir / filename}")


def plot_roc_curves(y_true, y_probs, num_classes, save_dir, max_classes_to_plot=10):
    """
    Plot ROC curves with AUC scores. For many classes, shows micro/macro average and sample of individual classes.
    """
    print(f"\n Generating ROC curves with AUC scores...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot micro and macro averages
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=3)
    
    # Plot sample of individual class ROC curves
    if num_classes <= max_classes_to_plot:
        classes_to_plot = range(num_classes)
    else:
        # Select classes with best, worst, and median AUC
        auc_values = [roc_auc[i] for i in range(num_classes)]
        sorted_indices = np.argsort(auc_values)
        classes_to_plot = [
            sorted_indices[0],  # Worst
            sorted_indices[len(sorted_indices)//4],
            sorted_indices[len(sorted_indices)//2],  # Median
            sorted_indices[3*len(sorted_indices)//4],
            sorted_indices[-1],  # Best
        ]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes_to_plot)))
    for i, color in zip(classes_to_plot, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.7,
                 label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class Classification')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', bbox_inches='tight')
    plt.close()
    
    # Save AUC scores to JSON
    auc_scores = {
        'micro_auc': float(roc_auc["micro"]),
        'macro_auc': float(roc_auc["macro"]),
        'per_class_auc': {f'class_{i}': float(roc_auc[i]) for i in range(num_classes)}
    }
    
    with open(save_dir / 'auc_scores.json', 'w') as f:
        json.dump(auc_scores, f, indent=2)
    
    print(f" ✓ ROC curves saved to {save_dir / 'roc_curves.png'}")
    print(f" ✓ AUC scores saved to {save_dir / 'auc_scores.json'}")
    print(f"   - Micro-average AUC: {roc_auc['micro']:.4f}")
    print(f"   - Macro-average AUC: {roc_auc['macro']:.4f}")


def plot_precision_recall_curves(y_true, y_probs, num_classes, save_dir, max_classes_to_plot=10):
    """
    Plot Precision-Recall curves with Average Precision scores.
    """
    print(f"\n Generating Precision-Recall curves...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute Precision-Recall curve and Average Precision for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    
    # Compute micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_probs.ravel()
    )
    avg_precision["micro"] = average_precision_score(y_true_bin, y_probs, average="micro")
    
    # Compute macro-average
    avg_precision["macro"] = average_precision_score(y_true_bin, y_probs, average="macro")
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot micro-average
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average (AP = {avg_precision["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    
    # Plot sample of individual classes
    if num_classes <= max_classes_to_plot:
        classes_to_plot = range(num_classes)
    else:
        # Select classes with best, worst, and median AP
        ap_values = [avg_precision[i] for i in range(num_classes)]
        sorted_indices = np.argsort(ap_values)
        classes_to_plot = [
            sorted_indices[0],
            sorted_indices[len(sorted_indices)//4],
            sorted_indices[len(sorted_indices)//2],
            sorted_indices[3*len(sorted_indices)//4],
            sorted_indices[-1],
        ]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes_to_plot)))
    for i, color in zip(classes_to_plot, colors):
        plt.plot(recall[i], precision[i], color=color, lw=1.5, alpha=0.7,
                 label=f'Class {i} (AP = {avg_precision[i]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves (Macro-avg AP = {avg_precision["macro"]:.3f})')
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curves.png', bbox_inches='tight')
    plt.close()
    
    print(f" ✓ Precision-Recall curves saved to {save_dir / 'precision_recall_curves.png'}")
    print(f"   - Micro-average AP: {avg_precision['micro']:.4f}")
    print(f"   - Macro-average AP: {avg_precision['macro']:.4f}")


def plot_top_k_accuracy(y_true, y_probs, save_dir, max_k=10):
    """
    Plot Top-K accuracy for different values of K.
    """
    print(f"\n Generating Top-K accuracy analysis...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    k_values = range(1, min(max_k + 1, y_probs.shape[1] + 1))
    top_k_accs = []
    
    for k in k_values:
        # Get top-k predictions
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        # Check if true label is in top-k
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        top_k_acc = correct.mean() * 100
        top_k_accs.append(top_k_acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, top_k_accs, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('K')
    plt.ylabel('Top-K Accuracy (%)')
    plt.title('Top-K Accuracy Analysis')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for k, acc in zip(k_values, top_k_accs):
        plt.text(k, acc + 1, f'{acc:.1f}%', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'top_k_accuracy.png', bbox_inches='tight')
    plt.close()
    
    print(f" ✓ Top-K accuracy plot saved to {save_dir / 'top_k_accuracy.png'}")
    
    # Print top-k accuracies
    for k, acc in zip(k_values, top_k_accs):
        print(f"   - Top-{k} Accuracy: {acc:.2f}%")


def plot_class_performance(y_true, y_pred, class_names, save_dir, top_n=20, bottom_n=20):
    """
    Plot per-class accuracy showing best and worst performing classes.
    """
    print(f"\n Generating per-class performance analysis...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = len(class_names)
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    
    # Avoid division by zero
    class_accuracy = np.divide(class_correct, class_total, 
                               out=np.zeros_like(class_correct), 
                               where=class_total != 0) * 100
    
    # Sort by accuracy
    sorted_indices = np.argsort(class_accuracy)
    
    # Plot worst performing classes
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Bottom N classes
    bottom_indices = sorted_indices[:bottom_n]
    bottom_accs = class_accuracy[bottom_indices]
    bottom_names = [f"Class {i}" for i in bottom_indices]
    
    axes[0].barh(range(len(bottom_indices)), bottom_accs, color='red', alpha=0.7)
    axes[0].set_yticks(range(len(bottom_indices)))
    axes[0].set_yticklabels(bottom_names, fontsize=8)
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title(f'Bottom {bottom_n} Performing Classes')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Top N classes
    top_indices = sorted_indices[-top_n:][::-1]
    top_accs = class_accuracy[top_indices]
    top_names = [f"Class {i}" for i in top_indices]
    
    axes[1].barh(range(len(top_indices)), top_accs, color='green', alpha=0.7)
    axes[1].set_yticks(range(len(top_indices)))
    axes[1].set_yticklabels(top_names, fontsize=8)
    axes[1].set_xlabel('Accuracy (%)')
    axes[1].set_title(f'Top {top_n} Performing Classes')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'class_performance.png', bbox_inches='tight')
    plt.close()
    
    print(f" ✓ Class performance plot saved to {save_dir / 'class_performance.png'}")
    
    # Save detailed class performance to JSON
    class_performance = {
        f'class_{i}': {
            'accuracy': float(class_accuracy[i]),
            'correct': int(class_correct[i]),
            'total': int(class_total[i])
        }
        for i in range(num_classes)
    }
    
    with open(save_dir / 'class_performance.json', 'w') as f:
        json.dump(class_performance, f, indent=2)
    
    print(f" ✓ Class performance details saved to {save_dir / 'class_performance.json'}")


def generate_all_visualizations(y_true, y_pred, y_probs, class_names, history, save_dir):
    """
    Generate all visualization plots in one call.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("GENERATING ALL VISUALIZATIONS")
    
    # Training history
    if history is not None:
        print("\n[1/7] Training History Plot")
        plot_training_history(history, save_dir)
    
    # Confusion matrices
    print("\n[2/7] Confusion Matrices")
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir, normalize=False)
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir, normalize=True)
    
    # ROC curves
    print("\n[3/7] ROC Curves with AUC")
    plot_roc_curves(y_true, y_probs, len(class_names), save_dir)
    
    # Precision-Recall curves
    print("\n[4/7] Precision-Recall Curves")
    plot_precision_recall_curves(y_true, y_probs, len(class_names), save_dir)
    
    # Top-K accuracy
    print("\n[5/7] Top-K Accuracy Analysis")
    plot_top_k_accuracy(y_true, y_probs, save_dir)
    
    # Class performance
    print("\n[6/7] Per-Class Performance Analysis")
    plot_class_performance(y_true, y_pred, class_names, save_dir)
    
    print("\n[7/7] ✓ All visualizations complete!")
    print(f"\nAll plots saved to: {save_dir}")
