"""
Sistema de visualización AVANZADO para análisis completo del modelo
Incluye: métricas detalladas, análisis por clase, curvas ROC, y más
"""
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from sklearn.metrics import ( #type: ignore
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize #type: ignore
import pandas as pd #type: ignore
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")


class AdvancedVisualizer:
    """Visualizador avanzado con métricas completas"""
    
    def __init__(self, figsize=(15, 10), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 10
    
    def plot_training_history(self, history, save_path=None):
        """Genera gráficas del historial de entrenamiento"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Historial de Entrenamiento - PyTorch', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=4)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Validation', linewidth=2, markersize=4)
        axes[0, 0].set_title('Pérdida del Modelo', fontweight='bold')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train', linewidth=2, markersize=4)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Validation', linewidth=2, markersize=4)
        axes[0, 1].set_title('Precisión del Modelo', fontweight='bold')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[1, 0].plot(epochs, history['train_top5'], 'b-o', label='Train', linewidth=2, markersize=4)
        axes[1, 0].plot(epochs, history['val_top5'], 'r-s', label='Validation', linewidth=2, markersize=4)
        axes[1, 0].set_title('Top-5 Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Top-5 Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, history['learning_rate'], 'g-o', linewidth=2, markersize=4)
        axes[1, 1].set_title('Learning Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Gráfica de entrenamiento guardada: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, history, save_path=None):
        """Compara train vs validation gap para detectar overfitting"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Análisis de Overfitting', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Gap de accuracy
        train_acc = np.array(history['train_acc'])
        val_acc = np.array(history['val_acc'])
        gap = train_acc - val_acc
        
        axes[0].plot(epochs, gap, 'purple', linewidth=2)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].fill_between(epochs, 0, gap, alpha=0.3, color='purple')
        axes[0].set_title('Gap Train-Val Accuracy', fontweight='bold')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Gap (%)')
        axes[0].grid(True, alpha=0.3)
        
        # Ratio de loss
        train_loss = np.array(history['train_loss'])
        val_loss = np.array(history['val_loss'])
        
        axes[1].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[1].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        axes[1].fill_between(epochs, train_loss, val_loss, alpha=0.2, color='red')
        axes[1].set_title('Comparación de Loss', fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Análisis de overfitting guardado: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                            max_classes=50, save_path=None):
        """Genera matriz de confusión con análisis"""
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_classes)
        
        if n_classes > max_classes:
            print(f"Mostrando {max_classes} clases más frecuentes de {n_classes} totales.")
            
            from collections import Counter
            class_counts = Counter(y_true)
            top_classes = [cls for cls, _ in class_counts.most_common(max_classes)]
            
            mask = np.isin(y_true, top_classes) & np.isin(y_pred, top_classes)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            
            if class_names:
                class_names_filtered = [class_names[i] for i in top_classes]
            else:
                class_names_filtered = [f"Clase_{i}" for i in top_classes]
        else:
            y_true_filtered = y_true
            y_pred_filtered = y_pred
            class_names_filtered = class_names or [f"Clase_{i}" for i in range(n_classes)]
        
        cm = confusion_matrix(y_true_filtered, y_pred_filtered)
        
        fig_size = (max(12, len(class_names_filtered) * 0.5), 
                   max(10, len(class_names_filtered) * 0.4))
        plt.figure(figsize=fig_size)
        
        if len(class_names_filtered) <= 30:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names_filtered,
                       yticklabels=class_names_filtered,
                       cbar_kws={'shrink': 0.8})
        else:
            sns.heatmap(cm, annot=False, cmap='Blues',
                       xticklabels=class_names_filtered,
                       yticklabels=class_names_filtered,
                       cbar_kws={'shrink': 0.8})
        
        plt.title('Matriz de Confusión', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicción', fontsize=12)
        plt.ylabel('Valor Real', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Matriz de confusión guardada: {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(self, y_true, y_pred, class_names=None,
                              max_classes=30, save_path=None):
        """Métricas detalladas por clase: Precision, Recall, F1-Score"""
        if class_names is None:
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            class_names = [f"Clase_{i}" for i in unique_classes]
        
        # Calcular métricas por clase
        report = classification_report(y_true, y_pred, target_names=class_names,
                                     output_dict=True, zero_division=0)
        
        df_report = pd.DataFrame(report).iloc[:-1, :].T
        class_rows = df_report.iloc[:-2, :]
        
        # Ordenar por F1-score
        class_rows = class_rows.sort_values('f1-score', ascending=False)
        
        if len(class_rows) > max_classes:
            print(f"Mostrando top {max_classes} clases por F1-score")
            class_rows = class_rows.head(max_classes)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(class_rows) * 0.3)))
        fig.suptitle('Métricas por Clase (Top por F1-Score)', fontsize=16, fontweight='bold')
        
        y_pos = np.arange(len(class_rows))
        
        # Precision
        axes[0].barh(y_pos, class_rows['precision'], color='skyblue')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(class_rows.index, fontsize=8)
        axes[0].set_title('Precisión por Clase')
        axes[0].set_xlabel('Precisión')
        axes[0].set_xlim(0, 1)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Recall
        axes[1].barh(y_pos, class_rows['recall'], color='lightcoral')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels([])
        axes[1].set_title('Recall por Clase')
        axes[1].set_xlabel('Recall')
        axes[1].set_xlim(0, 1)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # F1-Score
        axes[2].barh(y_pos, class_rows['f1-score'], color='lightgreen')
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels([])
        axes[2].set_title('F1-Score por Clase')
        axes[2].set_xlabel('F1-Score')
        axes[2].set_xlim(0, 1)
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Métricas por clase guardadas: {save_path}")
        
        plt.show()
        
        return class_rows
    
    def plot_roc_curves(self, y_true, y_pred_proba, class_names=None, 
                       max_classes=10, save_path=None):
        """Curvas ROC con AUC para clasificación multiclase"""
        n_classes = y_pred_proba.shape[1]
        
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=3)
        
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='navy', linestyle=':', linewidth=3)
        
        # Top clases por AUC
        if n_classes <= max_classes:
            classes_to_plot = range(n_classes)
        else:
            sorted_classes = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)
            classes_to_plot = sorted_classes[:max_classes]
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes_to_plot)))
        for i, color in zip(classes_to_plot, colors):
            class_name = class_names[i] if class_names else f'Clase {i}'
            plt.plot(fpr[i], tpr[i], color=color, lw=2, alpha=0.7,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Curvas ROC - Clasificación Multiclase', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Curvas ROC guardadas: {save_path}")
        
        plt.show()
        
        return roc_auc
    
    def plot_worst_classes(self, y_true, y_pred, class_names=None, 
                          top_n=20, save_path=None):
        """Identifica y visualiza las clases con peor desempeño"""
        if class_names is None:
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            class_names = [f"Clase_{i}" for i in unique_classes]
        
        report = classification_report(y_true, y_pred, target_names=class_names,
                                     output_dict=True, zero_division=0)
        
        df_report = pd.DataFrame(report).iloc[:-1, :].T
        class_rows = df_report.iloc[:-2, :]
        
        # Ordenar por F1-score (peores primero)
        worst_classes = class_rows.sort_values('f1-score').head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
        
        y_pos = np.arange(len(worst_classes))
        
        ax.barh(y_pos, worst_classes['f1-score'], color='salmon', alpha=0.7, label='F1-Score')
        ax.barh(y_pos, worst_classes['precision'], color='skyblue', alpha=0.5, label='Precision')
        ax.barh(y_pos, worst_classes['recall'], color='lightgreen', alpha=0.5, label='Recall')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(worst_classes.index, fontsize=9)
        ax.set_xlabel('Score')
        ax.set_title(f'Top {top_n} Clases con Peor Desempeño', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Análisis de peores clases guardado: {save_path}")
        
        plt.show()
        
        return worst_classes
    
    def generate_complete_report(self, y_true, y_pred, y_pred_proba, 
                                class_names=None, save_dir='outputs'):
        """Genera reporte completo con todas las visualizaciones"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("GENERANDO REPORTE COMPLETO")
        
        # 1. Matriz de confusión
        print("\n1. Matriz de confusión...")
        self.plot_confusion_matrix(y_true, y_pred, class_names,
                                  save_path=save_path / 'confusion_matrix.png')
        
        # 2. Métricas por clase
        print("\n2. Métricas por clase...")
        self.plot_per_class_metrics(y_true, y_pred, class_names,
                                   save_path=save_path / 'per_class_metrics.png')
        
        # 3. Curvas ROC
        print("\n3. Curvas ROC...")
        roc_auc = self.plot_roc_curves(y_true, y_pred_proba, class_names,
                                      save_path=save_path / 'roc_curves.png')
        
        # 4. Peores clases
        print("\n4. Análisis de peores clases...")
        self.plot_worst_classes(y_true, y_pred, class_names,
                              save_path=save_path / 'worst_classes.png')
        
        # 5. Resumen de métricas
        print("\n5. Generando resumen de métricas...")
        self._save_metrics_summary(y_true, y_pred, y_pred_proba, roc_auc, save_path)
        
        print(f"Reporte completo guardado en: {save_path}")
    
    def _save_metrics_summary(self, y_true, y_pred, y_pred_proba, roc_auc, save_path):
        """Guarda resumen de métricas en archivo de texto"""
        with open(save_path / 'metrics_summary.txt', 'w') as f:
            f.write("RESUMEN DE MÉTRICAS\n")
            
            # Métricas globales
            acc = 100 * np.mean(y_true == y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            f.write("MÉTRICAS GLOBALES:\n")
            f.write(f"  Accuracy: {acc:.2f}%\n")
            f.write(f"  Precision (macro): {precision:.4f}\n")
            f.write(f"  Recall (macro): {recall:.4f}\n")
            f.write(f"  F1-Score (macro): {f1:.4f}\n")
            f.write(f"  AUC (micro): {roc_auc['micro']:.4f}\n")
            f.write(f"  AUC (macro): {roc_auc['macro']:.4f}\n")
            f.write("\n")
            
        
        print(f"Resumen de métricas guardado: {save_path / 'metrics_summary.txt'}")
