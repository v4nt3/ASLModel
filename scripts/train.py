"""
Script de entrenamiento para el modelo ASL Recognition
Puede ser ejecutado independientemente o llamado desde main.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts #type: ignore
import json

from config.config1 import Config
from src.data.data_loader import (
    create_dataloaders_from_features_lazy,
    create_dataloaders_from_consolidated
)
from src.models.asl_classifier import ASLClassifierPreExtracted
from src.training.trainer import Trainer
from src.visualization.visualizer import AdvancedVisualizer


def setup_model(config: Config, device: str, num_classes: int):
    """Inicializar modelo"""
    print("\n" + "="*60)
    print("CONFIGURACIÓN DEL MODELO")
    print("="*60)
    
    model = ASLClassifierPreExtracted(
        num_classes=num_classes,
        feature_dim=config.FEATURE_DIM,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_dropout=config.LSTM_DROPOUT,
        lstm_bidirectional=config.LSTM_BIDIRECTIONAL,
        use_attention=config.USE_ATTENTION,
        classifier_hidden_dims=config.CLASSIFIER_HIDDEN_DIMS,
        classifier_dropout=config.CLASSIFIER_DROPOUT
    )
    
    model = model.to(device)
    
    if config.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        print("Compilando modelo con torch.compile()...")
        try:
            model = torch.compile(model)
            print("✓ Modelo compilado exitosamente")
        except Exception as e:
            print(f"⚠ No se pudo compilar el modelo: {e}")
    
    print(f"Modelo: ASLClassifierPreExtracted")
    print(f"Feature Dim: {config.FEATURE_DIM}")
    print(f"Hidden Size: {config.LSTM_HIDDEN_SIZE}")
    print(f"LSTM Layers: {config.LSTM_NUM_LAYERS}")
    print(f"Bidirectional: {config.LSTM_BIDIRECTIONAL}")
    print(f"Attention: {config.USE_ATTENTION}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*60)
    
    return model


def setup_training(model, config: Config):
    """Configurar optimizador, loss y scheduler"""
    print("\n" + "="*60)
    print("CONFIGURACIÓN DE ENTRENAMIENTO")
    print("="*60)
    
    # Optimizer
    if config.OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            momentum=0.9
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    # Scheduler
    if config.SCHEDULER_TYPE == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',  # Monitorear val_loss
            factor=0.5,  # Reducir LR a la mitad
            patience=5,  # Esperar 5 épocas sin mejora
            min_lr=config.MIN_LR
        )
    elif config.SCHEDULER_TYPE == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.SCHEDULER_T0,
            T_mult=config.SCHEDULER_TMULT,
            eta_min=config.MIN_LR
        )
    else:  # step
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print(f"Optimizer: {config.OPTIMIZER}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Loss: CrossEntropyLoss (label_smoothing={config.LABEL_SMOOTHING})")
    print(f"Scheduler: {config.SCHEDULER_TYPE}")
    print("="*60)
    
    return optimizer, criterion, scheduler


def train_model():
    """
    Entrenar el modelo ASL Recognition usando configuración centralizada
    """
    config = Config()
    
    # Device
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"DISPOSITIVO: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}")
    
    if config.USE_CONSOLIDATED_FEATURES:
        consolidated_file = config.FEATURES_DIR / "features_consolidated.npy"
        
        if consolidated_file.exists():
            print("\n✓ Usando features consolidadas (MODO RÁPIDO)")
            train_loader, val_loader, test_loader, num_classes, label_mapping = create_dataloaders_from_consolidated(
                features_dir=str(config.FEATURES_DIR),
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                prefetch_factor=config.PREFETCH_FACTOR,
                persistent_workers=config.PERSISTENT_WORKERS,
                train_split=config.TRAIN_SPLIT,
                val_split=config.VAL_SPLIT,
                random_state=config.RANDOM_SEED
            )
        else:
            print("\n⚠ Features consolidadas no encontradas")
            
            train_loader, val_loader, test_loader, num_classes, label_mapping = create_dataloaders_from_features_lazy(
                cnn_dir=str(config.FEATURES_DIR),
                label_mapping_path=str(config.FEATURES_DIR / "label_mapping.json"),
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                prefetch_factor=config.PREFETCH_FACTOR,
                persistent_workers=config.PERSISTENT_WORKERS,
                train_split=config.TRAIN_SPLIT,
                val_split=config.VAL_SPLIT,
                random_state=config.RANDOM_SEED
            )
    else:
        print("\nUsando lazy loading (bajo uso de memoria)")
        train_loader, val_loader, test_loader, num_classes, label_mapping = create_dataloaders_from_features_lazy(
            cnn_dir=str(config.FEATURES_DIR),
            label_mapping_path=str(config.FEATURES_DIR / "label_mapping.json"),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            prefetch_factor=config.PREFETCH_FACTOR,
            persistent_workers=config.PERSISTENT_WORKERS,
            train_split=config.TRAIN_SPLIT,
            val_split=config.VAL_SPLIT,
            random_state=config.RANDOM_SEED
        )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print(f"  Clases: {num_classes}")
    
    # Configurar modelo
    model = setup_model(config, device, num_classes)
    
    # Configurar entrenamiento
    optimizer, criterion, scheduler = setup_training(model, config)
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        class_names=list(label_mapping.keys()),
        checkpoint_dir=str(config.CHECKPOINTS_DIR),
        log_interval=config.LOG_INTERVAL,
        use_amp=config.USE_AMP,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        gradient_clip=config.GRADIENT_CLIP
    )
    
    # Entrenar
    trainer.train(
        num_epochs=config.NUM_EPOCHS,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )
    
    # Evaluar en test set
    test_results = trainer.test()
    
    # ------------------ INICIO: Bloque adicional de visualizaciones ------------------
    import torch.nn.functional as F #type: ignore
    import numpy as np

    def collect_test_predictions(model, loader, device):
        """Recorre el test_loader y devuelve y_true, y_pred, y_pred_proba (numpy)."""
        model.eval()
        y_trues = []
        y_preds = []
        y_probs = []

        with torch.no_grad():
            for batch in loader:
                # Intenta desempaquetar de forma flexible
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                elif isinstance(batch, dict):
                    # comunes: {'inputs': x, 'labels': y} o {'features': x, 'labels': y}
                    if 'inputs' in batch:
                        inputs, labels = batch['inputs'], batch.get('labels')
                    elif 'features' in batch:
                        inputs, labels = batch['features'], batch.get('labels')
                    else:
                        # toma los dos primeros valores del dict
                        vals = list(batch.values())
                        inputs, labels = vals[0], vals[1] if len(vals) > 1 else (vals[0], None)
                else:
                    # último recurso: esperar que batch sea (x, y)
                    try:
                        inputs, labels = batch
                    except Exception as e:
                        raise RuntimeError(
                            "No se pudo desempaquetar el batch en collect_test_predictions. "
                            "Ajusta el código según la estructura de tus DataLoaders."
                        ) from e

                inputs = inputs.to(device)
                outputs = trainer.model(inputs) if hasattr(trainer, 'model') else model(inputs)
                # Si outputs es un tupla (por ejemplo (logits, ...)), tomar first
                if isinstance(outputs, (list, tuple)):
                    logits = outputs[0]
                else:
                    logits = outputs

                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                # Manejar labels (pueden ser one-hot)
                if labels is None:
                    raise RuntimeError("Los batches no contienen etiquetas 'labels' - necesario para las gráficas.")
                if hasattr(labels, 'cpu'):
                    lab_np = labels.cpu().numpy()
                else:
                    lab_np = np.array(labels)

                # Si labels son one-hot -> convertir a indices
                if lab_np.ndim > 1 and lab_np.shape[1] > 1:
                    lab_np = np.argmax(lab_np, axis=1)

                y_trues.extend(lab_np.tolist())
                y_preds.extend(preds.tolist())
                y_probs.extend(probs.tolist())

        return np.array(y_trues), np.array(y_preds), np.vstack(y_probs)


    print("\nGenerando visualizaciones extendidas (confusion, per-class, ROC, worst-classes, reporte)...")

    # Recolectar predicciones (usa trainer.model para asegurar que sea el modelo final entrenado)
    y_true, y_pred, y_pred_proba = collect_test_predictions(
        model=trainer.model if hasattr(trainer, 'model') else model,
        loader=test_loader,
        device=device
    )

    # Inicializar visualizador (usa la misma configuración que antes)
    advanced_visualizer = AdvancedVisualizer()

    advanced_visualizer.plot_training_history(
        trainer.metrics_tracker.history,
        save_path=config.VISUALIZATIONS_DIR / 'training_history.png'
    )

    # 1) Gráficas adicionales sobre history (gap, comparaciones)
    try:
        advanced_visualizer.plot_metrics_comparison(trainer.metrics_tracker.history,
                                                save_path=config.VISUALIZATIONS_DIR / 'overfitting_analysis.png')
    except Exception as e:
        print(f"⚠ No se pudo generar plot_metrics_comparison: {e}")

    # 2) Matriz de confusión
    try:
        advanced_visualizer.plot_confusion_matrix(y_true, y_pred,
                                                class_names=list(label_mapping.keys()),
                                                save_path=config.VISUALIZATIONS_DIR / 'confusion_matrix.png')
    except Exception as e:
        print(f"⚠ No se pudo generar matriz de confusión: {e}")

    # 3) Métricas por clase
    try:
        advanced_visualizer.plot_per_class_metrics(y_true, y_pred,
                                                class_names=list(label_mapping.keys()),
                                                save_path=config.VISUALIZATIONS_DIR / 'per_class_metrics.png')
    except Exception as e:
        print(f"⚠ No se pudo generar métricas por clase: {e}")

    # 4) Curvas ROC (necesita probabilidades)
    try:
        advanced_visualizer.plot_roc_curves(y_true, y_pred_proba,
                                            class_names=list(label_mapping.keys()),
                                            save_path=config.VISUALIZATIONS_DIR / 'roc_curves.png')
    except Exception as e:
        print(f"⚠ No se pudo generar curvas ROC: {e}")

    # 5) Peores clases
    try:
        advanced_visualizer.plot_worst_classes(y_true, y_pred,
                                            class_names=list(label_mapping.keys()),
                                            top_n=20,
                                            save_path=config.VISUALIZATIONS_DIR / 'worst_classes.png')
    except Exception as e:
        print(f"⚠ No se pudo generar análisis de peores clases: {e}")

    # 6) Reporte completo (guarda todo en carpeta)
    try:
        advanced_visualizer.generate_complete_report(y_true, y_pred, y_pred_proba,
                                                    class_names=list(label_mapping.keys()),
                                                    save_dir=str(config.VISUALIZATIONS_DIR))
    except Exception as e:
        print(f"⚠ No se pudo generar el reporte completo: {e}")

    print("Visualizaciones extendidas generadas y guardadas en:", config.VISUALIZATIONS_DIR)
    # ------------------ FIN: Bloque adicional de visualizaciones ------------------

    
    # Guardar resultados finales
    results = {
        'best_epoch': trainer.best_epoch,
        'best_val_acc': trainer.best_val_acc,
        'test_acc': test_results['test_acc'],
        'test_top5': test_results['test_top5'],
        'test_loss': test_results['test_loss']
    }
    
    results_path = config.VISUALIZATIONS_DIR / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"Mejor época: {trainer.best_epoch}")
    print(f"Mejor Val Acc: {trainer.best_val_acc:.2f}%")
    print(f"Test Acc: {test_results['test_acc']:.2f}%")
    print(f"Test Top-5: {test_results['test_top5']:.2f}%")
    print(f"\nResultados guardados en: {results_path}")
    print(f"Checkpoints guardados en: {config.CHECKPOINTS_DIR}")
    print(f"Visualizaciones guardadas en: {config.VISUALIZATIONS_DIR}")
    print(f"{'='*60}")


def main():
    """Ejecutar entrenamiento"""
    train_model()


if __name__ == '__main__':
    main()
