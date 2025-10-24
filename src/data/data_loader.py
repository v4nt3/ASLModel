"""
Funciones para cargar y dividir datos eficientemente
"""
import numpy as np
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm #type: ignore
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader #type: ignore
from src.data.dataset import ASLVideoDataset, ASLPreExtractedDataset, ASLLazyFeaturesDataset


def load_all_cnn_features(cnn_dir: str, label_mapping_path: str = None):
    """
    Carga TODAS las CNN features en memoria de una vez
    Esto es más rápido que cargar archivo por archivo durante entrenamiento
    
    Args:
        cnn_dir: Directorio con features organizadas por clase
        label_mapping_path: Path al archivo label_mapping.json
    
    Returns:
        X: (N, seq_len, feat_dim) numpy array con todas las features
        y: (N,) numpy array con todas las etiquetas
        num_classes: Número total de clases
        label_mapping: Diccionario de mapeo clase -> índice
    """
    print("\n" + "="*60)
    print("CARGANDO CNN FEATURES EN MEMORIA")
    print("="*60)
    
    cnn_path = Path(cnn_dir)
    
    if label_mapping_path is None:
        label_mapping_path = cnn_path / "label_mapping.json"
    else:
        label_mapping_path = Path(label_mapping_path)
    
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"No se encontró {label_mapping_path}")
    
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    num_classes = len(label_mapping)
    print(f"✓ Cargado label_mapping.json: {num_classes} clases")
    
    label_indices = set(label_mapping.values())
    expected_indices = set(range(num_classes))
    
    if label_indices != expected_indices:
        missing = expected_indices - label_indices
        extra = label_indices - expected_indices
        print(f"\n⚠ ERROR: Inconsistencia en label_mapping!")
        if missing:
            print(f"  Índices faltantes: {sorted(missing)}")
        if extra:
            print(f"  Índices extra (fuera de rango): {sorted(extra)}")
        raise ValueError("label_mapping.json tiene índices incorrectos. Regenera con scripts/generate_label_mapping.py")
    
    # Cargar todos los CNN features
    all_features = []
    all_labels = []
    skipped_count = 0
    
    print("\nCargando CNN features desde carpetas de clases...")
    class_folders = sorted([d for d in cnn_path.iterdir() if d.is_dir()])
    
    print(f"✓ Carpetas encontradas: {len(class_folders)}")
    if len(class_folders) != num_classes:
        print(f"\n⚠ WARNING: Número de carpetas ({len(class_folders)}) != clases en label_mapping ({num_classes})")
        print(f"  Esto puede causar problemas. Considera regenerar label_mapping.json")
    
    for class_folder in tqdm(class_folders, desc="Cargando clases"):
        class_name = class_folder.name
        
        # Extraer el nombre de la clase sin el prefijo numérico
        # Formato esperado: "0001_hello" -> "hello"
        if '_' in class_name:
            parts = class_name.split('_', 1)
            if len(parts) == 2:
                _, actual_class_name = parts
            else:
                actual_class_name = class_name
        else:
            actual_class_name = class_name
        
        # Reemplazar guiones bajos por espacios
        actual_class_name = actual_class_name.replace('_', ' ')
        
        # Buscar en label_mapping
        if actual_class_name not in label_mapping:
            # Intentar variaciones comunes
            variations = [
                actual_class_name.lower(),
                actual_class_name.upper(),
                actual_class_name.title(),
            ]
            
            found = False
            for var in variations:
                if var in label_mapping:
                    actual_class_name = var
                    found = True
                    break
            
            if not found:
                print(f"\n⚠ Clase '{actual_class_name}' no encontrada en label_mapping")
                skipped_count += 1
                continue
        
        class_idx = label_mapping[actual_class_name]
        
        if class_idx < 0 or class_idx >= num_classes:
            print(f"\n⚠ ERROR: Clase '{actual_class_name}' tiene índice fuera de rango: {class_idx}")
            print(f"  Rango válido: [0, {num_classes-1}]")
            skipped_count += 1
            continue
        
        # Cargar todos los archivos .npy de esta clase
        npy_files = list(class_folder.glob("*.npy"))
        
        for npy_file in npy_files:
            try:
                features = np.load(npy_file)  # Shape: (30, 2048)
                
                if len(features.shape) != 2 or features.shape[0] != 30:
                    print(f"\n⚠ Shape incorrecto en {npy_file}: {features.shape}")
                    continue
                
                all_features.append(features)
                all_labels.append(class_idx)
            except Exception as e:
                print(f"\n⚠ Error cargando {npy_file}: {e}")
    
    if skipped_count > 0:
        print(f"\n⚠ Se omitieron {skipped_count} carpetas por problemas de mapeo")
    
    # Convertir a arrays numpy
    X = np.array(all_features, dtype=np.float32)  # (N, 30, 2048)
    y = np.array(all_labels, dtype=np.int64)
    
    print(f"\n✓ CNN features cargados:")
    print(f"  Total samples: {len(X)}")
    print(f"  Shape: {X.shape}")
    print(f"  Clases únicas: {len(np.unique(y))}")
    print(f"  Rango de etiquetas: [{y.min()}, {y.max()}]")
    
    if y.max() >= num_classes:
        print(f"\n❌ ERROR CRÍTICO: Etiqueta máxima ({y.max()}) >= num_classes ({num_classes})")
        print(f"  Esto significa que hay etiquetas fuera de rango.")
        print(f"  Para {num_classes} clases, las etiquetas deben estar en [0, {num_classes-1}]")
        print(f"\n  SOLUCIÓN: Regenera label_mapping.json ejecutando:")
        print(f"    python scripts/generate_label_mapping.py")
        raise ValueError("Etiquetas fuera de rango!")
    
    if y.min() < 0:
        raise ValueError(f"Etiquetas negativas encontradas! Min: {y.min()}")
    
    unique_labels = set(y.tolist())
    missing_classes = expected_indices - unique_labels
    if missing_classes:
        print(f"\n⚠ WARNING: {len(missing_classes)} clases sin samples:")
        if len(missing_classes) <= 10:
            for idx in sorted(missing_classes):
                class_name = [k for k, v in label_mapping.items() if v == idx][0]
                print(f"    Clase {idx}: {class_name}")
    
    return X, y, num_classes, label_mapping


def create_train_val_test_splits(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Crea splits estratificados de train/val/test
    
    Args:
        X: Features array (N, seq_len, feat_dim)
        y: Labels array (N,)
        test_size: Proporción para test
        val_size: Proporción para validation
        random_state: Semilla aleatoria
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print(f"\nCreando splits (test={test_size}, val={val_size})...")
    
    # Primero separar test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Luego separar train y val del resto
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\n✓ Splits creados:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\n✓ Distribución de clases:")
    print(f"  Train: {len(np.unique(y_train))} clases únicas")
    print(f"  Val:   {len(np.unique(y_val))} clases únicas")
    print(f"  Test:  {len(np.unique(y_test))} clases únicas")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_video_metadata(metadata_csv: str):
    """
    Carga metadata de videos desde CSV
    
    Args:
        metadata_csv: Path al archivo CSV con columnas [video_path, label, split]
    
    Returns:
        DataFrame con metadata
    """
    df = pd.read_csv(metadata_csv)
    
    required_cols = ['video_path', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV debe contener columna '{col}'")
    
    print(f"\n✓ Metadata cargada: {len(df)} videos")
    print(f"  Clases únicas: {df['label'].nunique()}")
    
    return df


def create_dataloaders_from_videos(video_paths, labels, frame_extractor, 
                                   augmenter=None, batch_size=32, 
                                   num_workers=4, pin_memory=True,
                                   train_split=0.7, val_split=0.15, 
                                   random_state=42):
    """
    Crea dataloaders para train/val/test desde videos
    
    Args:
        video_paths: Lista de rutas a videos
        labels: Lista de etiquetas
        frame_extractor: VideoFrameExtractor
        augmenter: FrameAugmenter (opcional)
        batch_size: Tamaño de batch
        num_workers: Workers para DataLoader
        pin_memory: Pin memory para GPU
        train_split: Proporción de train
        val_split: Proporción de validation
        random_state: Semilla aleatoria
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Convertir a arrays numpy para split
    video_paths = np.array(video_paths)
    labels = np.array(labels)
    
    # Split train/temp
    test_size = 1 - train_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        video_paths, labels, test_size=test_size, 
        random_state=random_state, stratify=labels
    )
    
    # Split val/test
    val_size_adjusted = val_split / test_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_size_adjusted),
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\n✓ Splits creados:")
    print(f"  Train: {len(X_train)} videos")
    print(f"  Val:   {len(X_val)} videos")
    print(f"  Test:  {len(X_test)} videos")
    
    # Crear datasets
    train_dataset = ASLVideoDataset(
        X_train.tolist(), y_train.tolist(), 
        frame_extractor, augmenter, use_augmentation=True
    )
    val_dataset = ASLVideoDataset(
        X_val.tolist(), y_val.tolist(),
        frame_extractor, augmenter=None, use_augmentation=False
    )
    test_dataset = ASLVideoDataset(
        X_test.tolist(), y_test.tolist(),
        frame_extractor, augmenter=None, use_augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def create_dataloaders_from_features(features, labels, batch_size=32,
                                     num_workers=4, pin_memory=True,
                                     train_split=0.7, val_split=0.15,
                                     random_state=42):
    """
    Crea dataloaders desde features pre-extraídas (más rápido)
    
    Args:
        features: Tensor o array de features (N, seq_len, feat_dim)
        labels: Tensor o array de labels (N,)
        batch_size: Tamaño de batch
        num_workers: Workers para DataLoader
        pin_memory: Pin memory para GPU
        train_split: Proporción de train
        val_split: Proporción de validation
        random_state: Semilla aleatoria
    
    Returns:
        train_loader, val_loader, test_loader
    """
    import torch #type: ignore
    
    # Convertir a tensors si es necesario
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features).float()
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels).long()
    
    # Crear splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_splits(
        features.numpy(), labels.numpy(), 
        test_size=(1-train_split-val_split), 
        val_size=val_split,
        random_state=random_state
    )
    
    # Convertir de vuelta a tensors
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()
    
    # Crear datasets
    train_dataset = ASLPreExtractedDataset(X_train, y_train)
    val_dataset = ASLPreExtractedDataset(X_val, y_val)
    test_dataset = ASLPreExtractedDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def scan_cnn_features_lazy(cnn_dir: str, label_mapping_path: str = None):
    """
    Escanea CNN features sin cargarlas en memoria (lazy loading)
    Retorna listas de paths y labels para procesamiento eficiente
    
    Args:
        cnn_dir: Directorio con features organizadas por clase
        label_mapping_path: Path al archivo label_mapping.json
    
    Returns:
        feature_paths: Lista de paths a archivos .npy
        labels: Lista de etiquetas correspondientes
        num_classes: Número total de clases
        label_mapping: Diccionario de mapeo clase -> índice
    """
    print("\n" + "="*60)
    print("ESCANEANDO CNN FEATURES (LAZY LOADING)")
    print("="*60)
    
    cnn_path = Path(cnn_dir)
    
    if label_mapping_path is None:
        label_mapping_path = cnn_path / "label_mapping.json"
    else:
        label_mapping_path = Path(label_mapping_path)
    
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"No se encontró {label_mapping_path}")
    
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    num_classes = len(label_mapping)
    print(f"✓ Cargado label_mapping.json: {num_classes} clases")
    
    label_indices = set(label_mapping.values())
    expected_indices = set(range(num_classes))
    
    if label_indices != expected_indices:
        missing = expected_indices - label_indices
        extra = label_indices - expected_indices
        print(f"\n⚠ ERROR: Inconsistencia en label_mapping!")
        if missing:
            print(f"  Índices faltantes: {sorted(missing)}")
        if extra:
            print(f"  Índices extra (fuera de rango): {sorted(extra)}")
        raise ValueError("label_mapping.json tiene índices incorrectos. Regenera con scripts/fix_label_mapping.py")
    
    feature_paths = []
    labels = []
    skipped_count = 0
    
    print("\nEscaneando archivos .npy...")
    class_folders = sorted([d for d in cnn_path.iterdir() if d.is_dir()])
    
    print(f"✓ Carpetas encontradas: {len(class_folders)}")
    
    for class_folder in tqdm(class_folders, desc="Escaneando clases"):
        class_name = class_folder.name
        
        # Extraer el nombre de la clase sin el prefijo numérico
        if '_' in class_name:
            parts = class_name.split('_', 1)
            if len(parts) == 2:
                _, actual_class_name = parts
            else:
                actual_class_name = class_name
        else:
            actual_class_name = class_name
        
        # Reemplazar guiones bajos por espacios
        actual_class_name = actual_class_name.replace('_', ' ')
        
        # Buscar en label_mapping
        if actual_class_name not in label_mapping:
            # Intentar variaciones comunes
            variations = [
                actual_class_name.lower(),
                actual_class_name.upper(),
                actual_class_name.title(),
            ]
            
            found = False
            for var in variations:
                if var in label_mapping:
                    actual_class_name = var
                    found = True
                    break
            
            if not found:
                skipped_count += 1
                continue
        
        class_idx = label_mapping[actual_class_name]
        
        if class_idx < 0 or class_idx >= num_classes:
            skipped_count += 1
            continue
        
        npy_files = list(class_folder.glob("*.npy"))
        
        for npy_file in npy_files:
            feature_paths.append(str(npy_file))
            labels.append(class_idx)
    
    if skipped_count > 0:
        print(f"\n⚠ Se omitieron {skipped_count} carpetas por problemas de mapeo")
    
    print(f"\n✓ Archivos escaneados:")
    print(f"  Total samples: {len(feature_paths)}")
    print(f"  Clases únicas: {len(set(labels))}")
    print(f"  Rango de etiquetas: [{min(labels)}, {max(labels)}]")
    print(f"  Memoria usada: ~{len(feature_paths) * 100 / 1024:.2f} KB (solo paths)")
    
    if max(labels) >= num_classes:
        print(f"\n❌ ERROR: Etiqueta máxima ({max(labels)}) >= num_classes ({num_classes})")
        raise ValueError("Etiquetas fuera de rango! Regenera label_mapping.json")
    
    return feature_paths, labels, num_classes, label_mapping


def create_dataloaders_from_features_lazy(cnn_dir: str, 
                                          label_mapping_path: str = None,
                                          batch_size: int = 32,
                                          num_workers: int = 4, 
                                          pin_memory: bool = True,
                                          prefetch_factor: int = 2,
                                          persistent_workers: bool = False,
                                          train_split: float = 0.7, 
                                          val_split: float = 0.15,
                                          random_state: int = 42):
    """
    Crea dataloaders usando lazy loading (bajo uso de memoria)
    
    Args:
        cnn_dir: Directorio con features
        label_mapping_path: Path al label_mapping.json
        batch_size: Tamaño de batch
        num_workers: Workers para DataLoader
        pin_memory: Pin memory para GPU
        prefetch_factor: Número de batches a pre-cargar por worker
        persistent_workers: Mantener workers vivos entre épocas
        train_split: Proporción de train
        val_split: Proporción de validation
        random_state: Semilla aleatoria
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, label_mapping
    """
    
    feature_paths, labels, num_classes, label_mapping = scan_cnn_features_lazy(
        cnn_dir, label_mapping_path
    )
    
    feature_paths = np.array(feature_paths)
    labels = np.array(labels)
    
    print(f"\nCreando splits (train={train_split}, val={val_split})...")
    
    test_size = 1 - train_split - val_split
    
    # Primero separar test
    paths_temp, paths_test, y_temp, y_test = train_test_split(
        feature_paths, labels, test_size=test_size, 
        random_state=random_state, stratify=labels
    )
    
    # Luego separar train y val
    val_size_adjusted = val_split / (train_split + val_split)
    paths_train, paths_val, y_train, y_val = train_test_split(
        paths_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\n✓ Splits creados:")
    print(f"  Train: {len(paths_train)} samples ({len(paths_train)/len(feature_paths)*100:.1f}%)")
    print(f"  Val:   {len(paths_val)} samples ({len(paths_val)/len(feature_paths)*100:.1f}%)")
    print(f"  Test:  {len(paths_test)} samples ({len(paths_test)/len(feature_paths)*100:.1f}%)")
    
    train_dataset = ASLLazyFeaturesDataset(paths_train.tolist(), y_train.tolist())
    val_dataset = ASLLazyFeaturesDataset(paths_val.tolist(), y_val.tolist())
    test_dataset = ASLLazyFeaturesDataset(paths_test.tolist(), y_test.tolist())
    
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    # Solo agregar prefetch_factor y persistent_workers si num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['persistent_workers'] = persistent_workers
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True,
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    print(f"\n✓ DataLoaders creados con lazy loading")
    print(f"  Memoria estimada por batch: ~{batch_size * 30 * 2048 * 4 / 1024 / 1024:.2f} MB")
    print(f"  Num workers: {num_workers}")
    print(f"  Prefetch factor: {prefetch_factor if num_workers > 0 else 'N/A'}")
    print(f"  Persistent workers: {persistent_workers if num_workers > 0 else 'N/A'}")
    print(f"  Batches pre-cargados: ~{num_workers * prefetch_factor if num_workers > 0 else 0}")
    
    return train_loader, val_loader, test_loader, num_classes, label_mapping


def create_dataloaders_from_consolidated(features_dir: str,
                                        batch_size: int = 128,
                                        num_workers: int = 12,
                                        pin_memory: bool = True,
                                        prefetch_factor: int = 4,
                                        persistent_workers: bool = True,
                                        train_split: float = 0.7,
                                        val_split: float = 0.15,
                                        random_state: int = 42):
    """
    Crea dataloaders desde features consolidadas (MUCHO MÁS RÁPIDO)
    
    Args:
        features_dir: Directorio con archivos consolidados
        batch_size: Tamaño de batch
        num_workers: Workers para DataLoader
        pin_memory: Pin memory para GPU
        prefetch_factor: Batches a pre-cargar por worker
        persistent_workers: Mantener workers vivos
        train_split: Proporción de train
        val_split: Proporción de validation
        random_state: Semilla aleatoria
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, label_mapping
    """
    from src.data.dataset import ASLConsolidatedDataset
    
    print("\n" + "="*60)
    print("CARGANDO FEATURES CONSOLIDADAS (RÁPIDO)")
    print("="*60)
    
    features_dir = Path(features_dir)
    
    # Archivos consolidados
    features_file = features_dir / "features_consolidated.npy"
    labels_file = features_dir / "labels_consolidated.npy"
    label_mapping_file = features_dir / "label_mapping.json"
    
    # Verificar que existan
    if not features_file.exists():
        raise FileNotFoundError(
            f"No se encontró {features_file}\n"
            f"Ejecuta primero: python scripts/consolidate_features.py"
        )
    
    if not labels_file.exists():
        raise FileNotFoundError(f"No se encontró {labels_file}")
    
    if not label_mapping_file.exists():
        raise FileNotFoundError(f"No se encontró {label_mapping_file}")
    
    # Cargar label mapping
    with open(label_mapping_file, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    num_classes = len(label_mapping)
    print(f"✓ Label mapping: {num_classes} clases")
    
    # Cargar labels para hacer splits
    labels = np.load(labels_file, mmap_mode='r')
    total_samples = len(labels)
    
    print(f"✓ Total samples: {total_samples}")
    
    # Crear índices para splits
    indices = np.arange(total_samples)
    
    test_size = 1 - train_split - val_split
    
    # Split train/temp
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        indices, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )
    
    # Split train/val
    val_size_adjusted = val_split / (train_split + val_split)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\n✓ Splits creados:")
    print(f"  Train: {len(idx_train)} samples ({len(idx_train)/total_samples*100:.1f}%)")
    print(f"  Val:   {len(idx_val)} samples ({len(idx_val)/total_samples*100:.1f}%)")
    print(f"  Test:  {len(idx_test)} samples ({len(idx_test)/total_samples*100:.1f}%)")
    
    # Crear datasets con memory mapping
    train_dataset = ASLConsolidatedDataset(
        str(features_file), str(labels_file), idx_train.tolist()
    )
    val_dataset = ASLConsolidatedDataset(
        str(features_file), str(labels_file), idx_val.tolist()
    )
    test_dataset = ASLConsolidatedDataset(
        str(features_file), str(labels_file), idx_test.tolist()
    )
    
    # Configuración de DataLoader
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['persistent_workers'] = persistent_workers
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    
    print(f"\n✓ DataLoaders creados (MODO RÁPIDO)")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Prefetch factor: {prefetch_factor if num_workers > 0 else 'N/A'}")
    print(f"  Persistent workers: {persistent_workers if num_workers > 0 else 'N/A'}")
    print(f"  Memory mapping: Activo (sin cargar en RAM)")
    
    return train_loader, val_loader, test_loader, num_classes, label_mapping
