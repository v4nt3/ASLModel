"""
Script to prepare dataset by creating train/val/test split JSON files
from raw video directory structure.

Expected directory structure:
data/videos/
    class_0/
        video1.mp4
        video2.mp4
    class_1/
        video1.mp4
        ...
"""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict

CONFIG = {
    'data_dir': 'data/dataset',  # Directory containing class folders with videos
    'output_dir': 'data',  # Directory to save split JSON files
    'train_ratio': 0.7,  # Proportion of data for training
    'val_ratio': 0.15,  # Proportion of data for validation
    'test_ratio': 0.15,  # Proportion of data for testing
    'seed': 42,  # Random seed for reproducibility
    'video_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm']  # Video file extensions
}
# ============================================================================

def create_split_files(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, 
                       test_ratio=0.1, seed=42, video_extensions=None):
    """
    Create train/val/test split files from video directory structure
    
    Args:
        data_dir: Directory containing class folders with videos
        output_dir: Directory to save JSON split files
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
        video_extensions: List of video file extensions to look for
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    np.random.seed(seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning directory: {data_dir}")
    
    # Collect all videos organized by class
    class_videos = defaultdict(list)
    class_to_idx = {}
    
    # Get all class directories
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir}")
    
    print(f"Found {len(class_dirs)} class directories")
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_to_idx[class_name] = class_idx
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(class_dir.glob(f'*{ext}')))
        
        for video_path in video_files:
            class_videos[class_idx].append({
                'video_path': str(video_path.relative_to(data_dir)),
                'label': class_idx,
                'class_name': class_name
            })
        
        if len(video_files) > 0:
            print(f"  Class {class_idx} ({class_name}): {len(video_files)} videos")
    
    # Stratified split - maintain class distribution in each split
    train_data = []
    val_data = []
    test_data = []
    
    class_distribution = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}
    
    for class_idx, videos in class_videos.items():
        if len(videos) == 0:
            continue
        
        # Shuffle videos for this class
        np.random.shuffle(videos)
        
        n_total = len(videos)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        
        # Ensure we have at least one sample in each split if possible
        if n_total >= 3:
            train_videos = videos[:n_train]
            val_videos = videos[n_train:n_train + n_val]
            test_videos = videos[n_train + n_val:]
        elif n_total == 2:
            train_videos = videos[:1]
            val_videos = videos[1:2]
            test_videos = []
        else:  # n_total == 1
            train_videos = videos
            val_videos = []
            test_videos = []
        
        train_data.extend(train_videos)
        val_data.extend(val_videos)
        test_data.extend(test_videos)
        
        class_distribution['train'][class_idx] = len(train_videos)
        class_distribution['val'][class_idx] = len(val_videos)
        class_distribution['test'][class_idx] = len(test_videos)
    
    # Shuffle the final splits
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    # Save splits
    train_file = output_dir / 'train.json'
    val_file = output_dir / 'val.json'
    test_file = output_dir / 'test.json'
    class_file = output_dir / 'class_to_idx.json'
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    with open(class_file, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    # Calculate and save class weights for handling imbalance
    class_counts = np.array([class_distribution['train'][i] for i in range(len(class_to_idx))])
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_to_idx) * class_counts + 1e-6)
    
    weights_file = output_dir / 'class_weights.json'
    with open(weights_file, 'w') as f:
        json.dump(class_weights.tolist(), f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Split Summary")
    print("="*60)
    print(f"Total classes: {len(class_to_idx)}")
    print(f"Total videos: {len(train_data) + len(val_data) + len(test_data)}")
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_data)} videos ({len(train_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
    print(f"  Val:   {len(val_data)} videos ({len(val_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
    print(f"  Test:  {len(test_data)} videos ({len(test_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
    
    print(f"\nClass distribution statistics:")
    train_counts = [class_distribution['train'][i] for i in range(len(class_to_idx))]
    print(f"  Train - Min: {min(train_counts)}, Max: {max(train_counts)}, Mean: {np.mean(train_counts):.1f}")
    
    print(f"\nFiles saved to: {output_dir}")
    print(f"  - {train_file.name}")
    print(f"  - {val_file.name}")
    print(f"  - {test_file.name}")
    print(f"  - {class_file.name}")
    print(f"  - {weights_file.name}")
    print("="*60)
    
    # Save detailed statistics
    stats = {
        'total_classes': len(class_to_idx),
        'total_videos': len(train_data) + len(val_data) + len(test_data),
        'train_videos': len(train_data),
        'val_videos': len(val_data),
        'test_videos': len(test_data),
        'class_distribution': {
            'train': dict(class_distribution['train']),
            'val': dict(class_distribution['val']),
            'test': dict(class_distribution['test'])
        }
    }
    
    stats_file = output_dir / 'dataset_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDetailed statistics saved to: {stats_file}")


def main():
    print("="*60)
    print("Dataset Preparation Script")
    print("="*60)
    print(f"Data directory: {CONFIG['data_dir']}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print(f"Train/Val/Test ratio: {CONFIG['train_ratio']}/{CONFIG['val_ratio']}/{CONFIG['test_ratio']}")
    print(f"Random seed: {CONFIG['seed']}")
    print("="*60)
    
    # Validate ratios
    total_ratio = CONFIG['train_ratio'] + CONFIG['val_ratio'] + CONFIG['test_ratio']
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")
    
    create_split_files(
        data_dir=CONFIG['data_dir'],
        output_dir=CONFIG['output_dir'],
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        test_ratio=CONFIG['test_ratio'],
        seed=CONFIG['seed'],
        video_extensions=CONFIG['video_extensions']
    )


if __name__ == '__main__':
    main()
