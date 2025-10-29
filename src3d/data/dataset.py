import os 
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ruta base del dataset
DATASET_DIR = Path("data/dataset")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Porcentajes
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def get_all_videos():
    """Recorre todas las carpetas de clases y retorna lista (video_path, class_name)"""
    all_videos = []
    for class_dir in DATASET_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        videos = list(class_dir.glob("*.mp4"))
        for video in videos:
            all_videos.append({
                "path": str(video.relative_to(DATASET_DIR)), 
                "label": class_dir.name
            })
    return all_videos

def split_dataset(videos):
    """Divide dataset en train, val y test con proporciones 70/15/15"""
    random.shuffle(videos)
    labels = [v["label"] for v in videos]

    train_videos, temp_videos = train_test_split(videos, test_size=(1 - TRAIN_SPLIT), stratify=labels)
    val_videos, test_videos = train_test_split(
        temp_videos, 
        test_size=TEST_SPLIT / (TEST_SPLIT + VAL_SPLIT),
        stratify=[v["label"] for v in temp_videos]
    )

    return train_videos, val_videos, test_videos

def save_json(data, filename):
    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"{filename} creado con {len(data)} muestras")

def main():
    print("Recorriendo dataset...")
    all_videos = get_all_videos()
    print(f"Total de videos encontrados: {len(all_videos)}")

    print("Dividiendo dataset en train/val/test...")
    train_data, val_data, test_data = split_dataset(all_videos)

    print("Guardando archivos JSON...")
    save_json(train_data, "train.json")
    save_json(val_data, "val.json")
    save_json(test_data, "test.json")

    print("\nDivisión completada correctamente!")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    main()
