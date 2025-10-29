import os
import shutil

DATASET_PATH = "data/dataset"

# Buscar carpetas que empiecen con "seed"
seed_folders = [d for d in os.listdir(DATASET_PATH)
                if os.path.isdir(os.path.join(DATASET_PATH, d)) and d.lower().startswith("seed")]

if seed_folders:
    print(f"\nEliminando {len(seed_folders)} carpetas que empiezan con 'seed':")
    for folder in seed_folders:
        path = os.path.join(DATASET_PATH, folder)
        try:
            shutil.rmtree(path)
            print(f"   {folder} eliminada")
        except Exception as e:
            print(f"   Error al eliminar {folder}: {e}")
else:
    print("\nNo se encontraron carpetas que comiencen con 'seed'.")
