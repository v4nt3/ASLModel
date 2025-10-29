import os
import shutil
import re

DATASET_PATH = "data/dataset"  # ajustar

# Crear una carpeta destino 
DEST_PATH = os.path.join(DATASET_PATH, "dataset")
os.makedirs(DEST_PATH, exist_ok=True)

# Obtener todos los archivos de video
videos = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".mp4", ".mov", ".avi"))]

for video in videos:
    # Separar nombre del archivo y extensión
    name, ext = os.path.splitext(video)

    # Separar usando el primer "-"
    if "-" not in name:
        print(f"Nombre inesperado: {video}")
        continue

    _, sign_name = name.split("-", 1)

    # Eliminar espacios extra
    sign_name = sign_name.strip()

    # Dividir por espacios
    words = sign_name.split()

    # última palabra es un número => eliminarla
    if words and re.match(r"^\d+$", words[-1]):
        words = words[:-1]

    # Volver a unir el nombre final
    class_name = " ".join(words).strip()

    # Crear carpeta de la clase
    class_folder = os.path.join(DEST_PATH, class_name)
    os.makedirs(class_folder, exist_ok=True)

    # Mover el video
    src = os.path.join(DATASET_PATH, video)
    dst = os.path.join(class_folder, video)
    shutil.move(src, dst)

    print(f"{video} → {class_name}/")

print("\nDataset organizado.")
