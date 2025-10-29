import os
import re
import shutil

DATASET_PATH = "data/dataset"

# Expresión regular para detectar nombres como "TIE UP_1", "TIE UP_2", etc.
pattern = re.compile(r"^(.*?)(?:_\d+)?$")

# Obtener todas las carpetas del dataset
folders = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

# Agrupar carpetas por nombre base
grouped = {}
for folder in folders:
    base_name = pattern.match(folder).group(1).strip()
    grouped.setdefault(base_name, []).append(folder)

# Fusionar carpetas que tienen versiones numeradas
for base_name, variants in grouped.items():
    if len(variants) > 1:
        print(f"Fusionando clases: {', '.join(variants)} → {base_name}")

        dest_dir = os.path.join(DATASET_PATH, base_name)
        os.makedirs(dest_dir, exist_ok=True)

        for variant in variants:
            src_dir = os.path.join(DATASET_PATH, variant)
            for file in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(dest_dir, file)

                # Evitar sobrescribir si el archivo ya existe
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file)
                    new_name = f"{base}_dup{ext}"
                    dst_file = os.path.join(dest_dir, new_name)

                shutil.move(src_file, dst_file)

            # Eliminar carpeta vacía
            shutil.rmtree(src_dir)

print("\nFusión completada")
