import os
import re
import shutil

DATASET_PATH = "data/dataset"

# Expresión regular para detectar nombres como "TIE UP_1", "TIE UP_2"
pattern = re.compile(r"^(.*?)(?:_\d+)?$")

# Obtener carpetas del dataset
folders = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

# Agrupar por nombre base
grouped = {}
for folder in folders:
    base_name = pattern.match(folder).group(1).strip()
    grouped.setdefault(base_name, []).append(folder)

# Fusionar carpetas
for base_name, variants in grouped.items():
    if len(variants) > 1:
        print(f"Fusionando clases: {', '.join(variants)} → {base_name}")
        dest_dir = os.path.join(DATASET_PATH, base_name)
        os.makedirs(dest_dir, exist_ok=True)

        for variant in variants:
            src_dir = os.path.join(DATASET_PATH, variant)
            if not os.path.exists(src_dir):
                print(f"Carpeta no encontrada: {src_dir}, saltando...")
                continue

            for file in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file)
                if not os.path.exists(src_file):
                    print(f"Archivo no encontrado: {src_file}, saltando...")
                    continue

                dst_file = os.path.join(dest_dir, file)

                # Evitar sobrescrituras
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file)
                    dst_file = os.path.join(dest_dir, f"{base}_dup{ext}")

                try:
                    shutil.move(src_file, dst_file)
                except Exception as e:
                    print(f"Error moviendo {src_file} → {dst_file}: {e}")

            # Eliminar carpeta si queda vacía
            try:
                os.rmdir(src_dir)
            except OSError:
                pass

print("\nFusión completada con verificación de existencia.")
