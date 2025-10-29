import os

DATASET_PATH = "data/dataset"

clases_pocas = []

for clase in os.listdir(DATASET_PATH):
    ruta_clase = os.path.join(DATASET_PATH, clase)
    if not os.path.isdir(ruta_clase):
        continue

    # contar archivos de video válidos
    videos = [f for f in os.listdir(ruta_clase)
              if f.lower().endswith((".mp4", ".mov", ".avi"))]

    cantidad = len(videos)
    if cantidad <= 15:
        clases_pocas.append((clase, cantidad))

print(f"\nClases con 15 o menos videos ({len(clases_pocas)} encontradas):\n")
for clase, cantidad in sorted(clases_pocas, key=lambda x: x[1]):
    print(f"  - {clase:30s} → {cantidad} videos")

print("\nRevisión completada.")
