import os
import re
import pandas as pd
from collections import defaultdict

# Ruta base
DATASET_PATH = "data/dataset"

# Función para normalizar nombres de clases
def normalizar_nombre(nombre: str) -> str:
    nombre = nombre.upper()                 # todo en mayúsculas
    nombre = nombre.replace("_", " ")       # guiones bajos → espacios
    nombre = re.sub(r"\s+", " ", nombre)    # espacios múltiples → uno solo
    nombre = nombre.strip()                 # quitar espacios laterales
    nombre = re.sub(r"\s*\d+$", "", nombre) # quitar número final
    return nombre.strip()

# Obtener carpetas
clases = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

conteo = {}
duplicados = defaultdict(list)

for clase in clases:
    ruta = os.path.join(DATASET_PATH, clase)
    videos = [f for f in os.listdir(ruta) if f.lower().endswith((".mp4", ".mov", ".avi"))]
    conteo[clase] = len(videos)

    # Clave normalizada para detectar duplicados
    clave = normalizar_nombre(clase)
    duplicados[clave].append(clase)

# Detectar clases vacías o con un solo video
vacias = [c for c, n in conteo.items() if n == 0]
una_sola = [c for c, n in conteo.items() if n == 1]

# Detectar duplicados por nombre normalizado
duplicados_reales = [v for v in duplicados.values() if len(v) > 1]

# Mostrar resultados
print("TOTAL DE CLASES:", len(conteo))
print("TOTAL DE VIDEOS:", sum(conteo.values()))
print()

if vacias:
    print(f"Clases vacías ({len(vacias)}):")
    for c in vacias:
        print(f"   - {c}")
    print()

if una_sola:
    print(f"Clases con un solo video ({len(una_sola)}):")
    for c in una_sola:
        print(f"   - {c}")
    print()

if duplicados_reales:
    print(f"Posibles clases duplicadas ({len(duplicados_reales)} grupos):")
    for grupo in duplicados_reales:
        print(f"   - {' | '.join(grupo)}")
    print()

# Guardar CSV
df = pd.DataFrame(list(conteo.items()), columns=["Clase", "Cantidad_videos"])
df = df.sort_values(by="Cantidad_videos", ascending=True)
csv_path = os.path.join(DATASET_PATH, "reporte_clases.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"Reporte guardado en: {csv_path}")
