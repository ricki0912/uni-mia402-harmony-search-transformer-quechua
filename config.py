"""
Configuraciones compartidas para HS y GA.
- Rutas de datasets, salidas y checkpoints.
- Se puede ajustar según el entorno/GPU.
"""

from pathlib import Path

# Directorios
OUT_DIR = Path("hs_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset
DATASET_QUECHUA_PATH = "data/data_preprocesada_sin_outliers_v4_biblia_part_0_1-1000.xlsx"

