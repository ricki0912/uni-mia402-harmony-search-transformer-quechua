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

# Idiomas (personalizable)
SOURCE_LANG = "Quechua"
TARGET_LANG = "Spanish"
# Columnas correspondientes en el dataset
SOURCE_COLUMN = "qu"
TARGET_COLUMN = "es"

# Tokenizacion SentencePiece
SPM_MODEL_TYPE = "bpe"
SPM_VOCAB_SIZE = 2000
# SPM_PREFIX opcional; si se omite, se usa data/spm_<source>_<target>
