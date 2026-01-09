"""
Configuraciones para Harmony Search:
- Rutas de datasets, salidas y checkpoints.
- Se puede ajustar segun el entorno/GPU.
- Permite overrides por variables de entorno (pensado para Colab).
"""

import json
import os
from pathlib import Path

from utils.search_space import HS_SEARCH_SPACE as DEFAULT_SEARCH_SPACE


def _env_path(var: str, default: str) -> Path:
    p = Path(os.getenv(var, default))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.getenv(var, default))
    except Exception:
        return default


def _env_float(var: str, default: float) -> float:
    try:
        return float(os.getenv(var, default))
    except Exception:
        return default


# Directorios (se pueden sobrescribir con vars de entorno)
OUT_DIR = _env_path("HS_OUT_DIR", "hs_runs")
CHECKPOINT_DIR = _env_path("HS_CHECKPOINT_DIR", "checkpoints")
LOG_DIR = _env_path("HS_LOG_DIR", "logs")

# Dataset
DATASET_QUECHUA_PATH = os.getenv("HS_DATASET_PATH", "data/data_preprocesada_sin_outliers_v4_biblia.xlsx")

# Idiomas (personalizable)
SOURCE_LANG = os.getenv("HS_SOURCE_LANG", "Quechua")
TARGET_LANG = os.getenv("HS_TARGET_LANG", "Spanish")
# Columnas correspondientes en el dataset
SOURCE_COLUMN = os.getenv("HS_SOURCE_COLUMN", "qu")
TARGET_COLUMN = os.getenv("HS_TARGET_COLUMN", "es")

# Tokenizacion SentencePiece
SPM_MODEL_TYPE = os.getenv("HS_SPM_MODEL_TYPE", "bpe")
SPM_VOCAB_SIZE = _env_int("HS_SPM_VOCAB_SIZE", 2000)
# SPM_PREFIX opcional; si se omite o viene vacío, se usa data/spm_<source>_<target>
def _env_default_str(var: str, default: str) -> str:
    val = os.getenv(var)
    return val if val not in (None, "", ".") else default

SPM_PREFIX = _env_default_str("HS_SPM_PREFIX", f"data/spm_{SOURCE_COLUMN}_{TARGET_COLUMN}")

# Parametros Harmony Search (sobrescribibles por env)
HS_PARAMS = {
    "HMS": _env_int("HS_HMS", 12),
    "HMCR": _env_float("HS_HMCR", 0.90),
    "PAR": _env_float("HS_PAR", 0.40),
    "BW": _env_float("HS_BW", 1.0),
    "NI": _env_int("HS_NI", 5),
    "seed": _env_int("HS_SEED", 123),
}
HS_STATE_PATH = Path(os.getenv("HS_STATE_PATH", OUT_DIR / "state_hs.json"))

# Espacio de busqueda (permite un JSON externo con la misma estructura)
SEARCH_SPACE = DEFAULT_SEARCH_SPACE
_search_json = os.getenv("HS_SEARCH_SPACE_JSON")
if _search_json and Path(_search_json).exists():
    try:
        SEARCH_SPACE = json.loads(Path(_search_json).read_text(encoding="utf-8"))
    except Exception:
        SEARCH_SPACE = DEFAULT_SEARCH_SPACE
