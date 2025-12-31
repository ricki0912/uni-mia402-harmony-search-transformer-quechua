import json
import sys
import time
from pathlib import Path
from typing import Optional

# Asegura que la raíz del repo esté en sys.path para importar config y model/*
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from model.training import build_training_artifacts, train_single_run
from utils.logger import Logger

# Hiperparámetros por defecto (se usan tal cual al ejecutar el script)
"""{
    "d_model": 256,
    "batch_size": 8,
    "ffn_hidden": 512,
    "num_heads": 4,
    "drop_prob": 0.1,
    "num_layers": 2,
    "max_sequence_length": 200,
    "epochs": 2,
    "lr": 1e-3,
}"""

DEFAULT_HP = {
  "d_model": 512,
  "ffn_hidden": 512,
  "num_heads": 8,
  "drop_prob": 0.07753112550551966,
  "num_layers": 3,
  "lr": 0.0001,
  "batch_size": 2,
  "max_sequence_length": 250,
  "epochs": 50
}

"""{
  "d_model": 1024,
  "ffn_hidden": 1024,
  "num_heads": 4,
  "drop_prob": 0.0021526250773962613,
  "num_layers": 3,
  "lr": 0.0001,
  "batch_size": 8,
  "max_sequence_length": 200,
  "epochs": 50
}"""

# Si quieres modificar hiperparámetros sin tocar DEFAULT_HP, edita aquí:
USER_HP_OVERRIDE: Optional[dict] = None

# Rutas de dataset y checkpoints por defecto (puedes editarlas abajo si deseas)
DATASET_PATH = Path(config.DATASET_QUECHUA_PATH)
#DATASET_PATH = Path("data/data_preprocesada_sin_outliers_v4_biblia.xlsx")
CHECKPOINT_DIR = Path(config.CHECKPOINT_DIR)
CHECKPOINT_EVERY = 5  # épocas
RESUME_FROM: Optional[str] = None  # Ruta a .pt para reanudar, o None


def load_hyperparameters() -> dict:
    """
    Devuelve los hiperparámetros a usar (override si está definido).
    """
    if USER_HP_OVERRIDE:
        hp = {**DEFAULT_HP, **USER_HP_OVERRIDE}
        Logger.print(f"Usando USER_HP_OVERRIDE sobre DEFAULT_HP: {hp}")
    else:
        hp = DEFAULT_HP.copy()
        Logger.print(f"Usando DEFAULT_HP: {hp}")
    return hp


def run_training_once(
    hp: dict,
    dataset_path: Path,
    checkpoint_dir: Path,
    checkpoint_every: int,
    resume_from: Optional[str],
):
    """
    Ejecuta un entrenamiento con hp y guarda métricas en OUT_DIR.
    """
    Logger.print("=====================================================")
    Logger.print(f"[ManualTrain] Iniciando entrenamiento con HP: {hp}")

    start_ts = time.time()
    trial_id = int(start_ts * 1000)
    status = "ok"

    artifacts = build_training_artifacts(
        max_sequence_length=hp["max_sequence_length"],
        dataset_path=str(dataset_path),
    )

    try:
        metrics = train_single_run(
            hp,
            artifacts,
            resume_from=resume_from,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every=checkpoint_every,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            status = "oom"
            Logger.print(f"OOM con hp={hp}, penalizando métricas y continuando.", level="ERROR")
            try:
                import torch, gc  # type: ignore

                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
            metrics = {
                "bleu": 0.0,
                "final_metrics": {"bleu": 0.0, "loss": float("inf")},
                "oom": True,
                "error": str(e),
            }
        else:
            Logger.print(f"Error en entrenamiento manual: {e}", level="ERROR")
            raise

    bleu = float(metrics.get("bleu", 0.0))
    end_ts = time.time()
    duration_s = end_ts - start_ts

    out_dir = config.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "algo": "manual_train",
        "trial_id": trial_id,
        "status": status,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_seconds": duration_s,
        "hp": hp,
        "dataset": str(dataset_path),
        "resume_from": resume_from,
        "checkpoint_dir": str(checkpoint_dir),
        "metrics": metrics,
    }
    out_path = out_dir / f"manual_train_{trial_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    Logger.print(f"[ManualTrain][trial={trial_id}] Finalizado BLEU={bleu:.6f}")
    Logger.print(f"Métricas guardadas en {out_path}")

    return record


def main():
    hp = load_hyperparameters()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run_training_once(
        hp=hp,
        dataset_path=DATASET_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_every=CHECKPOINT_EVERY,
        resume_from=RESUME_FROM,
    )


if __name__ == "__main__":
    main()
