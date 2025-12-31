import json
import time
from pathlib import Path

from utils.harmony_search import HarmonySearch
from utils.logger import Logger
from utils.search_space import HS_SEARCH_SPACE

from model.training import build_training_artifacts, train_single_run
import config

OUT_DIR = config.OUT_DIR
DATASET_QUECHUA_PATH=config.DATASET_QUECHUA_PATH

ARTIFACTS = build_training_artifacts(max_sequence_length=200, dataset_path=DATASET_QUECHUA_PATH)
SEARCH_SPACE = HS_SEARCH_SPACE

def fitness_fn(hp: dict) -> float:
    Logger.print("=====================================================")

    """
    Ejecuta una corrida de entrenamiento con los hiperparámetros hp y devuelve BLEU.
    El BLEU que devolvemos es el que reporta train_single_run en la última época.
    """

    start_ts = time.time()
    trial_id = int(start_ts * 1000)
    status = "ok"
    Logger.print(f"[HS][trial={trial_id}] Iniciando evaluación HP: {hp}")

    # Llama tu entrenamiento real con manejo de OOM para no interrumpir la búsqueda
    try:
        ARTIFACTS = build_training_artifacts(max_sequence_length=hp["max_sequence_length"], dataset_path=DATASET_QUECHUA_PATH)
        metrics = train_single_run(hp, ARTIFACTS)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            Logger.print(f"OOM con hp={hp}, penalizando fitness y continuando.", level="ERROR")
            try:
                import torch, gc  # type: ignore
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
            status = "oom"
            metrics = {
                "bleu": 0.0,
                "final_metrics": {"bleu": 0.0, "loss": float("inf")},
                "oom": True,
                "error": str(e),
            }
        else:
            Logger.print(f"Error en trial con hp={hp}: {e}", level="ERROR")
            raise

    # La métrica objetivo será BLEU (maximización)
    bleu = float(metrics.get("bleu", 0.0))

    # Guarda un registro ligero por armonía:
    end_ts = time.time()
    duration_s = end_ts - start_ts
    record = {
        "algo": "HS",
        "trial_id": trial_id,
        "status": status,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_seconds": duration_s,
        "hp": hp,
        "metrics": metrics,
    }
    with open(OUT_DIR / f"trial_{trial_id}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    Logger.print(f"[HS][trial={trial_id}] Finalizado BLEU={bleu:.6f}")

    return bleu

# ---------- Correr Harmony Search ----------
def main():
    hs = HarmonySearch(
        search_space=SEARCH_SPACE,
        fitness_fn=fitness_fn,
        HMS=12,         # tamaño de memoria
        HMCR=0.90,      # prob. de tomar valores desde la memoria
        PAR=0.40,       # prob. de ajustar el pitch (vecindad)
        BW=1.0,         # amplitud del ajuste (ya internalizada por _pitch_adjust)
        NI=5,           # iteraciones de mejora sobre la HM
        seed=123,
        resume_state_path=OUT_DIR / "state_hs.json"
    )

    best_hp, best_bleu, history = hs.run()

    # Persistimos resultados finales
    result = {
        "best_hp": best_hp,
        "best_bleu": best_bleu,
        "history": [{"hp": hp, "bleu": bleu} for (hp, bleu) in history],
    }
    with open(OUT_DIR / "hs_best.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n================ RESULTADO HS ================")
    print(json.dumps(result["best_hp"], indent=2, ensure_ascii=False))
    print(f"Mejor BLEU: {best_bleu:.4f}")

if __name__ == "__main__":
    main()
