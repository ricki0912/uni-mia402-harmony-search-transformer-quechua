import json
import time
from pathlib import Path

from utils.genetic_algorithm import GeneticAlgorithm
from utils.logger import Logger
from utils.search_space import GA_SEARCH_SPACE
from model.training import build_training_artifacts, train_single_run
import config

# Directorio de salidas GA
OUT_DIR = config.OUT_DIR

# Dataset y artefactos (reutilizables)
DATASET_QUECHUA_PATH = config.DATASET_QUECHUA_PATH
ARTIFACTS = build_training_artifacts(max_sequence_length=200, dataset_path=DATASET_QUECHUA_PATH)

# Espacio de búsqueda acotado para evitar OOM
SEARCH_SPACE = GA_SEARCH_SPACE


def fitness_fn(hp: dict) -> float:
    Logger.print("=====================================================")

    """
    Ejecuta una corrida de entrenamiento con los hiperparámetros hp y devuelve BLEU.
    Maneja OOM penalizando el fitness y registrando el trial igualmente.
    """
    start_ts = time.time()
    trial_id = int(start_ts * 1000)
    status = "ok"
    Logger.print(f"[GA][trial={trial_id}] Iniciando evaluación HP: {hp}")

    try:
        metrics = train_single_run(hp, ARTIFACTS)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
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

    bleu = float(metrics.get("bleu", 0.0))

    # Guardar trial
    end_ts = time.time()
    duration_s = end_ts - start_ts
    record = {
        "algo": "GA",
        "trial_id": trial_id,
        "status": status,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_seconds": duration_s,
        "hp": hp,
        "metrics": metrics,
    }
    with open(OUT_DIR / f"trial_ga_{trial_id}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    Logger.print(f"[GA][trial={trial_id}] Finalizado BLEU={bleu:.6f}")

    return bleu


def main():
    ga = GeneticAlgorithm(
        search_space=SEARCH_SPACE,
        fitness_fn=fitness_fn,
        population_size=24,
        generations=10,
        tournament_k=3,
        crossover_rate=0.9,
        mutation_rate=0.15,
        elitism=1,
        seed=123,
        resume_state_path=OUT_DIR / "state_ga.json",
    )

    best_hp, best_bleu, history = ga.run()

    result = {
        "algo": "GA",
        "best_hp": best_hp,
        "best_bleu": best_bleu,
        "history": [{"hp": hp, "bleu": score} for (hp, score) in history],
    }
    with open(OUT_DIR / "ga_best.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n================ RESULTADO GA ================")
    print(json.dumps(result["best_hp"], indent=2, ensure_ascii=False))
    print(f"Mejor BLEU: {best_bleu:.4f}")


if __name__ == "__main__":
    main()
