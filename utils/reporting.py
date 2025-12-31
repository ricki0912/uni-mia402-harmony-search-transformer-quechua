import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt


class TrialReport:
    """
    Utilidad para cargar un trial_xxx.json, exponer hiperparámetros/métricas
    y graficar las historias de entrenamiento.
    """

    def __init__(self, trial_path: Path):
        self.path = trial_path
        with open(trial_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.hp: Dict[str, Any] = self.data.get("hp", {})
        self.metrics: Dict[str, Any] = self.data.get("metrics", {})

    @property
    def loss_history(self) -> List[float]:
        return self.metrics.get("loss_history", [])

    @property
    def bleu_history(self) -> List[float]:
        return self.metrics.get("bleu_history", [])

    @property
    def rouge_histories(self) -> Dict[str, List[float]]:
        return self.metrics.get("rouge", {})

    def summary(self) -> Dict[str, Any]:
        fm = self.metrics.get("final_metrics", {})
        return {
            "path": str(self.path),
            "hp": self.hp,
            "final_loss": fm.get("loss"),
            "final_bleu": fm.get("bleu"),
            "final_rouge1": fm.get("rouge1"),
            "final_rouge2": fm.get("rouge2"),
            "final_rougeL": fm.get("rougeL"),
            "epochs_run": self.metrics.get("epochs_run"),
            "best_epoch_loss": self.metrics.get("stats", {}).get("best_epoch_loss"),
            "best_epoch_bleu": self.metrics.get("stats", {}).get("best_epoch_bleu"),
        }

    def plot_histories(self, show: bool = True, save_path: Optional[Path] = None):
        """Grafica loss/BLEU/ROUGE. Si save_path se define, guarda la imagen."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_loss, ax_bleu, ax_r1, ax_r2 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax_loss.plot(self.loss_history, label="loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()

        ax_bleu.plot(self.bleu_history, label="BLEU", color="orange")
        ax_bleu.set_title("BLEU")
        ax_bleu.legend()

        rouge = self.rouge_histories or {}
        ax_r1.plot(rouge.get("rouge1_history", []), label="ROUGE-1", color="green")
        ax_r1.plot(rouge.get("rougeL_history", []), label="ROUGE-L", color="red")
        ax_r1.set_title("ROUGE-1 / ROUGE-L")
        ax_r1.legend()

        ax_r2.plot(rouge.get("rouge2_history", []), label="ROUGE-2", color="blue")
        ax_r2.set_title("ROUGE-2")
        ax_r2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def load_all_trials(trials_dir: Path) -> List[TrialReport]:
    """Carga todos los trial_*.json de un directorio."""
    return [TrialReport(p) for p in sorted(trials_dir.glob("trial_*.json"))]
