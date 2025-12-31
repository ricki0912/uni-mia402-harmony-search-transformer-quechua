"""
harmony_search_adjusted.py

Ajuste de Harmony Search (HS) alineado con el Algoritmo 1 del paper:
- HM inicial aleatoria (HMS armonías).
- Por cada iteración (t = 1..NI) se improvisa 1 nueva armonía x_new (forma canónica),
  y se actualiza HM reemplazando la peor si x_new es mejor.
- HMC: con prob HMCR, cada componente se toma de HM; caso contrario, se inicializa al azar.
- PA: si el componente vino de HM y random < PAR, se ajusta con ± random * BW (paper).
- Control de límites (l,u) para rangos numéricos.

Soporta optimización en modo 'max' (p.ej., BLEU) o 'min' (p.ej., pérdida).
Incluye reanudación (resume) y registro básico.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Harmony:
    hp: Dict[str, Any]
    fitness: float


class HarmonySearch:
    def __init__(
        self,
        search_space: Dict[str, Dict[str, Any]],
        fitness_fn: Callable[[Dict[str, Any]], float],
        HMS: int = 20,
        HMCR: float = 0.95,
        PAR: float = 0.30,
        NI: int = 100,
        seed: int = 42,
        objective: str = "max",  # "max" (BLEU) o "min" (loss)
        resume_state_path: Optional[Path] = None,
    ):
        self.sp = search_space
        self.fitness_fn = fitness_fn

        self.HMS = int(HMS)
        self.HMCR = float(HMCR)
        self.PAR = float(PAR)
        self.NI = int(NI)
        if objective not in ("max", "min"):
            raise ValueError("objective debe ser 'max' o 'min'")
        self.objective = objective

        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.resume_state_path = Path(resume_state_path) if resume_state_path else None

        self.HM: List[Harmony] = []
        self._start_iter = 0

    # ---------- sampling ----------
    def _sample_param(self, name: str) -> Any:
        spec = self.sp[name]
        t = spec["type"]
        if t == "choice":
            return self.rng.choice(spec["values"])
        if t == "int_range":
            return self.rng.randint(int(spec["low"]), int(spec["high"]))
        if t == "float_range":
            return self.rng.uniform(float(spec["low"]), float(spec["high"]))
        raise ValueError(f"Tipo no soportado para {name}: {t}")

    def _random_harmony(self) -> Dict[str, Any]:
        return {k: self._sample_param(k) for k in self.sp.keys()}

    # ---------- HM operations ----------
    def _is_better(self, a: float, b: float) -> bool:
        """True si a es mejor que b según objective."""
        return a > b if self.objective == "max" else a < b

    def _sort_hm(self) -> None:
        reverse = True if self.objective == "max" else False
        self.HM.sort(key=lambda h: h.fitness, reverse=reverse)

    def _worst_index(self) -> int:
        if not self.HM:
            return -1
        return len(self.HM) - 1  # tras ordenar, el peor queda al final

    # ---------- HMC + PA (paper-like) ----------
    def _pick_from_memory(self, name: str) -> Any:
        # x_new_j = x_kj, k aleatorio en HM
        k = self.rng.randrange(len(self.HM))
        return self.HM[k].hp[name]

    def _pitch_adjust(self, name: str, value: Any) -> Any:
        """
        Paper: x_new_j = x_new_j ± random * BW
        Aquí BW es por parámetro (spec['bw']). Para 'choice', interpretamos BW como
        cantidad máxima de pasos a moverse en la lista (>=1).
        """
        spec = self.sp[name]
        bw = float(spec.get("bw", 0))
        if bw == 0:
            return value

        t = spec["type"]
        sign = -1.0 if self.rng.random() < 0.5 else 1.0

        if t == "float_range":
            delta = sign * self.rng.random() * bw
            new_val = float(value) + delta
            return float(max(float(spec["low"]), min(float(spec["high"]), new_val)))

        if t == "int_range":
            # usar random*bw y redondear
            delta = sign * self.rng.random() * bw
            new_val = int(round(float(value) + delta))
            return int(max(int(spec["low"]), min(int(spec["high"]), new_val)))

        if t == "choice":
            vals = list(spec["values"])
            if len(vals) <= 1:
                return value
            idx = vals.index(value)
            max_step = max(1, int(round(bw)))  # bw=200 -> max_step=200, pero se recorta a tamaño lista
            step = self.rng.randint(1, max_step)
            new_idx = idx + (step if sign > 0 else -step)
            new_idx = max(0, min(len(vals) - 1, new_idx))
            return vals[new_idx]

        return value

    def _improvise_new_harmony(self) -> Dict[str, Any]:
        """
        Implementa el núcleo del Algoritmo 1:
        para cada parámetro j:
          si random < HMCR: tomar de HM (HMC) y, si random < PAR, aplicar PA
          si no: inicialización aleatoria en [l,u] o en lista
        """
        x_new: Dict[str, Any] = {}
        for name in self.sp.keys():
            if self.HM and (self.rng.random() < self.HMCR):
                v = self._pick_from_memory(name)
                if self.rng.random() < self.PAR:
                    v = self._pitch_adjust(name, v)
                x_new[name] = v
            else:
                x_new[name] = self._sample_param(name)
        return x_new

    def _evaluate(self, x: Dict[str, Any]) -> float:
        return float(self.fitness_fn(x))

    def _hm_update(self, x_new: Dict[str, Any], fit_new: float) -> None:
        """
        Paper: si f(x_new) mejor que f(x_worst), reemplaza peor.
        Aquí primero ordenamos, luego comparamos con el peor.
        """
        if len(self.HM) < self.HMS:
            self.HM.append(Harmony(x_new, fit_new))
            self._sort_hm()
            return

        self._sort_hm()
        worst_idx = self._worst_index()
        worst_fit = self.HM[worst_idx].fitness
        if self._is_better(fit_new, worst_fit):
            self.HM[worst_idx] = Harmony(x_new, fit_new)
            self._sort_hm()

    # ---------- resume / save ----------
    def _save_state(self, iteration: int) -> None:
        if not self.resume_state_path:
            return
        state = {
            "iteration": iteration,
            "HMS": self.HMS,
            "HMCR": self.HMCR,
            "PAR": self.PAR,
            "NI": self.NI,
            "objective": self.objective,
            "hm": [{"hp": h.hp, "fitness": h.fitness} for h in self.HM],
        }
        self.resume_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.resume_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _try_resume(self) -> None:
        if not (self.resume_state_path and self.resume_state_path.exists()):
            return
        try:
            state = json.loads(self.resume_state_path.read_text(encoding="utf-8"))
            self._start_iter = int(state.get("iteration", 0))
            self.objective = state.get("objective", self.objective)
            hm_entries = state.get("hm", [])
            self.HM = [Harmony(e["hp"], float(e["fitness"])) for e in hm_entries]
            self._sort_hm()
        except Exception:
            self._start_iter = 0
            self.HM = []

    def run(self) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
        """
        Ejecuta HS y devuelve (best_hp, best_fitness, hm_list_sorted).
        """
        self._try_resume()

        # 1) Inicializar HM si está vacía
        if not self.HM:
            for _ in range(self.HMS):
                hp = self._random_harmony()
                fit = self._evaluate(hp)
                self.HM.append(Harmony(hp, fit))
            self._sort_hm()
            self._save_state(0)

        # 2) Improvisar 1 armonía por iteración (NI iteraciones)
        for t in range(self._start_iter, self.NI):
            x_new = self._improvise_new_harmony()
            fit_new = self._evaluate(x_new)
            self._hm_update(x_new, fit_new)
            self._save_state(t + 1)

        self._sort_hm()
        best = self.HM[0]
        return best.hp, best.fitness, [(h.hp, h.fitness) for h in self.HM]
