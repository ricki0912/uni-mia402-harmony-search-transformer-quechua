import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.logger import Logger


class GeneticAlgorithm:
    """
    Algoritmo Genético simple para optimizar hiperparámetros.
    - Representación: dict de hiperparámetros conforme al search_space (mismo formato que HarmonySearch).
    - Selección: torneo.
    - Crossover: uniforme por clave.
    - Mutación: depende del tipo (choice/int_range/float_range).
    - Elitismo: conserva los mejores n individuos.
    """

    def __init__(
        self,
        search_space: Dict[str, Dict[str, Any]],
        fitness_fn: Callable[[Dict[str, Any]], float],
        population_size: int = 20,
        generations: int = 10,
        tournament_k: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        elitism: int = 1,
        seed: int = 42,
        resume_state_path: Optional[Path] = None,
    ):
        self.sp = search_space
        self.fitness_fn = fitness_fn
        self.pop_size = population_size
        self.generations = generations
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.rng = random.Random(seed)
        self.resume_state_path = Path(resume_state_path) if resume_state_path else None

        self.population: List[Dict[str, Any]] = []
        self.scores: List[float] = []
        self._start_gen = 0
        self._start_eval_idx = 0  # para reanudar dentro de una generación

    # -------- Helpers --------
    def _sample_param(self, name: str) -> Any:
        spec = self.sp[name]
        t = spec["type"]
        if t == "choice":
            return self.rng.choice(spec["values"])
        elif t == "int_range":
            return self.rng.randint(spec["low"], spec["high"])
        elif t == "float_range":
            return self.rng.uniform(spec["low"], spec["high"])
        else:
            raise ValueError(f"Tipo no soportado para {name}: {t}")

    def _random_individual(self) -> Dict[str, Any]:
        return {k: self._sample_param(k) for k in self.sp.keys()}

    def _mutate(self, ind: Dict[str, Any]) -> Dict[str, Any]:
        child = dict(ind)
        for k, spec in self.sp.items():
            Logger.print(f"[GA] Considerando mutación para parámetro '{k}'")
            if self.rng.random() > self.mutation_rate:
                continue
            t = spec["type"]
            if t == "choice":
                vals = spec["values"]
                options = [v for v in vals if v != child[k]]
                child[k] = self.rng.choice(options) if options else child[k]
            elif t == "int_range":
                bw = spec.get("bw", 1)
                delta = self.rng.choice([-bw, bw])
                child[k] = max(spec["low"], min(spec["high"], int(child[k] + delta)))
            elif t == "float_range":
                bw = spec.get("bw", (spec["high"] - spec["low"]) / 10)
                delta = (self.rng.random() * bw) * self.rng.choice([-1.0, 1.0])
                val = float(child[k] + delta)
                child[k] = max(spec["low"], min(spec["high"], val))
            Logger.print(f"[GA] Parámetro '{k}' mutado a {child[k]}")
        Logger.print(f"[GA] Individuo después de mutación: {child}")
        return child

    def _crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        child = {}
        for k in self.sp.keys():
            Logger.print(f"[GA] Cruzando parámetro '{k}'") 
            if self.rng.random() < 0.5:
                child[k] = a[k]
            else:
                child[k] = b[k]
        Logger.print(f"[GA] Individuo hijo generado: {child}")
        return child

    def _tournament_select(self) -> Dict[str, Any]:
        idxs = self.rng.sample(range(len(self.population)), k=min(self.tournament_k, len(self.population)))
        best = max(idxs, key=lambda i: self.scores[i])
        Logger.print(f"[GA] Individuo seleccionado: {self.population[best]} con fitness {self.scores[best]}")
        return self.population[best]

    def _evaluate_population(self, pop: List[Dict[str, Any]]) -> List[float]:
        Logger.print(f"[GA] Evaluando población de tamaño {len(pop)}")
        return [self.fitness_fn(ind) for ind in pop]

    def _save_state(self, gen: int):
        if not self.resume_state_path:
            return
        state = {
            "generation": gen,
            "population": self.population,
            "scores": self.scores,
            "eval_idx": len(self.scores),
            "params": {
                "population_size": self.pop_size,
                "generations": self.generations,
                "tournament_k": self.tournament_k,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "elitism": self.elitism,
            },
        }
        self.resume_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.resume_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_state(self):
        if not self.resume_state_path or not self.resume_state_path.exists():
            return False
        try:
            state = json.loads(self.resume_state_path.read_text(encoding="utf-8"))
            self.population = state.get("population", [])
            self.scores = state.get("scores", [])
            self._start_gen = int(state.get("generation", 0)) + 1
            self._start_eval_idx = int(state.get("eval_idx", 0))
            return True
        except Exception:
            return False

    # -------- Run --------
    def run(self) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
        # Intentar reanudar
        resumed = self._load_state()
        if not resumed or not self.population:
            self.population = [self._random_individual() for _ in range(self.pop_size)]
            self.scores = []
            self._start_gen = 0
            self._start_eval_idx = 0

        try:
            for gen in range(self._start_gen, self.generations):
                Logger.print(f"[GA][gen={gen}/{self.generations}] Inicio generación")

                # Si estamos reanudando la evaluación de esta generación, completar las faltantes
                if self.scores and len(self.scores) < len(self.population):
                    eval_start = len(self.scores)
                    for i in range(eval_start, len(self.population)):
                        self.scores.append(self.fitness_fn(self.population[i]))
                        self._save_state(gen)
                else:
                    # Si es una generación nueva, aplicar elitismo + reproducción
                    elite_indices = sorted(range(len(self.population)), key=lambda i: self.scores[i], reverse=True)[: self.elitism] if self.scores else []
                    new_pop = [self.population[i] for i in elite_indices]
                    while len(new_pop) < self.pop_size:
                        p1 = self._tournament_select()
                        p2 = self._tournament_select()
                        if self.rng.random() < self.crossover_rate:
                            child = self._crossover(p1, p2)
                        else:
                            child = dict(p1)
                        child = self._mutate(child)
                        new_pop.append(child)

                    self.population = new_pop
                    self.scores = []
                    for i in range(len(self.population)):
                        self.scores.append(self.fitness_fn(self.population[i]))
                        self._save_state(gen)

                best_idx = max(range(len(self.population)), key=lambda i: self.scores[i])
                mean_score = sum(self.scores) / len(self.scores) if self.scores else 0.0
                Logger.print(
                    f"[GA][gen={gen}/{self.generations}] Best BLEU={self.scores[best_idx]:.6f} hp={self.population[best_idx]}"
                )
                progress = (gen + 1) / self.generations * 100.0
                Logger.print(
                    f"[GA] Progreso total: {progress:.1f}% ({gen + 1}/{self.generations}) | BLEU(mean)={mean_score:.6f}"
                )
        except KeyboardInterrupt:
            # Guardar estado inmediato al interrumpir
            current_gen = gen if 'gen' in locals() else self._start_gen
            self._save_state(current_gen)
            Logger.print("[GA] Interrupción manual: estado guardado para reanudar.", level="WARNING")
            raise

        best_idx = max(range(len(self.population)), key=lambda i: self.scores[i])
        best_ind = self.population[best_idx]
        best_score = self.scores[best_idx]
        history = list(zip(self.population, self.scores))
        return best_ind, best_score, history
