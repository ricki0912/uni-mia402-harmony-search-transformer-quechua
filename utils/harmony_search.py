# harmony_search.py
# Implementacion de Harmony Search (HS) para optimizar hiperparametros del Transformer
# Basado en el esquema del paper (HMS, HMCR, PAR, BW, NI) y usando BLEU como fitness.

import random
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Callable, Tuple, Any, Optional
from utils.logger import Logger


class HarmonySearch:
    """
    Harmony Search para optimizacion de hiperparametros.
    - Cada 'armonia' es un dict de hiperparametros.
    - HS opera con:
        HMS: tamano de memoria
        HMCR: tasa de consideracion de memoria
        PAR: tasa de ajuste de 'pitch' (variacion local)
        BW: amplitud del ajuste (para continuos o desplazamiento discreto)
        NI: numero de iteraciones (ciclos de mejora)
    - search_space: dict de especificaciones por parametro:
        {
          "d_model": {"type":"choice", "values":[64,128,256,512,1024], "bw":200},
          "heads":   {"type":"int_range", "low":2, "high":4, "bw":1},
          "ffn_hidden":{"type":"choice","values":[64,128,256,512,1024], "bw":200},
          "batch_size":{"type":"choice","values":[2,4,8,16,32,64,128], "bw":30},
          "layers":  {"type":"int_range","low":2,"high":6,"bw":1},
          "lr":      {"type":"choice","values":[1e-4,1e-3,1e-2],"bw":0}, # sin ajuste
          "dropout": {"type":"float_range","low":0.0,"high":0.3,"bw":0.05},
          "epochs":  {"type":"choice","values":[50,100,150,200,300,400,500],"bw":50},
        }
    """

    def __init__(
        self,
        search_space: Dict[str, Dict[str, Any]],
        fitness_fn: Callable[[Dict[str, Any]], float],
        HMS: int = 20,
        HMCR: float = 0.95,
        PAR: float = 0.30,
        BW: float = 1.0,
        NI: int = 5,
        seed: int = 42,
        resume_state_path: Optional[Path] = None,
        verbose: bool = True,
        log_path: Optional[Path] = None,
        trace_path: Optional[Path] = None,
    ):
        self.sp = search_space
        self.fitness_fn = fitness_fn
        self.HMS = int(HMS)
        self.HMCR = float(HMCR)
        self.PAR = float(PAR)
        self.BW = float(BW)
        self.NI = int(NI)
        self.objective = "max"  # modo paper base (BLEU); extensible a "min" si se requiere
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.resume_state_path = Path(resume_state_path) if resume_state_path else None
        self.verbose = bool(verbose)
        self.log_path = Path(log_path) if log_path else None
        log_dir = self.log_path.parent if self.log_path else None
        self.logger = Logger(log_dir)

        # Trace en forma de arbol para cada ejecucion
        base = Path(__file__).resolve().parent.parent
        if trace_path:
            self.trace_path = Path(trace_path)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.trace_path = base / "hs_runs" / f"trace_{ts}.json"
        self.trace: Dict[str, Any] = {
            "run_meta": {
                "algo": "HS",
                "seed": seed,
                "HMS": self.HMS,
                "HMCR": self.HMCR,
                "PAR": self.PAR,
                "BW": self.BW,
                "NI": self.NI,
                "objetivo": self.objective,
                "timestamp": datetime.now().isoformat(),
            },
            "init_hm": [],
            "iterations": [],
            "final": {},
        }

        self.HM: List[Tuple[Dict[str, Any], float]] = []  # lista de (armonia, fitness)
        self._start_iter = 0  # para reanudar
        self._start_eval_idx = 0  # armonias ya evaluadas en la iteracion actual
        self.eval_count = 0

    def _log(self, msg: str, indent: int = 0) -> None:
        prefix = "    " * max(indent, 0)
        txt = f"{prefix}{msg}"
        if self.verbose:
            Logger.print(txt)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] INFO: {txt}"
            with self.logger.log_file.open("a", encoding="utf-8") as f:
                f.write(log_message + "\n")

    # -------- Helpers de muestreo y pitch-adjustment --------
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

    def _evaluate(self, x: Dict[str, Any]) -> float:
        self.eval_count += 1
        return self.fitness_fn(x)

    def _pitch_adjust(self, name: str, value: Any, return_meta: bool = False, indent: int = 0) -> Any:
        """
        Ajuste de pitch siguiendo el esquema HS del paper: prioriza BW por parametro;
        si no se define usa el BW global y, si BW=0, omite el ajuste.
        """
        spec = self.sp[name]
        # BW local (si existe) tiene prioridad; si no, se usa BW global. BW==0 desactiva ajuste (paper).
        if "bw" in spec:
            bw = float(spec["bw"])
        else:
            bw = float(self.BW)
        if bw == 0:
            return (value, bw, None, False) if return_meta else value  # sin ajuste

        t = spec["type"]
        sign = -1.0 if self.rng.random() < 0.5 else 1.0
        clamped = False

        if t == "choice":
            vals = spec["values"]
            if len(vals) == 1:
                return (value, bw, sign, clamped) if return_meta else value
            idx = vals.index(value)
            max_step = max(1, int(round(bw)))
            step = self.rng.randint(1, max_step)
            new_idx = idx + (step if sign > 0 else -step)
            raw_idx = new_idx
            new_idx = max(0, min(len(vals) - 1, new_idx))
            clamped = new_idx != raw_idx
            if clamped:
                self._log(f"[Clamp] j={name} acotado a [0,{len(vals) - 1}] => {vals[new_idx]}", indent=indent)
            result = vals[new_idx]
        elif t == "int_range":
            delta = sign * self.rng.random() * bw
            new_val = int(round(value + delta))
            clamped_val = int(max(spec["low"], min(spec["high"], new_val)))
            clamped = clamped_val != new_val
            if clamped:
                self._log(f"[Clamp] j={name} acotado a [{spec['low']},{spec['high']}] => {clamped_val}", indent=indent)
            result = clamped_val
        elif t == "float_range":
            delta = sign * self.rng.random() * bw
            new_val = float(value + delta)
            clamped_val = float(max(spec["low"], min(spec["high"], new_val)))
            clamped = clamped_val != new_val
            if clamped:
                self._log(f"[Clamp] j={name} acotado a [{spec['low']},{spec['high']}] => {clamped_val}", indent=indent)
            result = clamped_val
        else:
            result = value

        if return_meta:
            return result, bw, sign, clamped
        return result

    def _random_harmony(self) -> Dict[str, Any]:
        return {k: self._sample_param(k) for k in self.sp.keys()}

    def _consider_memory(self, j: str) -> Tuple[Any, int]:
        # Elegir valor existente de HM
        if not self.HM:
            return self._sample_param(j), -1
        k = self.rng.randrange(len(self.HM))
        cand_val = self.HM[k][0][j]
        return cand_val, k

    def _new_harmony(self, record: bool = False, indent: int = 0):
        x_new = {}
        j_logs: Dict[str, Any] = {}
        for j in self.sp.keys():
            r_hmcr = self.rng.random()
            use_memory = bool(self.HM) and r_hmcr < self.HMCR
            if use_memory:
                v_from_hm, k = self._consider_memory(j)
                self._log(f"[HMC] j={j} r={r_hmcr:.4f} < HMCR => desde HM: valor={v_from_hm} (k={k})", indent=indent)
                v = v_from_hm
                r_par = self.rng.random()
                if r_par < self.PAR:
                    adjusted, bw_used, sign, clamped = self._pitch_adjust(j, v, return_meta=True, indent=indent)
                    sign_str = "n/a" if sign is None else ("+" if sign > 0 else "-")
                    self._log(f"[PA] j={j} r={r_par:.4f} < PAR => ajuste de pitch: antes={v} despues={adjusted} usando BW={bw_used} y signo={sign_str}", indent=indent)
                    v = adjusted
                else:
                    bw_used = None
                    sign_str = None
                    clamped = False
                x_new[j] = v
                if record:
                    j_logs[j] = {
                        "source": "HM",
                        "r_hmcr": r_hmcr,
                        "r_par": r_par,
                        "pitch_before": v_from_hm,
                        "pitch_after": v,
                        "bw_usado": bw_used,
                        "signo": sign_str,
                        "clamp": clamped,
                        "k": k,
                    }
            else:
                v_rand = self._sample_param(j)
                if self.HM:
                    rand_reason = f"r={r_hmcr:.4f} >= HMCR"
                else:
                    rand_reason = f"HM vacia (r={r_hmcr:.4f})"
                self._log(f"[Rand] j={j} {rand_reason} => muestreo aleatorio en dominio: valor={v_rand}", indent=indent)
                x_new[j] = v_rand
                if record:
                    j_logs[j] = {
                        "source": "rand",
                        "r_hmcr": r_hmcr,
                        "valor": v_rand,
                        "motivo": rand_reason,
                    }
        if record:
            return x_new, j_logs
        return x_new

    def _evaluate_and_update(self, x: Dict[str, Any]) -> float:
        fitness = self._evaluate(x)   # Mayor BLEU = mejor
        self.HM.append((x, fitness))
        # Mantener HM ordenada por fitness descendente y recortar a HMS
        self.HM.sort(key=lambda t: t[1], reverse=True)
        if len(self.HM) > self.HMS:
            self.HM = self.HM[:self.HMS]
        return fitness

    def _save_state(self, iteration: int):
        if not self.resume_state_path:
            return
        state = {
            "iteration": iteration,
            "HMS": self.HMS,
            "HMCR": self.HMCR,
            "PAR": self.PAR,
            "BW": self.BW,
            "NI": self.NI,
            "hm": [{"hp": hp, "fitness": fit} for hp, fit in self.HM],
            "eval_idx": self._start_eval_idx,
        }
        self.resume_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.resume_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_trace(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text(json.dumps(self.trace, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
        self.eval_count = 0
        self._log(f"HS inicio HMS={self.HMS} HMCR={self.HMCR} PAR={self.PAR} BW={self.BW} NI={self.NI} objetivo={self.objective} eval_count={self.eval_count}")

        # Intentar reanudar desde estado guardado
        if self.resume_state_path and self.resume_state_path.exists():
            try:
                state = json.loads(self.resume_state_path.read_text(encoding="utf-8"))
                self.HM = [(entry["hp"], entry["fitness"]) for entry in state.get("hm", [])]
                self._start_iter = int(state.get("iteration", 0))
                self._start_eval_idx = int(state.get("eval_idx", 0))
                self._log(f"[Resume] reanudando en iteracion={self._start_iter} eval_idx={self._start_eval_idx} tam_HM={len(self.HM)}")
            except Exception:
                # Si falla la carga, inicializa normalmente
                self.HM = []
                self._start_iter = 0
                self._start_eval_idx = 0
        else:
            self._start_iter = 0
            self._start_eval_idx = 0

        if self.HM:
            self.HM.sort(key=lambda t: t[1], reverse=True)
            if len(self.HM) > self.HMS:
                self.HM = self.HM[:self.HMS]

        # Inicializacion si no hay HM previa
        if not self.HM:
            self._log("[InitHM] HM vacia. Inicializando de forma aleatoria")
            for i in range(self.HMS):
                x_i = self._random_harmony()
                fit_i = self._evaluate_and_update(x_i)
                self.trace["init_hm"].append({"hp": x_i, "fitness": fit_i})
                self._log(f"[InitHM] i={i + 1}/{self.HMS} x_i={x_i} f(x_i)={fit_i}", indent=1)
            best_init = self.HM[0][1]
            worst_init = self.HM[-1][1]
            self._log(f"[InitHM] HM creada. Mejor f(x_mejor)={best_init} Peor f(x_peor)={worst_init}")

        # Alias para la nomenclatura solicitada sin modificar otros metodos
        if not hasattr(self, "_improvise_new_harmony"):
            def _improvise_new_harmony_wrapper(record: bool = False, indent: int = 0):
                return self._new_harmony(record=record, indent=indent)
            self._improvise_new_harmony = _improvise_new_harmony_wrapper  # type: ignore[attr-defined]
        if not hasattr(self, "_hm_update"):
            def _hm_update_local(x_new: Dict[str, Any], fit_new: float) -> None:
                self.HM.append((x_new, fit_new))
                self.HM.sort(key=lambda t: t[1], reverse=True)
                if len(self.HM) > self.HMS:
                    self.HM = self.HM[:self.HMS]
            self._hm_update = _hm_update_local  # type: ignore[attr-defined]

        for t in range(self._start_iter, self.NI):
            iter_idx = t + 1
            best_f = self.HM[0][1]
            worst_f = self.HM[-1][1]
            self._log(f"=== Iteracion t={iter_idx}/{self.NI} === (eval_count={self.eval_count})")
            self._log(f"[IterState] antes de improvisar: mejor f(x_mejor)={best_f} peor f(x_peor)={worst_f}", indent=1)
            iter_trace: Dict[str, Any] = {
                "iter_idx": iter_idx,
                "before_best": best_f,
                "before_worst": worst_f,
                "improvisations": [],
            }

            for i in range(self.HMS):
                harmony_idx = i + 1
                self._log(f"[Improvise] t={iter_idx} i={harmony_idx}/{self.HMS} construyendo x_new", indent=1)
                x_new, j_log = self._improvise_new_harmony(record=True, indent=2)
                fit_new = self._evaluate(x_new)
                self._log(f"[Eval] f(x_new)={fit_new} x_new={x_new}", indent=1)
                f_worst_before = self.HM[-1][1]
                replace = fit_new > f_worst_before
                decision = "SI" if replace else "NO"
                self._log(f"[Compare] f(x_new)={fit_new} vs f(x_peor)={f_worst_before} => reemplazar? {decision}", indent=1)
                self._hm_update(x_new, fit_new)
                f_worst_after = self.HM[-1][1]
                f_best_after = self.HM[0][1]
                if replace:
                    self._log(f"[UpdateHM] Reemplazado x_peor. Nuevo peor={f_worst_after}, nuevo mejor={f_best_after}", indent=1)
                else:
                    self._log("[UpdateHM] Sin cambios", indent=1)

                improv_trace = {
                    "harmony_idx": harmony_idx,
                    "j_decisions": j_log,
                    "x_new": x_new,
                    "fitness": fit_new,
                    "replace": replace,
                    "f_worst_before": f_worst_before,
                    "f_worst_after": f_worst_after,
                    "f_best_after": f_best_after,
                }
                iter_trace["improvisations"].append(improv_trace)

            iter_trace["after_best"] = self.HM[0][1]
            iter_trace["after_worst"] = self.HM[-1][1]
            iter_trace["eval_count"] = self.eval_count
            self.trace["iterations"].append(iter_trace)

            self._save_state(t + 1)
            if self.resume_state_path:
                self._log(f"[SaveState] iteracion={iter_idx}", indent=1)
            self._log(f"[IterEnd] t={iter_idx} mejor f(x_mejor)={self.HM[0][1]} peor f(x_peor)={self.HM[-1][1]} eval_count={self.eval_count}", indent=1)

        best_x, best_f = self.HM[0]
        self.trace["final"] = {
            "best_hp": best_x,
            "best_fitness": best_f,
            "eval_count_total": self.eval_count,
        }
        self._write_trace()
        self._log(f"HS fin best_hp={best_x} mejor_fitness={best_f} evaluaciones_totales={self.eval_count}")
        return best_x, best_f, self.HM
