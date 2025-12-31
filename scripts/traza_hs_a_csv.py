"""
Convierte traces de Harmony Search (trace_*.json) en una tabla CSV con hiperparámetros y BLEU por iteración/armonía.
Uso:
    python scripts/traza_hs_a_csv.py                   # procesa todos los hs_runs/trace_*.json
    python scripts/traza_hs_a_csv.py --file hs_runs/trace_20250101_120000.json
Genera un CSV junto al trace con sufijo `_table.csv`.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def load_trace(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def trace_to_rows_best_per_iter(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Devuelve solo la mejor armonía (mayor BLEU) por iteración.
    """
    rows: List[Dict[str, Any]] = []
    iterations = trace.get("iterations", [])
    for iter_entry in iterations:
        iter_idx = iter_entry.get("iter_idx")
        improvisations = iter_entry.get("improvisations", [])
        if not improvisations:
            continue
        best_imp = max(improvisations, key=lambda imp: imp.get("fitness", float("-inf")))
        row = {
            "iter_idx": iter_idx,
            "harmony_idx": best_imp.get("harmony_idx"),
            "bleu": best_imp.get("fitness"),
        }
        hp = best_imp.get("x_new", {}) or {}
        for k, v in hp.items():
            row[f"hp.{k}"] = v
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        print(f"Sin datos para escribir en {out_path}")
        return
    # headers: fixed columns first, then hp.*
    fixed = ["iter_idx", "harmony_idx", "bleu"]
    hp_keys = sorted({k for r in rows for k in r.keys() if k.startswith("hp.")})
    headers = fixed + hp_keys
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV generado: {out_path}")


def process_file(path: Path):
    trace = load_trace(path)
    rows = trace_to_rows_best_per_iter(trace)
    out_path = path.with_name(path.stem + "_best_per_iter.csv")
    write_csv(rows, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Ruta a un trace_*.json específico.")
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(Path("hs_runs").glob("trace_*.json"))

    if not files:
        print("No se encontraron traces trace_*.json en hs_runs.")
        return

    for fp in files:
        process_file(fp)


if __name__ == "__main__":
    main()
