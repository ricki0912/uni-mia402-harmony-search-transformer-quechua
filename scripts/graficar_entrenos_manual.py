"""
Genera graficos (loss, BLEU, ROUGE) a partir de archivos hs_runs/manual_train_*.json.
Uso:
    python scripts/graficar_entrenos_manual.py            # procesa todos los manual_train_*.json
    python scripts/graficar_entrenos_manual.py --file hs_runs/manual_train_123.json  # solo uno
Guarda PNG y HTML junto al JSON.
"""

import argparse
import base64
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_records(files: List[Path]):
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        yield fp, data


def _save_html_with_image(png_path: Path, data: dict):
    """Genera un HTML simple con la imagen embebida y resumen de metricas."""
    png_b64 = png_path.read_bytes()
    b64_str = base64.b64encode(png_b64).decode("ascii")
    metrics = data.get("metrics", {})
    final = metrics.get("final_metrics", {})
    hp = data.get("hp", {})

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Manual Train {data.get('trial_id', '')}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1280px; margin: 0 auto; padding: 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }}
    .metrics {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .tag {{ background: #eef4ff; color: #1e3a8a; border-radius: 4px; padding: 4px 8px; font-size: 13px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Manual Train {data.get('trial_id', '')}</h1>
  <div class="card">
    <h3>Metricas por epoca</h3>
    <img src="data:image/png;base64,{b64_str}" alt="Graficos metricas" />
  </div>
  <div class="card">
    <div class="metrics">
      <div class="tag">BLEU final: {final.get('bleu')}</div>
      <div class="tag">Loss final: {final.get('loss')}</div>
      <div class="tag">ROUGE1: {final.get('rouge1')}</div>
      <div class="tag">ROUGE2: {final.get('rouge2')}</div>
      <div class="tag">ROUGEL: {final.get('rougeL')}</div>
      <div class="tag">Epocas: {metrics.get('epochs_run')}</div>
      <div class="tag">Duracion (s): {data.get('duration_seconds')}</div>
    </div>
  </div>
  <div class="card">
    <h3>Hiperparametros</h3>
    <table>
      <tbody>
        {''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in hp.items())}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    html_path = png_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML guardado en {html_path}")


def plot_history(fp: Path, data: dict):
    metrics = data.get("metrics", {})
    loss_hist = metrics.get("loss_history", [])
    bleu_hist = metrics.get("bleu_history", [])
    rouge = metrics.get("rouge", {})
    rouge1 = rouge.get("rouge1_history", [])
    rouge2 = rouge.get("rouge2_history", [])
    rougeL = rouge.get("rougeL_history", [])

    epochs = list(range(1, len(loss_hist) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    fig.suptitle(f"Manual Train {data.get('trial_id', '')}", fontsize=14)

    axes[0, 0].plot(epochs, loss_hist, label="Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, bleu_hist, label="BLEU", color="orange")
    axes[0, 1].set_title("BLEU")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("BLEU")
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, rouge1, label="ROUGE-1")
    axes[1, 0].plot(epochs, rouge2, label="ROUGE-2")
    axes[1, 0].plot(epochs, rougeL, label="ROUGE-L")
    axes[1, 0].set_title("ROUGE")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].axis("off")

    png_path = fp.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Grafico guardado en {png_path}")
    _save_html_with_image(png_path, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        help="Ruta a un manual_train_*.json especifico. Si se omite, procesa todos.",
    )
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(Path("hs_runs").glob("manual_train_*.json"))

    if not files:
        print("No se encontraron archivos manual_train_*.json")
        return

    for fp, data in load_records(files):
        plot_history(fp, data)


if __name__ == "__main__":
    main()
