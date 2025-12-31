# HS Transformer Quechua

Optimización de un Transformer de traducción español→quechua usando Harmony Search. Se incluye entrenamiento con checkpoints/reanudación, manejo de OOM en el orquestador y un dashboard HTML para explorar los trials.

## Requisitos rápidos
- Python 3.10 (sugerido conda: `conda create -n hs_transformer_quechua python=3.10 -y; conda activate hs_transformer_quechua`).
- `pip install -r requirements.txt` (pandas, openpyxl, torch, numpy, nltk, rouge-score).
- GPU opcional; si no hay GPU, el entrenamiento usará CPU (más lento).

## Estructura clave
- `main.py`: arranca la búsqueda de hiperparámetros (Harmony Search) y usa el orquestador.
- `search_orchestrator.py`: define `SEARCH_SPACE`, `fitness_fn` y lanza `HarmonySearch`; guarda trials en `hs_runs/` y maneja OOM penalizando el fitness.
- `utils/harmony_search.py`: implementación HS con reanudación; guarda estado en `hs_runs/state_hs.json` tras cada iteración.
- `model/training.py`: carga/preprocesa datos, entrena el Transformer y soporta checkpoints (`resume_from`, `checkpoint_every`).
- `hs_runs/`: se guardan `trial_*.json`, `hs_best.json`, `state_hs.json` y el `dashboard.html`.
- `utils/reporting.py`: utilidades para leer trials y graficar métricas.
- `scripts/gen_dashboard.py`: genera un dashboard interactivo (Bootstrap + Chart.js) para filtrar/ordenar trials y ver métricas.
- `ga_orchestrator.py`: búsqueda con Algoritmo Genético (GA) como alternativa a HS, con reanudación (`state_ga.json`).
- `utils/search_space.py`: define los espacios de búsqueda HS/GA centralizados.
- `config.py`: rutas comunes (dataset, salidas, checkpoints).

## Uso básico
1) Activar entorno e instalar dependencias.
2) Asegurar dataset en `data/` (ruta configurada en `search_orchestrator.py` → `DATASET_QUECHUA_PATH`).
3) Ejecutar búsqueda HS:  
   ```bash
   python main.py
   ```
   - Se generan `trial_*.json` y `hs_best.json` en `hs_runs/`.
   - Si hay OOM, el trial se marca con `oom: true`, BLEU=0 y la búsqueda continúa.
4) Ejecutar búsqueda GA:  
   ```bash
   python ga_orchestrator.py
   ```
   - Se generan `trial_ga_*.json` y `ga_best.json` en `hs_runs/`.
   - Reanuda desde `hs_runs/state_ga.json` si existe.
5) Dashboard de resultados:  
   ```bash
   python scripts/gen_dashboard.py
   # abrir hs_runs/dashboard.html en el navegador
   ```
   Permite filtrar por hiperparámetro y ordenar por métricas; cada trial abre un modal con gráficas de loss/BLEU/ROUGE.

## Checkpoints y reanudación
- Entrenamiento: `train_single_run` guarda un checkpoint cada `checkpoint_every` épocas en `checkpoints/ckpt_<timestamp>.pt`. Para reanudar:
  ```python
  train_single_run(hp, artifacts, resume_from="checkpoints/ckpt_XXXX.pt")
  ```
- Harmony Search: guarda estado en `hs_runs/state_hs.json` al final de cada iteración. Si el proceso se interrumpe (Ctrl+C o apagado), al relanzar `main.py` reanudará desde ese estado.

## Notas de espacio de búsqueda
- Ajusta `SEARCH_SPACE` a los límites de tu GPU/CPU. Para evitar OOM, usa batch_size ≤16, d_model ≤256, ffn_hidden ≤1024, num_layers 2–3, num_heads 4.

## Logs
- `utils/logger.py` escribe logs diarios en `logs/log_YYYYMMDD.txt` y también imprime por consola.

## Scripts útiles
- `scripts/gen_dashboard.py`: genera dashboard HTML con filtros/orden.  
- `utils/reporting.py`: carga trials y genera plots de loss/BLEU/ROUGE programáticamente.
