import json
from pathlib import Path

# Dashboard interactivo (Bootstrap + Chart.js) para explorar los trial_*.json.
# Filtros: un combo múltiple por cada hiperparámetro; modal con gráficas.

TRIALS_DIR = Path("hs_runs")
OUT_HTML = TRIALS_DIR / "dashboard.html"


def load_trials():
    trials = []
    for p in sorted(TRIALS_DIR.glob("trial_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        trials.append({"path": p.name, **data})
    return trials


def render(trials):
    data_json = json.dumps(trials, ensure_ascii=False)
    template = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Harmony Search Trials</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
  <style>
    body { padding: 1.5rem; }
    td, th { vertical-align: top; }
    code { font-size: 0.9rem; }
  </style>
</head>
<body>
  <h2 class="mb-3">Harmony Search Trials</h2>
  <div class="row g-2 mb-3" id="summaryRow">
    <!-- resumen global -->
  </div>
  <div class="row g-3 mb-4">
    <div class="col-lg-6">
      <div class="card shadow-sm">
        <div class="card-body">
          <h6 class="card-title">BLEU HS a lo largo del tiempo</h6>
          <canvas id="bleuOverTimeHS"></canvas>
        </div>
      </div>
    </div>
  </div>
  <div class="row g-2 mb-3">
    <div class="col-12">
      <div class="row" id="hpFilters">
        <!-- combos por hiperparámetro -->
      </div>
    </div>
    <div class="col-12 d-flex justify-content-between align-items-center mt-2">
      <div class="d-flex gap-2 align-items-center">
        <label class="form-label mb-0">Ordenar por:</label>
        <select id="sortField" class="form-select form-select-sm" style="max-width: 180px;">
          <option value="">(sin orden)</option>
          <option value="loss">Loss</option>
          <option value="bleu">BLEU</option>
          <option value="rouge1">ROUGE-1</option>
          <option value="rouge2">ROUGE-2</option>
          <option value="rougeL">ROUGE-L</option>
          <option value="epochs_run">Epochs</option>
        </select>
        <select id="sortDir" class="form-select form-select-sm" style="max-width: 120px;">
          <option value="desc">Descendente</option>
          <option value="asc">Ascendente</option>
        </select>
      </div>
      <button id="clearFilters" class="btn btn-secondary">Limpiar filtros</button>
    </div>
  </div>

  <div class="table-responsive">
    <table class="table table-sm table-hover" id="tbl">
      <thead class="table-light">
        <tr>
          <th>Algo</th>
          <th>Archivo</th>
          <th>Hiperparámetros</th>
          <th>Loss</th>
          <th>BLEU</th>
          <th>ROUGE-1</th>
          <th>ROUGE-2</th>
          <th>ROUGE-L</th>
          <th>Epochs</th>
          <th>Acciones</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="modal fade" id="chartModal" tabindex="-1" aria-labelledby="chartModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg modal-dialog-scrollable">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="chartModalLabel">Detalles</h5>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div id="hpDetail" class="mb-3"></div>
          <div class="row g-3">
            <div class="col-md-6">
              <canvas id="lossChart"></canvas>
            </div>
            <div class="col-md-6">
              <canvas id="bleuChart"></canvas>
            </div>
            <div class="col-md-12">
              <canvas id="rougeChart"></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const trials = __DATA__;
    let lossChart, bleuChart, rougeChart;
    let bleuTimeChartHS;

    const tbody = document.querySelector('#tbl tbody');
    const summaryRow = document.getElementById('summaryRow');
    const hpFiltersContainer = document.getElementById('hpFilters');
    const clearFiltersBtn = document.getElementById('clearFilters');
    const sortField = document.getElementById('sortField');
    const sortDir = document.getElementById('sortDir');

    function hpToHtml(hp) {
      return Object.entries(hp).map(([k,v]) => `<code>${k}: ${v}</code>`).join('<br>');
    }

    function matchesFilters(trial) {
      const hpSelections = hpFiltersContainer.querySelectorAll('select[data-hpkey]');
      for (const sel of hpSelections) {
        const key = sel.dataset.hpkey;
        const selectedVals = Array.from(sel.selectedOptions).map(o => o.value);
        if (!selectedVals.length) continue;
        const val = (trial.hp || {})[key];
        if (!selectedVals.includes(String(val))) return false;
      }
      return true;
    }

    function renderTable() {
      tbody.innerHTML = '';
      const filtered = trials.filter(matchesFilters);

      const field = sortField.value;
      const dir = sortDir.value === 'asc' ? 1 : -1;
      if (field) {
        filtered.sort((a, b) => {
          const am = a.metrics || {};
          const bm = b.metrics || {};
          const afm = am.final_metrics || {};
          const bfm = bm.final_metrics || {};
          const aval = field === 'epochs_run' ? (am.epochs_run ?? NaN) : (afm[field] ?? NaN);
          const bval = field === 'epochs_run' ? (bm.epochs_run ?? NaN) : (bfm[field] ?? NaN);
          if (isNaN(aval) && isNaN(bval)) return 0;
          if (isNaN(aval)) return 1;
          if (isNaN(bval)) return -1;
          return (aval - bval) * dir;
        });
      }

      filtered.forEach((t, idx) => {
        const m = t.metrics || {};
        const fm = m.final_metrics || {};
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${t.algo || 'HS'}</td>
          <td>${t.path}</td>
          <td>${hpToHtml(t.hp || {})}</td>
          <td>${fm.loss ?? ''}</td>
          <td>${fm.bleu ?? ''}</td>
          <td>${fm.rouge1 ?? ''}</td>
          <td>${fm.rouge2 ?? ''}</td>
          <td>${fm.rougeL ?? ''}</td>
          <td>${m.epochs_run ?? ''}</td>
          <td><button class="btn btn-sm btn-primary" data-idx="${idx}" data-bs-toggle="modal" data-bs-target="#chartModal">Ver</button></td>
        `;
        tbody.appendChild(row);
      });
      attachRowEvents();
    }

    function attachRowEvents() {
      tbody.querySelectorAll('button[data-idx]').forEach(btn => {
        btn.onclick = () => openModal(trials[btn.dataset.idx]);
      });
    }

    function openModal(trial) {
      const m = trial.metrics || {};
      const losses = m.loss_history || [];
      const bleus = m.bleu_history || [];
      const rouge = (m.rouge) || {};
      const r1 = rouge.rouge1_history || [];
      const r2 = rouge.rouge2_history || [];
      const rL = rouge.rougeL_history || [];

      document.getElementById('chartModalLabel').textContent = trial.path;
      document.getElementById('hpDetail').innerHTML = hpToHtml(trial.hp || {});

      const labels = losses.map((_, i) => `epoch ${i+1}`);

      if (lossChart) lossChart.destroy();
      lossChart = new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: { labels, datasets: [{ label: 'Loss', data: losses, borderColor: '#d9534f', tension: 0.2 }] },
      });

      if (bleuChart) bleuChart.destroy();
      bleuChart = new Chart(document.getElementById('bleuChart'), {
        type: 'line',
        data: { labels, datasets: [{ label: 'BLEU', data: bleus, borderColor: '#f0ad4e', tension: 0.2 }] },
        options: { scales: { y: { min: 0, max: 1 } } }
      });

      if (rougeChart) rougeChart.destroy();
      rougeChart = new Chart(document.getElementById('rougeChart'), {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: 'ROUGE-1', data: r1, borderColor: '#5cb85c', tension: 0.2 },
            { label: 'ROUGE-2', data: r2, borderColor: '#428bca', tension: 0.2 },
            { label: 'ROUGE-L', data: rL, borderColor: '#d9534f', tension: 0.2 },
          ]
        },
        options: { scales: { y: { min: 0, max: 1 } } }
      });
    }

    function buildHpFilters() {
      hpFiltersContainer.innerHTML = '';
      const keys = new Set();
      trials.forEach(t => Object.keys(t.hp || {}).forEach(k => keys.add(k)));
      Array.from(keys).sort().forEach(k => {
        const col = document.createElement('div');
        col.className = 'col-xl-2 col-lg-3 col-md-4 col-sm-6 mb-2';
        const label = document.createElement('label');
        label.className = 'form-label mb-1';
        label.textContent = k;
        const select = document.createElement('select');
        select.className = 'form-select form-select-sm';
        select.multiple = true;
        select.size = 4;
        select.dataset.hpkey = k;
        const values = new Set();
        trials.forEach(t => {
          const v = (t.hp || {})[k];
          if (v !== undefined) values.add(String(v));
        });
        Array.from(values).sort().forEach(v => {
          const opt = document.createElement('option');
          opt.value = opt.textContent = v;
          select.appendChild(opt);
        });
        select.addEventListener('change', renderTable);
        col.appendChild(label);
        col.appendChild(select);
        hpFiltersContainer.appendChild(col);
      });
    }

    clearFiltersBtn.onclick = () => {
      buildHpFilters();
      renderTable();
    };

    sortField.addEventListener('change', renderTable);
    sortDir.addEventListener('change', renderTable);

    function renderSummary() {
      const total = trials.length;
      const bestBleu = (arr) => {
        if (!arr.length) return 0;
        return Math.max(...arr.map(t => (t.metrics?.final_metrics?.bleu ?? t.metrics?.bleu ?? 0) || 0));
      };
      const bestHS = bestBleu(trials);
      const oomCount = trials.filter(t => t.metrics?.oom).length;
      const sumTime = (arr) => arr.reduce((acc, t) => acc + (t.metrics?.stats?.training_time_minutes || 0), 0);
      const timeHS = sumTime(trials);

      summaryRow.innerHTML = `
        <div class="col-sm-6 col-lg-3">
          <div class="card shadow-sm">
            <div class="card-body">
              <h6 class="card-title mb-1">Total trials</h6>
              <div class="fs-5 fw-bold">${total}</div>
              <small>Solo Harmony Search</small>
            </div>
          </div>
        </div>
        <div class="col-sm-6 col-lg-3">
          <div class="card shadow-sm">
            <div class="card-body">
              <h6 class="card-title mb-1">Mejor BLEU</h6>
              <div class="fs-5 fw-bold">${bestHS.toFixed(6)}</div>
              <small>Mejor BLEU registrado</small>
            </div>
          </div>
        </div>
        <div class="col-sm-6 col-lg-3">
          <div class="card shadow-sm">
            <div class="card-body">
              <h6 class="card-title mb-1">OOM (penalizados)</h6>
              <div class="fs-5 fw-bold">${oomCount}</div>
              <small>Trials con oom=true</small>
            </div>
          </div>
        </div>
        <div class="col-sm-6 col-lg-3">
          <div class="card shadow-sm">
            <div class="card-body">
              <h6 class="card-title mb-1">Tiempo entrenado (min)</h6>
              <div class="fs-5 fw-bold">${timeHS.toFixed(1)}</div>
              <small>Minutos totales reportados</small>
            </div>
          </div>
        </div>
      `;

      renderCharts();
    }

    function renderCharts() {
      // BLEU vs trial_id (ordenados) para HS
      const extractId = (t, idx) => {
        if (t.trial_id) return t.trial_id;
        const m = (t.path || "").match(/(\d{10,})/);
        if (m) return parseInt(m[1], 10);
        return idx + 1; // fallback
      };
      const hsSeq = [];
      trials.forEach((t, idx) => {
        const bleu = t.metrics?.final_metrics?.bleu ?? t.metrics?.bleu ?? 0;
        const algo = (t.algo || 'HS').toUpperCase();
        const id = extractId(t, idx);
        if (algo === 'HS') hsSeq.push({ id, bleu });
      });
      hsSeq.sort((a,b) => a.id - b.id);

      const makeLine = (canvasId, arr, label, color) => {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        return new Chart(ctx, {
          type: 'line',
          data: {
            labels: arr.map(p => p.id),
            datasets: [{
              label,
              data: arr.map(p => p.bleu),
              borderColor: color,
              backgroundColor: color,
              tension: 0.1
            }]
          },
          options: {
            scales: { y: { min: 0 } },
            plugins: { legend: { display: true } }
          }
        });
      };

      if (bleuTimeChartHS) bleuTimeChartHS.destroy();
      bleuTimeChartHS = makeLine('bleuOverTimeHS', hsSeq, 'HS BLEU', '#428bca');
    }
    renderSummary();
    buildHpFilters();
    renderTable();
  </script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    return template.replace("__DATA__", data_json)


def main():
    trials = load_trials()
    OUT_HTML.write_text(render(trials), encoding="utf-8")
    print(f"Dashboard generado en {OUT_HTML}")


if __name__ == "__main__":
    main()
