"""
Genera un dashboard HTML interactivo a partir de un trace JSON de Harmony Search.
Si no se pasa --trace, usa el trace_*.json más reciente en hs_runs/.
"""

import argparse
import json
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Plantilla HTML con Plotly + Tabulator y navegación por hash.
HTML_TEMPLATE = string.Template(
    """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Dashboard Harmony Search</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tabulator-tables@5.6.2/dist/css/tabulator.min.css">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/tabulator-tables@5.6.2/dist/js/tabulator.min.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --surface: #111827;
      --border: #1f2937;
      --text: #e2e8f0;
      --muted: #cbd5e1;
      --primary: #38bdf8;
      --btn: #2563eb;
      --code: #0b1220;
    }
    body.light {
      --bg: #ffffff;
      --surface: #f8fafc;
      --border: #cbd5e1;
      --text: #0f172a;
      --muted: #475569;
      --primary: #2563eb;
      --btn: #2563eb;
      --code: #e2e8f0;
    }
    body { font-family: Arial, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
    header { padding: 16px 24px; background: var(--surface); position: sticky; top: 0; z-index: 100; display: flex; align-items: center; gap: 12px; }
    h1 { margin: 0; font-size: 22px; flex: 1; }
    nav { display: flex; gap: 12px; margin-top: 0; flex-wrap: wrap; }
    nav a { color: var(--muted); text-decoration: none; padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); }
    nav a.active { background: var(--border); color: var(--primary); }
    main { padding: 16px 24px 32px; }
    .section { display: none; }
    .section.active { display: block; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 12px 0 20px; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
    .card h3 { margin: 0 0 4px; font-size: 14px; color: var(--muted); }
    .card .val { font-size: 18px; font-weight: 700; color: var(--primary); }
    .grid-two { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
    .chart { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
    .chart h3 { margin: 0 0 6px; font-size: 15px; color: var(--muted); }
    .table-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
    button { background: var(--btn); color: white; border: none; padding: 6px 10px; border-radius: 6px; cursor: pointer; }
    button.secondary { background: var(--border); color: var(--text); }
    pre { background: var(--code); padding: 10px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); }
  </style>
</head>
<body>
  <header>
    <h1>Dashboard Harmony Search</h1>
    <button class="secondary" onclick="toggleTheme()">Modo claro/oscuro</button>
    <nav id="nav"></nav>
  </header>
  <main id="content"></main>

  <script>
    const payload = $payload_json;
    const sections = [
      {id: "summary", label: "Resumen"},
      {id: "kpis", label: "KPIs"},
      {id: "besthp", label: "Best HP"},
      {id: "convergence", label: "Convergencia"},
      {id: "timeline", label: "Evaluaciones"},
      {id: "tables", label: "Tablas"},
      {id: "distrib", label: "Distribuciones"},
      {id: "relations", label: "HP vs Fitness"},
      {id: "diagnostics", label: "Diagnostico"},
      {id: "flow", label: "Diagrama"},
      {id: "audit", label: "Auditoria"},
      {id: "insights", label: "Insights"}
    ];
    const content = document.getElementById("content");
    const nav = document.getElementById("nav");
    function renderNav() {
      nav.innerHTML = "";
      sections.forEach(s => {
        const a = document.createElement("a");
        a.href = `#${s.id}`;
        a.textContent = s.label;
        if (location.hash === `#${s.id}`) a.classList.add("active");
        nav.appendChild(a);
      });
    }
    function card(title, val) { return `<div class="card"><h3>${title}</h3><div class="val">${val}</div></div>`; }
    function renderSummary() {
      const m = payload.meta, b = payload.best;
      return `<section class="section" id="summary"><div class="cards">
        ${card("Algoritmo", m.algo)}${card("Timestamp", m.timestamp)}${card("Seed", m.seed)}${card("Objetivo", m.objetivo)}
        ${card("HMS", m.HMS)}${card("HMCR", m.HMCR)}${card("PAR", m.PAR)}${card("BW", m.BW)}${card("NI", m.NI)}
        ${card("Eval totales", b.eval_count_total)}${card("Best fitness final", b.best_fitness.toFixed(6))}
      </div></section>`;
    }
    function renderKPIs() {
      const k = payload.kpis;
      const pct = k.delta_pct !== null ? (k.delta_pct*100).toFixed(2)+\"%\" : \"n/a\";
      return `<section class="section" id="kpis"><div class="cards">
        ${card("Best final", k.best_final.toFixed(6))}${card("Best inicial", k.best_init.toFixed(6))}
        ${card("Δ absoluta", k.delta_abs.toFixed(6))}${card("% mejora", pct)}${card("Worst final", k.worst_final.toFixed(6))}
        ${card("Reemplazos", k.replacements)}${card("Replace rate", (k.replace_rate*100).toFixed(2)+"%")}
        ${card("Iteraciones", k.NI)}${card("# params", k.num_params)}
      </div></section>`;
    }
    function renderBestHP() {
      const hp = payload.best.best_hp;
      const rows = Object.entries(hp).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
      return `<section class="section" id="besthp"><div class="table-wrap"><h3>Mejor configuración</h3>
        <table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:4px;">Parámetro</th><th style="text-align:left; padding:4px;">Valor</th></tr></thead><tbody>${rows}</tbody></table>
        <div style="margin-top:8px;"><button onclick="copyBest()">Copiar config JSON</button><pre id="best-json">${JSON.stringify(hp, null, 2)}</pre></div></div></section>`;
    }
    function renderConvergence() {
      return `<section class="section" id="convergence"><div class="grid-two">
        <div class="chart"><h3>Mejor / Peor por iteración</h3><div id="line-best"></div></div>
        <div class="chart"><h3>Delta best por iteración</h3><div id="bar-delta"></div></div>
        <div class="chart"><h3>Eval acumulado vs after_best</h3><div id="line-eval"></div></div>
      </div></section>`;
    }
    function renderTimeline() { return `<section class="section" id="timeline"><div class="chart"><h3>Fitness por evaluación</h3><div id="scatter-evals"></div></div></section>`; }
    function renderTables() { return `<section class="section" id="tables">
      <div class="table-wrap"><h3>Todas las evaluaciones</h3><div id="table-evals"></div></div>
      <div class="table-wrap"><h3>Top-K mejores</h3><div id="table-top"></div></div>
      <div class="table-wrap"><h3>Top-K peores</h3><div id="table-worst"></div></div>
      <div class="table-wrap"><h3>Harmony Memory inicial</h3><div id="table-init"></div></div>
    </section>`; }
    function renderDistrib() { return `<section class="section" id="distrib"><div class="grid-two">
      <div class="chart"><h3>Histograma fitness</h3><div id="hist-fitness"></div></div>
      <div class="chart"><h3>Boxplot fitness por iteración</h3><div id="box-iter"></div></div>
      <div class="chart"><h3>Histograma d_model</h3><div id="hist-dmodel"></div></div>
      <div class="chart"><h3>Histograma ffn_hidden</h3><div id="hist-ffn"></div></div>
      <div class="chart"><h3>Histograma num_heads</h3><div id="hist-heads"></div></div>
      <div class="chart"><h3>Histograma num_layers</h3><div id="hist-layers"></div></div>
      <div class="chart"><h3>Histograma batch_size</h3><div id="hist-batch"></div></div>
      <div class="chart"><h3>Histograma epochs</h3><div id="hist-epochs"></div></div>
      <div class="chart"><h3>Histograma drop_prob</h3><div id="hist-drop"></div></div>
      <div class="chart"><h3>Histograma lr (log bins)</h3><div id="hist-lr"></div></div>
    </div></section>`; }
    function renderRelations() { return `<section class="section" id="relations"><div class="grid-two">
      <div class="chart"><h3>lr vs fitness</h3><div id="scat-lr"></div></div>
      <div class="chart"><h3>drop_prob vs fitness</h3><div id="scat-drop"></div></div>
      <div class="chart"><h3>d_model vs fitness</h3><div id="scat-dmodel"></div></div>
      <div class="chart"><h3>ffn_hidden vs fitness</h3><div id="scat-ffn"></div></div>
      <div class="chart"><h3>num_layers vs fitness</h3><div id="scat-layers"></div></div>
      <div class="chart"><h3>batch_size vs fitness</h3><div id="scat-batch"></div></div>
    </div></section>`; }
    function renderDiagnostics() { return `<section class="section" id="diagnostics"><div class="grid-two">
      <div class="chart"><h3>Uso HM vs Rand (global)</h3><div id="bar-hmrand-global"></div></div>
      <div class="chart"><h3>Uso HM vs Rand por iteracion</h3><div id="bar-hmrand-iter"></div></div>
      <div class="chart"><h3>Uso HM vs Rand por parametro</h3><div id="bar-hmrand-param"></div></div>
      <div class="chart"><h3>Pitch aplicado (tasa global / signo)</h3><div id="bar-pitch"></div></div>
      <div class="chart"><h3>Clamps por parametro</h3><div id="bar-clamp-param"></div></div>
      <div class="chart"><h3>Reemplazos por iteracion</h3><div id="bar-replace-iter"></div></div>
    </div><div class="table-wrap"><h3>Eventos clamp</h3><div id="table-clamps"></div></div></section>`; }
    function renderFlow() { return `<section class="section" id="flow">
      <div style="margin-bottom:8px; display:flex; gap:8px;">
        <button onclick="expandAll()">Expandir todo</button>
        <button onclick="collapseAll()">Colapsar todo</button>
      </div>
      <div class="card" style="background: var(--surface); border:1px solid var(--border); padding:10px;">
        ${payload.tree_html}
      </div>
    </section>`; }
    function renderAudit() { return `<section class="section" id="audit"><div class="card" style="background:#111827;">
      <h3>run_meta</h3><pre>${JSON.stringify(payload.meta_raw, null, 2)}</pre></div>
      <div style="margin-top:8px;"><button onclick="downloadJSON()">Descargar trace</button>
      <button class="secondary" onclick="downloadCSV()">Exportar CSV evaluaciones</button></div></section>`; }
    function renderInsights() { const list = payload.insights.map(t => `<li>${t}</li>`).join(""); return `<section class="section" id="insights"><div class="card" style="background:#111827;"><h3>Insights</h3><ul>${list}</ul></div></section>`; }
    function render() {
      renderNav();
      content.innerHTML = [renderSummary(),renderKPIs(),renderBestHP(),renderConvergence(),renderTimeline(),renderTables(),renderDistrib(),renderRelations(),renderDiagnostics(),renderFlow(),renderAudit(),renderInsights()].join("");
      const target = location.hash ? location.hash.substring(1) : "summary";
      activate(target);
      setTimeout(drawCharts,0); setTimeout(drawTables,0);
    }
    function activate(id) {
      document.querySelectorAll(".section").forEach(sec => sec.classList.remove("active"));
      const sec = document.getElementById(id); if (sec) sec.classList.add("active");
      nav.querySelectorAll("a").forEach(a => a.classList.toggle("active", a.getAttribute("href")===`#${id}`));
    }
    window.addEventListener("hashchange", ()=>{ const target = location.hash ? location.hash.substring(1) : "summary"; activate(target); });
    function copyBest(){ navigator.clipboard.writeText(JSON.stringify(payload.best.best_hp,null,2)); alert("Config copiada"); }
    function toggleTheme(){ document.body.classList.toggle("light"); drawCharts(); }
    function expandAll(){ document.querySelectorAll("#flow details").forEach(d=>d.open=true); }
    function collapseAll(){ document.querySelectorAll("#flow details").forEach(d=>d.open=false); }
    function downloadJSON(){ const blob=new Blob([JSON.stringify(payload.trace_raw,null,2)],{type:"application/json"}); const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=payload.meta.trace_name||"trace.json"; a.click(); }
    function downloadCSV(){ const rows=payload.tables.evals.map(r=>({iter_idx:r.iter_idx,harmony_idx:r.harmony_idx,fitness:r.fitness,replace:r.replace,...r.hp})); const header=Object.keys(rows[0]||{}); const csv=[header.join(\",\")].concat(rows.map(r=>header.map(h=>r[h]).join(\",\"))).join(\"\\n\"); const blob=new Blob([csv],{type:\"text/csv\"}); const a=document.createElement(\"a\"); a.href=URL.createObjectURL(blob); a.download=\"evaluations.csv\"; a.click(); }

    function drawCharts(){
      Plotly.newPlot("line-best",[
        {x:payload.series.iter_idx,y:payload.series.after_best,mode:"lines+markers",name:"after_best",line:{color:"#38bdf8"}},
        {x:payload.series.iter_idx,y:payload.series.after_worst,mode:"lines+markers",name:"after_worst",line:{color:"#f97316"}},
        {x:payload.series.iter_idx,y:payload.series.before_best,mode:"lines+markers",name:"before_best",line:{color:"#22c55e",dash:"dash"}},
        {x:payload.series.iter_idx,y:payload.series.before_worst,mode:"lines+markers",name:"before_worst",line:{color:"#eab308",dash:"dash"}},
      ],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
      Plotly.newPlot("bar-delta",[{x:payload.series.iter_idx,y:payload.series.delta_best,type:"bar",marker:{color:"#38bdf8"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"},xaxis:{title:"iter"},yaxis:{title:"Δ best"}});
      Plotly.newPlot("line-eval",[{x:payload.series.eval_cum,y:payload.series.after_best,mode:"lines+markers",name:"after_best",line:{color:"#38bdf8"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"},xaxis:{title:"eval acumulado"},yaxis:{title:"after_best"}});
      Plotly.newPlot("scatter-evals",[{x:payload.evals.map(e=>e.eval_id),y:payload.evals.map(e=>e.fitness),mode:"markers",name:"fitness",marker:{color:payload.evals.map(e=>e.replace?\"#34d399\":\"#94a3b8\"),size:8}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"},xaxis:{title:"eval_id"},yaxis:{title:"fitness"},shapes:[{type:"line",x0:0,x1:payload.evals.length+1,y0:payload.best.best_fitness,y1:payload.best.best_fitness,line:{dash:"dot",color:"#f43f5e"}}]});
      Plotly.newPlot("hist-fitness",[{x:payload.distrib.fitness_all,type:"histogram",marker:{color:"#38bdf8"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
      Plotly.newPlot("box-iter",payload.distrib.box_iter,{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
      const histParams=[["hist-dmodel","d_model"],["hist-ffn","ffn_hidden"],["hist-heads","num_heads"],["hist-layers","num_layers"],["hist-batch","batch_size"],["hist-epochs","epochs"]];
      histParams.forEach(([div,key])=>{Plotly.newPlot(div,[{x:payload.distrib.param_values[key],type:"histogram",marker:{color:"#38bdf8"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"},xaxis:{title:key}});});
      Plotly.newPlot("hist-drop",[{x:payload.distrib.param_values["drop_prob"],type:"histogram",marker:{color:"#38bdf8"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"},xaxis:{title:"drop_prob"}});
      Plotly.newPlot("hist-lr",[{x:payload.distrib.param_values["lr"],type:"histogram",marker:{color:"#38bdf8"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"},xaxis:{title:"lr (log bins)"},xbins:{size:0.0001}});
      const scatters=[["scat-lr","lr"],["scat-drop","drop_prob"],["scat-dmodel","d_model"],["scat-ffn","ffn_hidden"],["scat-layers","num_layers"],["scat-batch","batch_size"]];
      scatters.forEach(([div,key])=>{Plotly.newPlot(div,[{x:payload.evals.map(e=>e.hp[key]),y:payload.evals.map(e=>e.fitness),mode:"markers",marker:{color:"#38bdf8"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"},xaxis:{title:key},yaxis:{title:"fitness"}});});
      Plotly.newPlot("bar-hmrand-global",[{x:["HM","Rand"],y:[payload.diagnostics.hm_usage.HM,payload.diagnostics.hm_usage.rand],type:"bar",marker:{color:["#38bdf8","#f97316"]}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
      Plotly.newPlot("bar-hmrand-iter",[{x:payload.series.iter_idx,y:payload.diagnostics.hm_iter.HM,type:"bar",name:"HM",marker:{color:"#38bdf8"}},{x:payload.series.iter_idx,y:payload.diagnostics.hm_iter.rand,type:"bar",name:"Rand",marker:{color:"#f97316"}}],{barmode:"stack",paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
      const params=Object.keys(payload.diagnostics.hm_param.HM); Plotly.newPlot("bar-hmrand-param",[{x:params,y:params.map(p=>payload.diagnostics.hm_param.HM[p]||0),type:"bar",name:"HM",marker:{color:"#38bdf8"}},{x:params,y:params.map(p=>payload.diagnostics.hm_param.rand[p]||0),type:"bar",name:"Rand",marker:{color:"#f97316"}}],{barmode:"stack",paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
      Plotly.newPlot("bar-pitch",[{x:["Pitch aplicado","Signo +","Signo -"],y:[payload.diagnostics.pitch.count_applied,payload.diagnostics.pitch.sign_pos,payload.diagnostics.pitch.sign_neg],type:"bar",marker:{color:"#38bdf8"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
      Plotly.newPlot("bar-clamp-param",[{x:Object.keys(payload.diagnostics.clamp_param),y:Object.values(payload.diagnostics.clamp_param),type:"bar",marker:{color:"#f43f5e"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
      Plotly.newPlot("bar-replace-iter",[{x:payload.series.iter_idx,y:payload.diagnostics.replace_per_iter,type:"bar",marker:{color:"#34d399"}}],{paper_bgcolor:"#111827",plot_bgcolor:"#111827",font:{color:"#e2e8f0"}});
    }

    function drawTables(){
      new Tabulator("#table-evals",{data:payload.tables.evals,layout:"fitColumns",height:300,columns:[
        {title:"iter",field:"iter_idx",width:60},{title:"harmony",field:"harmony_idx",width:80},{title:"fitness",field:"fitness",formatter:"money",formatterParams:{precision:6}},{title:"replace",field:"replace",width:80},
        {title:"d_model",field:"hp.d_model"},{title:"ffn_hidden",field:"hp.ffn_hidden"},{title:"num_heads",field:"hp.num_heads"},{title:"drop_prob",field:"hp.drop_prob"},{title:"num_layers",field:"hp.num_layers"},{title:"lr",field:"hp.lr"},{title:"batch_size",field:"hp.batch_size"},{title:"max_seq",field:"hp.max_sequence_length"},{title:"epochs",field:"hp.epochs"},
      ]});
      new Tabulator("#table-top",{data:payload.tables.top,layout:"fitColumns",height:220,columns:[
        {title:"fitness",field:"fitness",formatter:"money",formatterParams:{precision:6}},{title:"iter",field:"iter_idx",width:60},{title:"harmony",field:"harmony_idx",width:80},
      ]});
      new Tabulator("#table-worst",{data:payload.tables.worst,layout:"fitColumns",height:220,columns:[
        {title:"fitness",field:"fitness",formatter:"money",formatterParams:{precision:6}},{title:"iter",field:"iter_idx",width:60},{title:"harmony",field:"harmony_idx",width:80},
      ]});
      new Tabulator("#table-init",{data:payload.tables.init_hm,layout:"fitColumns",height:220,columns:[
        {title:"fitness",field:"fitness",formatter:"money",formatterParams:{precision:6}},{title:"d_model",field:"hp.d_model"},{title:"ffn_hidden",field:"hp.ffn_hidden"},{title:"num_heads",field:"hp.num_heads"},{title:"drop_prob",field:"hp.drop_prob"},{title:"num_layers",field:"hp.num_layers"},{title:"lr",field:"hp.lr"},{title:"batch_size",field:"hp.batch_size"},{title:"max_seq",field:"hp.max_sequence_length"},{title:"epochs",field:"hp.epochs"},
      ]});
      new Tabulator("#table-clamps",{data:payload.diagnostics.clamps,layout:"fitColumns",height:200,columns:[
        {title:"iter",field:"iter_idx",width:60},{title:"harmony",field:"harmony_idx",width:80},{title:"param",field:"param",width:100},{title:"before",field:"before"},{title:"after",field:"after"},
      ]});
    }
    render();
  </script>
</body>
</html>"""
)


def find_latest_trace(hs_dir: Path) -> Path:
    traces = list(hs_dir.glob("trace_*.json"))
    if not traces:
        raise FileNotFoundError(f"No se encontraron trace_*.json en {hs_dir}")
    return max(traces, key=lambda p: p.stat().st_mtime)


def gather_evals(trace: dict) -> Tuple[List[dict], List[dict], List[int], List[int]]:
    evals = []
    replacements = []
    hm_iter_hm = []
    hm_iter_rand = []
    eval_id = 1
    for it in trace["iterations"]:
        hm_hm = 0
        hm_rand = 0
        for im in it["improvisations"]:
            for _, jd in im["j_decisions"].items():
                if jd.get("source") == "HM":
                    hm_hm += 1
                elif jd.get("source") == "rand":
                    hm_rand += 1
            evals.append(
                {
                    "eval_id": eval_id,
                    "iter_idx": it["iter_idx"],
                    "harmony_idx": im["harmony_idx"],
                    "fitness": im["fitness"],
                    "replace": im["replace"],
                    "hp": im["x_new"],
                }
            )
            if im["replace"]:
                replacements.append(
                    {
                        "iter_idx": it["iter_idx"],
                        "harmony_idx": im["harmony_idx"],
                        "fitness": im["fitness"],
                        "f_worst_before": im["f_worst_before"],
                        "f_worst_after": im["f_worst_after"],
                    }
                )
            eval_id += 1
        hm_iter_hm.append(hm_hm)
        hm_iter_rand.append(hm_rand)
    return evals, replacements, hm_iter_hm, hm_iter_rand


def clamp_events(trace: dict) -> List[dict]:
    events = []
    for it in trace["iterations"]:
        for im in it["improvisations"]:
            for param, jd in im["j_decisions"].items():
                if jd.get("clamp"):
                    events.append(
                        {
                            "iter_idx": it["iter_idx"],
                            "harmony_idx": im["harmony_idx"],
                            "param": param,
                            "before": jd.get("pitch_before"),
                            "after": jd.get("pitch_after"),
                        }
                    )
    return events


def hist_values(trace: dict) -> Dict[str, List[Any]]:
    vals: Dict[str, List[Any]] = defaultdict(list)
    for item in trace["init_hm"]:
        for k, v in item["hp"].items():
            vals[k].append(v)
    for it in trace["iterations"]:
        for im in it["improvisations"]:
            for k, v in im["x_new"].items():
                vals[k].append(v)
    return vals


def pitch_stats(trace: dict) -> dict:
    total = 0
    applied = 0
    sign_pos = 0
    sign_neg = 0
    for it in trace["iterations"]:
        for im in it["improvisations"]:
            for _, jd in im["j_decisions"].items():
                total += 1
                if jd.get("bw_usado") is not None:
                    applied += 1
                    signo = jd.get("signo")
                    if signo == "+":
                        sign_pos += 1
                    elif signo == "-":
                        sign_neg += 1
    return {"total": total, "count_applied": applied, "sign_pos": sign_pos, "sign_neg": sign_neg}


def hm_usage(trace: dict) -> dict:
    c = Counter()
    per_param_hm = Counter()
    per_param_rand = Counter()
    for it in trace["iterations"]:
        for im in it["improvisations"]:
            for param, jd in im["j_decisions"].items():
                src = jd.get("source")
                if src == "HM":
                    c["HM"] += 1
                    per_param_hm[param] += 1
                elif src == "rand":
                    c["rand"] += 1
                    per_param_rand[param] += 1
    return {
        "HM": c.get("HM", 0),
        "rand": c.get("rand", 0),
        "per_param_hm": per_param_hm,
        "per_param_rand": per_param_rand,
    }


def box_iter(trace: dict) -> List[dict]:
    data = []
    for it in trace["iterations"]:
        data.append(
            {
                "type": "box",
                "name": f"iter {it['iter_idx']}",
                "y": [im["fitness"] for im in it["improvisations"]],
                "marker": {"color": "#38bdf8"},
            }
        )
    return data


def build_tree_html(trace: dict) -> str:
    lines: List[str] = []

    def add(line: str, indent: int = 0) -> None:
        lines.append("  " * indent + line)

    # Init HM
    add("<details open><summary>Init HM</summary>")
    for idx, item in enumerate(trace.get("init_hm", []), start=1):
        fit = item.get("fitness", 0.0)
        hp_json = json.dumps(item.get("hp", {}), ensure_ascii=False, indent=2)
        add(f"<details><summary>i={idx} f={fit:.6f}</summary>", 1)
        add(f"<pre>{hp_json}</pre>", 2)
        add("</details>", 1)
    add("</details>")

    # Iterations
    for it in trace.get("iterations", []):
        ib = it.get("before_best", 0.0)
        ia = it.get("after_best", 0.0)
        iw = it.get("after_worst", 0.0)
        add(f"<details><summary>Iter {it.get('iter_idx')} (before_best={ib:.6f} after_best={ia:.6f} after_worst={iw:.6f})</summary>")
        for im in it.get("improvisations", []):
            rep = "SI" if im.get("replace") else "NO"
            fit = im.get("fitness", 0.0)
            add(f"<details><summary>harmony {im.get('harmony_idx')} f={fit:.6f} replace={rep}</summary>", 1)
            add("<ul>", 2)
            for param, jd in im.get("j_decisions", {}).items():
                source = jd.get("source")
                pb = jd.get("pitch_before")
                pa = jd.get("pitch_after")
                bw = jd.get("bw_usado")
                sign = jd.get("signo", "n/a")
                clamp = "SI" if jd.get("clamp") else "NO"
                add(f"<li><strong>{param}</strong>: src={source} pitch {pb}→{pa} bw={bw} sign={sign} clamp={clamp}</li>", 3)
            add("</ul>", 2)
            add("</details>", 1)
        add("</details>")

    return "\n".join(lines)


def build_payload(trace: dict, trace_path: Path) -> dict:
    meta = trace.get("run_meta", {})
    init_hm = trace.get("init_hm", []) or []
    iterations = trace.get("iterations", []) or []
    final = trace.get("final", {}) or {}
    best_hp = final.get("best_hp", {})
    best_fitness = float(final.get("best_fitness", 0.0))
    best_init = max(init_hm, key=lambda x: x.get("fitness", 0.0)).get("fitness", 0.0) if init_hm else 0.0
    worst_final = iterations[-1]["after_worst"] if iterations else 0.0

    evals, replacements, hm_iter_hm, hm_iter_rand = gather_evals(trace)
    hm_usage_stats = hm_usage(trace)
    clamps = clamp_events(trace)
    pitch = pitch_stats(trace)

    iter_idx = [it.get("iter_idx") for it in iterations]
    after_best = [it.get("after_best", 0.0) for it in iterations]
    after_worst = [it.get("after_worst", 0.0) for it in iterations]
    before_best = [it.get("before_best", 0.0) for it in iterations]
    before_worst = [it.get("before_worst", 0.0) for it in iterations]
    delta_best = [a - b for a, b in zip(after_best, before_best)]
    eval_cum = list(range(1, len(after_best) + 1))

    fitness_all = [item.get("fitness", 0.0) for item in init_hm] + [e["fitness"] for e in evals]
    param_values = hist_values(trace)
    replace_per_iter = [sum(1 for im in it.get("improvisations", []) if im.get("replace")) for it in iterations]

    clamp_param = Counter()
    for ev in clamps:
        clamp_param[ev["param"]] += 1

    top = sorted(evals, key=lambda x: x["fitness"], reverse=True)[:10]
    worst = sorted(evals, key=lambda x: x["fitness"])[:10]

    insights = [
        f"Mejor fitness {best_fitness:.6f} (best after_best final).",
        f"Mejora de {best_init:.6f} a {best_fitness:.6f} ({((best_fitness-best_init)/best_init*100):.2f}% si best_init>0)."
        if best_init > 0
        else "Mejora no calculable porque best_init es 0 o HM inicial vacia.",
        f"Reemplazos totales: {len(replacements)} / {len(evals)} evals (rate {(len(replacements)/len(evals)*100 if evals else 0):.2f}%).",
        f"Pitch aplicado {pitch['count_applied']} de {pitch['total']} parametros.",
        f"Clamps: {len(clamps)} eventos.",
        f"Uso HM vs rand (params): {hm_usage_stats['HM']} HM vs {hm_usage_stats['rand']} rand.",
    ]

    payload = {
        "meta": {
            "algo": meta.get("algo"),
            "seed": meta.get("seed"),
            "HMS": meta.get("HMS"),
            "HMCR": meta.get("HMCR"),
            "PAR": meta.get("PAR"),
            "BW": meta.get("BW"),
            "NI": meta.get("NI"),
            "objetivo": meta.get("objetivo"),
            "timestamp": meta.get("timestamp"),
            "trace_name": trace_path.name,
        },
        "meta_raw": meta,
        "trace_raw": trace,
        "best": {"best_hp": best_hp, "best_fitness": best_fitness, "eval_count_total": final.get("eval_count_total")},
        "kpis": {
            "best_final": best_fitness,
            "best_init": best_init,
            "delta_abs": best_fitness - best_init,
            "delta_pct": (best_fitness - best_init) / best_init if best_init > 0 else None,
            "worst_final": worst_final,
            "replacements": len(replacements),
            "replace_rate": len(replacements) / len(evals) if evals else 0.0,
            "NI": meta.get("NI"),
            "num_params": len(best_hp),
        },
        "series": {
            "iter_idx": iter_idx,
            "after_best": after_best,
            "after_worst": after_worst,
            "before_best": before_best,
            "before_worst": before_worst,
            "delta_best": delta_best,
            "eval_cum": eval_cum,
        },
        "evals": evals,
        "tables": {"evals": evals, "top": top, "worst": worst, "init_hm": init_hm},
        "distrib": {"fitness_all": fitness_all, "box_iter": box_iter(trace), "param_values": param_values},
        "diagnostics": {
            "hm_usage": {"HM": hm_usage_stats["HM"], "rand": hm_usage_stats["rand"]},
            "hm_iter": {"HM": hm_iter_hm, "rand": hm_iter_rand},
            "hm_param": {"HM": dict(hm_usage_stats["per_param_hm"]), "rand": dict(hm_usage_stats["per_param_rand"])} ,
            "pitch": pitch,
            "clamps": clamps,
            "clamp_param": dict(clamp_param),
            "replace_per_iter": replace_per_iter,
        },
        "insights": insights,
    }
    payload["tree_html"] = build_tree_html(trace)
    return payload

def main():
    parser = argparse.ArgumentParser(description="Genera un dashboard HTML a partir de un trace de Harmony Search.")
    parser.add_argument("--trace", type=Path, help="Ruta al trace_*.json (por defecto, ultimo en hs_runs)")
    parser.add_argument("--out", type=Path, help="Ruta de salida del HTML (por defecto, mismo nombre que el trace pero .html)")
    args = parser.parse_args()

    trace_path = args.trace if args.trace else find_latest_trace(Path("hs_runs"))
    if not trace_path.exists():
        raise FileNotFoundError(f"No se encuentra el trace: {trace_path}")

    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    payload = build_payload(trace, trace_path)
    out_path = args.out if args.out else trace_path.with_suffix(".html")

    html = HTML_TEMPLATE.safe_substitute(payload_json=json.dumps(payload, ensure_ascii=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Reporte generado en: {out_path}")


if __name__ == "__main__":
    main()
