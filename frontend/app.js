/* GempaWaspada frontend logic */
"use strict";

const SEV = {
  0: { label: "Ringan", cls: "sev-ringan", fill: "fill-ringan", tag: "tag--ringan" },
  1: { label: "Sedang", cls: "sev-sedang", fill: "fill-sedang", tag: "tag--sedang" },
  2: { label: "Berat", cls: "sev-berat", fill: "fill-berat", tag: "tag--berat" },
};

const $ = (id) => document.getElementById(id);

async function api(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status} untuk ${url}`);
  return res.json();
}

const fmtInt = new Intl.NumberFormat("id-ID");
const fmt1 = new Intl.NumberFormat("id-ID", { maximumFractionDigits: 1 });

/* ---------- init ---------- */
async function init() {
  try {
    const [summary, seismicity, trend, regions, quakes, confusion] = await Promise.all([
      api("/api/summary"),
      api("/api/seismicity"),
      api("/api/trend"),
      api("/api/regions"),
      api("/api/earthquakes"),
      api("/api/confusion"),
    ]);

    renderKpis(summary);
    renderSeismicity(seismicity);
    renderTrend(trend);
    renderRegions(regions);
    renderPerformance(summary, confusion);
    renderCatalog(quakes);
    wirePredictor();
  } catch (err) {
    showFatal(err);
  }
}

function showFatal(err) {
  const el = document.createElement("div");
  el.className = "error-state";
  el.textContent = `Gagal memuat data: ${err.message}. Pastikan backend berjalan (uvicorn backend.app:app).`;
  document.querySelector("main").prepend(el);
}

/* ---------- KPI ---------- */
function renderKpis(s) {
  $("kpi-usgs").textContent = fmtInt.format(s.usgs_n_events);
  $("kpi-gap").textContent = formatGapHours(s.usgs_median_gap_hours);
  $("kpi-events").textContent = fmtInt.format(s.n_events);
  $("kpi-berat").textContent = fmtInt.format(s.severity["Berat"]);
}

function formatGapHours(h) {
  if (h < 24) return `${Math.round(h)} jam`;
  return `${fmt1.format(h / 24)} hari`;
}

/* ---------- seismicity ---------- */
function renderSeismicity(s) {
  $("seis-desc").textContent = `${fmtInt.format(s.total_events)} gempa ${s.threshold} (USGS), 2000–2023.`;
  $("seis-gap").textContent = formatGapHours(s.median_gap_days * 24);

  const maxLambda = Math.max(...s.regions.map((r) => r.lambda), 1);
  const list = $("seis-list");
  list.innerHTML = "";
  s.regions.forEach((r) => {
    const li = document.createElement("li");
    li.className = "risk-item";
    li.innerHTML = `
      <div class="risk-item__top">
        <span class="risk-item__name">${escapeHtml(r.region)}</span>
        <span class="risk-item__rate">${fmt1.format(r.lambda)}/th</span>
      </div>
      <div class="risk-item__bar"><div class="risk-item__fill" style="width:${(r.lambda / maxLambda) * 100}%"></div></div>
      <span class="risk-item__meta">${fmtInt.format(r.events)} gempa · jeda median ${fmt1.format(r.median_gap_days)} hari</span>
    `;
    list.appendChild(li);
  });
}

/* ---------- trend (SVG bar chart) ---------- */
function renderTrend(t) {
  const svg = $("trend-chart");
  const w = 1000, h = 120, pad = 6;
  const max = Math.max(...t.year_counts, 1);
  const n = t.years.length;
  const bw = (w - pad * 2) / n;
  let bars = "";
  t.years.forEach((y, i) => {
    const bh = (t.year_counts[i] / max) * (h - pad * 2 - 14);
    const x = pad + i * bw + bw * 0.15;
    const by = h - pad - bh;
    bars += `<rect class="trend__bar" x="${x.toFixed(1)}" y="${by.toFixed(1)}" width="${(bw * 0.7).toFixed(1)}" height="${bh.toFixed(1)}" rx="2">
      <title>${y}: ${t.year_counts[i]}</title></rect>`;
  });
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = bars;
}

/* ---------- regions ---------- */
let regionCentroids = {};

function renderRegions(regions) {
  regionCentroids = {};
  const sel = $("region");
  sel.innerHTML = "";
  regions.forEach((r) => {
    regionCentroids[r.name] = { lat: r.lat, lon: r.lon };
    const opt = document.createElement("option");
    opt.value = r.name;
    opt.textContent = `${r.name} (${r.events})`;
    sel.appendChild(opt);
  });
  if (regions.length) {
    sel.value = regions[0].name;
    syncLocationDefaults();
  }
}

function syncLocationDefaults() {
  const c = regionCentroids[$("region").value];
  if (c) {
    $("latitude").value = c.lat;
    $("longitude").value = c.lon;
  }
}

/* ---------- performance ---------- */
function renderPerformance(s, confusion) {
  $("perf-macro").textContent = s.macro_f1.toFixed(3);
  $("perf-recall").textContent = `${Math.round(s.recall_berat * 100)}%`;
  $("perf-baseline").textContent = s.baseline_macro_f1.toFixed(3);

  const c = $("confusion");
  const labels = confusion.labels;
  let html = `<div class="confusion__cell confusion__cell--head"></div>`;
  labels.forEach((l) => { html += `<div class="confusion__cell confusion__cell--head">${l} (prediksi)</div>`; });
  confusion.matrix.forEach((row, i) => {
    html += `<div class="confusion__cell confusion__cell--rowhead">${labels[i]} (aktual)</div>`;
    row.forEach((v, j) => {
      const diag = i === j;
      html += `<div class="confusion__cell ${diag ? `confusion__cell--diag ${SEV[i].cls}` : ""}">${v}</div>`;
    });
  });
  c.innerHTML = html;

  const d = $("distribution");
  const total = Object.values(s.severity).reduce((a, b) => a + b, 0);
  let dist = "";
  [0, 1, 2].forEach((i) => {
    const name = SEV[i].label;
    const n = s.severity[name];
    const pct = Math.round((n / total) * 100);
    dist += `
      <div class="dist-item">
        <div class="dist-item__top"><span class="dist-item__name">${name}</span><span class="dist-item__count">${n} (${pct}%)</span></div>
        <div class="dist-item__bar"><div class="dist-item__fill ${SEV[i].fill}" style="width:${pct}%"></div></div>
      </div>`;
  });
  d.innerHTML = dist;
}

/* ---------- catalog ---------- */
let quakes = [];
let sortKey = "year";
let sortDir = -1;

function renderCatalog(data) {
  quakes = data;
  applyCatalog();
}

function applyCatalog() {
  const filter = $("severity-filter").value;
  const rows = quakes
    .filter((q) => filter === "" || q.severity === Number(filter))
    .sort((a, b) => {
      let av = a[sortKey], bv = b[sortKey];
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "string") { av = av.toLowerCase(); bv = bv.toLowerCase(); }
      return (av < bv ? -1 : av > bv ? 1 : 0) * sortDir;
    });

  const tbody = $("catalog-body");
  tbody.innerHTML = "";
  rows.forEach((q) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${q.year}</td>
      <td>${escapeHtml(q.wilayah)}</td>
      <td>${escapeHtml(q.region)}</td>
      <td class="num">${fmt1.format(q.magnitude)}</td>
      <td class="num">${fmt1.format(q.depth)}</td>
      <td class="num">${q.mmi ?? "—"}</td>
      <td>${q.tsunami ? '<span class="tag tag--tsunami">Ya</span>' : "—"}</td>
      <td><span class="tag ${SEV[q.severity].tag}">${SEV[q.severity].label}</span></td>
    `;
    tbody.appendChild(tr);
  });

  $("catalog-meta").textContent = `Menampilkan ${fmtInt.format(rows.length)} dari ${fmtInt.format(quakes.length)} kejadian`;
}

function wireCatalog() {
  document.querySelectorAll("#catalog-table th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (sortKey === key) sortDir *= -1; else { sortKey = key; sortDir = 1; }
      document.querySelectorAll("#catalog-table th").forEach((x) => x.removeAttribute("aria-sort"));
      th.setAttribute("aria-sort", sortDir === 1 ? "ascending" : "descending");
      applyCatalog();
    });
  });
  $("severity-filter").addEventListener("change", applyCatalog);
}

/* ---------- predictor ---------- */
let predictSeq = 0;

function wirePredictor() {
  wireCatalog();
  const inputs = ["magnitude", "depth", "mmi", "tsunami", "region", "latitude", "longitude"];
  inputs.forEach((id) => {
    const el = $(id);
    el.addEventListener("input", () => {
      if (id === "magnitude") $("magnitude-out").value = el.value;
      if (id === "depth") $("depth-out").value = el.value;
      if (id === "mmi") $("mmi-out").value = el.value;
      if (id === "region") syncLocationDefaults();
      predict();
    });
    el.addEventListener("change", () => {
      if (id === "region") syncLocationDefaults();
      predict();
    });
  });
  predict();
}

async function predict() {
  const seq = ++predictSeq;
  const lat = $("latitude").value;
  const lon = $("longitude").value;
  const body = {
    magnitude: parseFloat($("magnitude").value),
    depth_km: parseFloat($("depth").value),
    mmi_max: parseFloat($("mmi").value),
    tsunami: $("tsunami").checked,
    region: $("region").value,
    latitude: lat === "" ? null : parseFloat(lat),
    longitude: lon === "" ? null : parseFloat(lon),
  };
  try {
    const r = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (seq !== predictSeq) return;
    renderResult(r);
  } catch (err) {
    $("result-empty").textContent = `Gagal prediksi: ${err.message}`;
    $("result-body").hidden = true;
    $("result-empty").hidden = false;
  }
}

function renderResult(r) {
  $("result-empty").hidden = true;
  const body = $("result-body");
  body.hidden = false;
  const val = $("result-value");
  val.textContent = r.label;
  val.className = "result__value " + SEV[r.class].cls;

  const bars = $("result-bars");
  bars.innerHTML = "";
  ["Ringan", "Sedang", "Berat"].forEach((name) => {
    const i = Object.keys(SEV).find((k) => SEV[k].label === name);
    const p = r.probabilities[name];
    const div = document.createElement("div");
    div.className = "pbar";
    div.innerHTML = `
      <span class="pbar__name">${name}</span>
      <span class="pbar__track"><span class="pbar__fill ${SEV[i].fill}" style="width:${(p * 100).toFixed(0)}%"></span></span>
      <span class="pbar__val">${Math.round(p * 100)}%</span>`;
    bars.appendChild(div);
  });

  const e = r.exposure;
  $("result-exposure").innerHTML = e
    ? `<span class="exposure__item">Jarak permukiman terdekat <strong>${fmt1.format(e.dist_nearest_settlement_km)} km</strong></span>` +
      `<span class="exposure__item">Permukiman 50 km <strong>${fmtInt.format(e.n_settlements_50km)}</strong></span>`
    : "";
}

/* ---------- util ---------- */
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

init();
