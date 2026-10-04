// Nablaframe viewer (Phase 2, D13). Contract for the implementing thread:
//
// * Served by `python3 -m http.server` from the repo root. Every path it fetches is
//   relative; it contacts no other host, and loads no library from outside the repo.
// * Derived views: it reads runs/phase2/views/index.json, written last by
//   src/nablaframe/views/build.py, then each payload file the index names. The payload
//   shapes are in build.py's docstring. Every payload carries a "meter", and the page shows
//   it beside its view; logit_lens and attribution also carry a "badge" (D14), shown with it.
// * Readouts come straight from the trace store: it parses events.jsonl, and for each
//   tensor it shows, fetches the shard and slices the "tensor" entry's bytes exactly as
//   src/nablaframe/views/shards.py does. Before displaying a tensor, it hashes that
//   slice with crypto.subtle.digest("SHA-256") and compares it with the event's
//   sha256. A mismatch is shown, never hidden. That check is a readout's meter.
// * The timeline lists the index's steps on its val_loss_curve; every panel follows it.
//   The probe text is rendered with the index's vocab. Each load shows progress.
//
// How this page meets it:
//
// * Paths, relative to viewer/: the views at ../runs/phase2/views/ (G2.6's second build)
//   and the trace store at ../runs/phase1/g16_trace/store/ (G1.6's, per
//   gates/phase2/_common.py). `?views=<dir>&trace=<dir>` points either elsewhere, as a
//   relative path only; anything resolving to another origin is refused.
// * Trace events: the current ones (not superseded) that carry a tensor, matched by step,
//   tap and the batch hash of the index's probe (SHA-256 of its int32 little-endian
//   bytes, as data/corpus.py's batch_hash). None, or more than one, is shown as an error.
// * Readouts shown from shards, each hash-checked before it is drawn: the attention
//   pattern of the chosen layer, every residual stream tap behind the norm panel, and the
//   logits behind the per-token loss (hashed, not drawn; the loss values come from the
//   per_token_loss payload, whose meter names the logits hash it was built from, and the
//   page checks that name against the trace too). A mismatch outlines the panel in red,
//   says so in its meter and in the header's count, and the values stay on screen,
//   marked as not matching the trace. Without Web Crypto (a page not opened through
//   localhost), every hash check reads "not checked", never "matches".
// * Derived payloads without a "meter" are not rendered.
// * State lives in the address's #fragment (step, sequence, position, layer, head), so a
//   view can be reopened as it was.
"use strict";

(() => {
  // ---------------------------------------------------------------- basics

  const $ = (id) => document.getElementById(id);
  // The SVG namespace, taken from the parser rather than written out here.
  const SVG_NS = (() => {
    const holder = document.createElement("div");
    holder.innerHTML = "<svg></svg>";
    return holder.firstChild.namespaceURI;
  })();

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    setAttrs(node, attrs);
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }

  function svg(tag, attrs, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    setAttrs(node, attrs);
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }

  function setAttrs(node, attrs) {
    if (!attrs) return;
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.setAttribute("class", value);
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
      else node.setAttribute(key, value === true ? "" : value);
    }
  }

  function clear(node, ...children) {
    node.replaceChildren(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false));
  }

  const fmtBytes = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : n >= 1024 ? `${(n / 1024).toFixed(1)} kB` : `${n} B`);
  const short = (hash) => (typeof hash === "string" ? `${hash.slice(0, 12)}…` : String(hash));

  function fmt(x, digits = 3) {
    if (x === null || x === undefined) return "blank";
    if (!Number.isFinite(x)) return String(x);
    const a = Math.abs(x);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(digits - 1);
    return x.toFixed(digits);
  }

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const isDark = () => cssVar("color-scheme") === "dark";

  const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

  // ---------------------------------------------------------------- paths

  function dirParam(name, fallback) {
    const value = new URLSearchParams(location.search).get(name) || fallback;
    return value.endsWith("/") ? value : `${value}/`;
  }

  const VIEWS_DIR = dirParam("views", "../runs/phase2/views/");
  const TRACE_DIR = dirParam("trace", "../runs/phase1/g16_trace/store/");

  // Resolves a relative path against the page; refuses anything that leaves its origin.
  function localUrl(path) {
    if (/^[a-z][a-z0-9+.-]*:|^[\\/]{2}/i.test(path)) throw new Error(`refusing a non-relative path: ${path}`);
    const url = new URL(path, location.href);
    if (url.origin !== location.origin) throw new Error(`refusing a path outside this server: ${path}`);
    return url.href;
  }

  // ---------------------------------------------------------------- loads, with progress

  const loadsBox = $("loads");

  function startLoad(label) {
    const name = el("span", { text: label });
    const amount = el("span", { class: "muted", text: "starting" });
    const bar = el("progress", { max: "1" });
    const row = el("div", { class: "load" }, el("div", { class: "name" }, name, amount), bar);
    loadsBox.append(row);
    return {
      progress(done, total) {
        if (total > 0) {
          bar.max = total;
          bar.value = Math.min(done, total);
          amount.textContent = `${fmtBytes(done)} of ${fmtBytes(total)}`;
        } else {
          bar.removeAttribute("value");
          amount.textContent = fmtBytes(done);
        }
      },
      stage(text) {
        amount.textContent = text;
      },
      done() {
        bar.max = 1;
        bar.value = 1;
        amount.textContent = "done";
        setTimeout(() => row.remove(), 900);
      },
      fail(error) {
        row.classList.add("failed");
        amount.textContent = "failed";
        bar.remove();
        row.append(el("div", { text: String(error.message || error) }),
          el("button", { type: "button", text: "dismiss", onclick: () => row.remove() }));
      },
    };
  }

  async function fetchBytes(path, label, load = startLoad(label)) {
    try {
      const response = await fetch(localUrl(path), { cache: "no-cache" });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${path}`);
      const total = Number(response.headers.get("Content-Length")) || 0;
      let bytes;
      if (!response.body) {
        bytes = new Uint8Array(await response.arrayBuffer());
      } else {
        const reader = response.body.getReader();
        const chunks = [];
        let received = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.length;
          load.progress(received, total);
        }
        bytes = new Uint8Array(received);
        let offset = 0;
        for (const chunk of chunks) {
          bytes.set(chunk, offset);
          offset += chunk.length;
        }
      }
      return { bytes, load };
    } catch (error) {
      load.fail(error);
      throw error;
    }
  }

  async function fetchText(path, label) {
    const { bytes, load } = await fetchBytes(path, label);
    load.done();
    return new TextDecoder().decode(bytes);
  }

  async function fetchJSON(path, label) {
    const text = await fetchText(path, label);
    try {
      return JSON.parse(text);
    } catch (error) {
      throw new Error(`${path} is not valid JSON: ${error.message}`);
    }
  }

  // ---------------------------------------------------------------- hashing

  const HASH_AVAILABLE = !!(globalThis.crypto && crypto.subtle && crypto.subtle.digest);

  async function sha256Hex(bytes) {
    if (!HASH_AVAILABLE) return null;
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  const tally = { ok: 0, bad: 0, unchecked: 0 };

  // ---------------------------------------------------------------- safetensors

  // The "tensor" entry's bytes, sliced as src/nablaframe/views/shards.py slices them.
  function shardTensor(bytes) {
    if (bytes.length < 8) throw new Error(`${bytes.length} bytes is too short for a safetensors header length`);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const headerLength = view.getUint32(0, true) + view.getUint32(4, true) * 2 ** 32;
    const dataStart = 8 + headerLength;
    if (dataStart > bytes.length) throw new Error(`header length ${headerLength} runs past the end of the file`);
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, dataStart)));
    const entry = header && header.tensor;
    if (!entry || typeof entry !== "object") throw new Error('the header has no "tensor" entry');
    const [begin, end] = entry.data_offsets;
    if (!(begin >= 0 && begin <= end && end <= bytes.length - dataStart)) {
      throw new Error(`data_offsets [${begin}, ${end}] fall outside the ${bytes.length - dataStart} data bytes`);
    }
    return { dtype: entry.dtype, shape: entry.shape, data: bytes.subarray(dataStart + begin, dataStart + end) };
  }

  const DTYPES = {
    F64: [8, Float64Array, "getFloat64"],
    F32: [4, Float32Array, "getFloat32"],
    I32: [4, Int32Array, "getInt32"],
    I16: [2, Int16Array, "getInt16"],
    I8: [1, Int8Array, "getInt8"],
    U8: [1, Uint8Array, "getUint8"],
  };

  function decode(dtype, shape, data) {
    const spec = DTYPES[dtype];
    if (!spec) throw new Error(`dtype ${dtype} isn't one the viewer decodes`);
    const [size, Typed, getter] = spec;
    const count = shape.reduce((a, b) => a * b, 1);
    if (count * size !== data.length) throw new Error(`shape [${shape}] of ${dtype} needs ${count * size} bytes, the slice has ${data.length}`);
    if (LITTLE_ENDIAN || size === 1) return new Typed(data.slice().buffer);
    const out = new Typed(count);
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    for (let i = 0; i < count; i++) out[i] = view[getter](i * size, true);
    return out;
  }

  // ---------------------------------------------------------------- the trace

  const trace = { byKey: new Map(), runIds: new Set(), count: 0, current: 0 };
  const keyOf = (step, tap) => `${step}\u0000${tap}`;

  function indexEvents(text) {
    const events = [];
    text.split("\n").forEach((line, i) => {
      if (!line.trim()) return;
      try {
        events.push(JSON.parse(line));
      } catch (error) {
        throw new Error(`events.jsonl line ${i + 1} is not JSON: ${error.message}`);
      }
    });
    const superseded = new Set(events.filter((e) => e.supersedes).map((e) => e.supersedes));
    trace.count = events.length;
    for (const event of events) {
      if (superseded.has(event.event_id) || !event.tensor_ref) continue;
      trace.current += 1;
      trace.runIds.add(event.run_id);
      const key = keyOf(event.step, event.tap);
      if (!trace.byKey.has(key)) trace.byKey.set(key, []);
      trace.byKey.get(key).push(event);
    }
  }

  // {event} or {error}: the one current event for this step and tap on the probe's batch.
  function findEvent(step, tap) {
    const all = trace.byKey.get(keyOf(step, tap)) || [];
    if (!all.length) return { error: `the trace has no current event for ${tap} at step ${step}` };
    const onProbe = all.filter((e) => e.batch_hash === state.probeHash);
    if (!onProbe.length) {
      return { error: `the trace's events for ${tap} at step ${step} were recorded on batch ${short(all[0].batch_hash)}, not the index's probe (${short(state.probeHash)})` };
    }
    if (onProbe.length > 1) return { error: `the trace has ${onProbe.length} current events for ${tap} at step ${step}; it should have one` };
    return { event: onProbe[0] };
  }

  // A loaded readout: {event, status: "ok" | "bad" | "unchecked", computed, shape, dtype, values}.
  const tensorCache = new Map();
  const TENSOR_CACHE_LIMIT = 40;

  function loadTensor(event, { keepValues = true } = {}) {
    const key = `${event.event_id}:${keepValues}`;
    if (tensorCache.has(key)) {
      const hit = tensorCache.get(key);
      tensorCache.delete(key);
      tensorCache.set(key, hit);
      return hit;
    }
    const promise = (async () => {
      const label = `${event.tap} · step ${event.step}`;
      const { bytes, load } = await fetchBytes(TRACE_DIR + event.tensor_ref, label);
      try {
        const { dtype, shape, data } = shardTensor(bytes);
        load.stage("hashing");
        const computed = await sha256Hex(data);
        const status = computed === null ? "unchecked" : computed === event.sha256 ? "ok" : "bad";
        tally[status] += 1;
        renderFacts();
        const values = keepValues ? decode(dtype, shape, data) : null;
        load.done();
        return { event, status, computed, dtype, shape, values };
      } catch (error) {
        load.fail(error);
        throw new Error(`${event.tensor_ref}: ${error.message}`);
      }
    })();
    promise.catch(() => tensorCache.delete(key));
    tensorCache.set(key, promise);
    while (tensorCache.size > TENSOR_CACHE_LIMIT) tensorCache.delete(tensorCache.keys().next().value);
    return promise;
  }

  // ---------------------------------------------------------------- meters

  const ICONS = { ok: "✓", bad: "✗", warn: "?", info: "·" };

  function meterRow(kind, label, detail, title) {
    return el("div", { class: `row ${kind}`, title },
      el("span", { class: "icon", "aria-hidden": "true", text: ICONS[kind] }),
      el("span", { class: "label", text: label }),
      detail ? el("span", { class: "detail", text: detail }) : null);
  }

  function hashRow(loaded, what) {
    const { event, status, computed } = loaded;
    const title = `event ${event.event_id}\ntrace sha256 ${event.sha256}\nshard sha256 ${computed ?? "not computed"}`;
    if (status === "ok") return meterRow("ok", "hash check passed", `${what}: SHA-256 of the shard's bytes equals the trace's ${short(event.sha256)}`, title);
    if (status === "bad") {
      return meterRow("bad", "HASH MISMATCH", `${what}: the shard hashes to ${short(computed)}, the trace records ${short(event.sha256)}. The values shown are not the traced tensor.`, title);
    }
    return meterRow("warn", "hash not checked", `${what}: this browser has no Web Crypto here; open the page through localhost to check it.`, title);
  }

  function setPanelBad(panelId, bad) {
    $(panelId).classList.toggle("bad", !!bad);
  }

  function badge(text) {
    return el("div", { class: "badge" }, el("b", { text: "Badge (D14): " }), text);
  }

  function errorBox(message) {
    return el("div", { class: "error", text: message });
  }

  // ---------------------------------------------------------------- colors

  function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function ramp(stops) {
    const rgb = stops.map(hexToRgb);
    return (t) => {
      if (!Number.isFinite(t)) t = 0;
      t = Math.min(1, Math.max(0, t)) * (rgb.length - 1);
      const i = Math.min(rgb.length - 2, Math.floor(t));
      const f = t - i;
      return rgb[i].map((c, k) => Math.round(c + (rgb[i + 1][k] - c) * f));
    };
  }

  // Sequential ramps, one hue each, receding toward the surface at zero.
  const RAMPS = {
    blue: { light: ramp(["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
            dark: ramp(["#1d2230", "#104281", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"]) },
    orange: { light: ramp(["#fdf3ee", "#f9cdb7", "#f39a70", "#eb6834", "#b8481e", "#7a2a0c"]),
              dark: ramp(["#2a201c", "#6b2d14", "#a8461f", "#d95926", "#f0946b", "#fbd0bd"]) },
  };
  const rampFor = (name) => RAMPS[name][isDark() ? "dark" : "light"];
  const rgbCss = ([r, g, b]) => `rgb(${r},${g},${b})`;
  // Text over a ramp cell: dark ink on light cells, light ink on dark ones.
  const inkFor = ([r, g, b]) => (0.2126 * r + 0.7152 * g + 0.0722 * b > 140 ? "#0b0b0b" : "#ffffff");
  const SERIES = () => [1, 2, 3, 4, 5, 6, 7, 8].map((i) => cssVar(`--series-${i}`));

  function rampLegend(name, lo, hi, unit) {
    const f = rampFor(name);
    const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => `${rgbCss(f(t))} ${t * 100}%`).join(",");
    return el("span", {}, `${lo}`, el("span", { class: "ramp", style: { background: `linear-gradient(90deg,${stops})` } }), `${hi} ${unit}`);
  }

  // ---------------------------------------------------------------- tooltip

  const tooltip = $("tooltip");
  function showTip(event, lines) {
    clear(tooltip, lines.map((line, i) => (i ? [el("br"), line] : line)));
    tooltip.style.display = "block";
    const pad = 14;
    const { innerWidth: w, innerHeight: h } = window;
    const box = tooltip.getBoundingClientRect();
    tooltip.style.left = `${Math.min(event.clientX + pad, w - box.width - 4)}px`;
    tooltip.style.top = `${Math.min(event.clientY + pad, h - box.height - 4)}px`;
  }
  const hideTip = () => { tooltip.style.display = "none"; };

  // ---------------------------------------------------------------- state

  const state = {
    index: null,
    probeHash: null,
    stepIndex: 0,
    seq: 0,
    pos: 0,
    layer: 0,
    head: 0,
    nHeads: null,
    payloads: new Map(), // `${view}:${step}` -> Promise<payload>
    layerStructure: new Map(), // step -> payload
    layerStructureErrors: [],
    layerStructureDone: false,
  };

  const step = () => state.index.steps[state.stepIndex];

  function tokenText(id) {
    const vocab = state.index.vocab;
    return vocab && id >= 0 && id < vocab.length ? vocab[id] : `#${id}`;
  }
  const probeRow = () => state.index.probe[state.seq];

  function readHash() {
    const values = new URLSearchParams(location.hash.slice(1));
    const int = (name) => (values.has(name) && /^-?\d+$/.test(values.get(name)) ? Number(values.get(name)) : null);
    return { step: int("step"), seq: int("seq"), pos: int("pos"), layer: int("layer"), head: int("head") };
  }

  function writeHash() {
    const values = new URLSearchParams({ step: step(), seq: state.seq, pos: state.pos, layer: state.layer, head: state.head });
    history.replaceState(null, "", `${location.pathname}${location.search}#${values}`);
  }

  function payload(view, atStep) {
    const key = `${view}:${atStep}`;
    if (!state.payloads.has(key)) {
      const file = state.index.files?.[view]?.[String(atStep)];
      const promise = file
        ? fetchJSON(VIEWS_DIR + file, `${view} · step ${atStep}`).then((data) => checkPayload(view, atStep, data))
        : Promise.reject(new Error(`the index names no ${view} file for step ${atStep}`));
      promise.catch(() => state.payloads.delete(key));
      state.payloads.set(key, promise);
      if (state.payloads.size > 48) state.payloads.delete(state.payloads.keys().next().value);
    }
    return state.payloads.get(key);
  }

  // A derived view without its meter can't render.
  function checkPayload(view, atStep, data) {
    if (!data || typeof data !== "object" || !("meter" in data)) throw new Error(`the ${view} view for step ${atStep} has no meter, so it isn't shown`);
    if (data.step !== atStep) throw new Error(`the ${view} file the index names for step ${atStep} holds step ${data.step}`);
    return data;
  }

  // ---------------------------------------------------------------- header

  function renderFacts() {
    const facts = $("facts");
    if (!state.index) return;
    const idx = state.index;
    const traceIds = [...trace.runIds];
    const runMatches = traceIds.length === 1 && (traceIds[0].startsWith(idx.run_id) || idx.run_id.startsWith(traceIds[0]));
    const parts = [
      el("span", {}, "run ", el("code", { text: short(idx.run_id) })),
      el("span", {}, `${idx.steps.length} checkpoints · ${idx.n_layers} layers · probe ${idx.probe.length} × ${idx.probe[0]?.length ?? 0}`),
      el("span", {}, `trace: ${trace.current} current tensor events of ${trace.count}`),
    ];
    const batches = new Set([...trace.byKey.values()].flat().map((e) => e.batch_hash));
    if (!batches.has(state.probeHash)) {
      parts.push(el("span", { class: "error", text: `no trace event was recorded on the index's probe (batch ${short(state.probeHash)})` }));
    }
    if (!runMatches) {
      parts.push(el("span", { class: "error", text: `trace run_id ${traceIds.map(short).join(", ") || "none"} doesn't match the index's ${short(idx.run_id)}` }));
    }
    const checks = el("span", {}, "hash checks: ",
      el("span", { style: { color: "var(--good-text)", fontWeight: 600 }, text: `${tally.ok} ✓` }), " · ",
      el("span", { class: tally.bad ? "error" : "muted", text: `${tally.bad} ✗ mismatched` }));
    if (tally.unchecked) checks.append(" · ", el("span", { style: { color: "var(--warning-text)" }, text: `${tally.unchecked} not checked (no Web Crypto)` }));
    parts.push(checks);
    clear(facts, parts);
  }

  // ---------------------------------------------------------------- timeline

  const xScaleLog = (maxStep, x0, x1) => (s) => x0 + (x1 - x0) * (Math.log2(s + 1) / Math.log2(maxStep + 1 || 2));

  // 0, powers of 4, and the last step, dropping any power too close to the last to label.
  function logTicks(maxStep, minGap = 0.08) {
    const span = Math.log2(maxStep + 1) || 1;
    const ticks = [0];
    for (let s = 1; s < maxStep; s *= 4) {
      if ((Math.log2(maxStep + 1) - Math.log2(s + 1)) / span >= minGap) ticks.push(s);
    }
    if (ticks[ticks.length - 1] !== maxStep) ticks.push(maxStep);
    return ticks;
  }

  function niceTicks(lo, hi, count = 4) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const stepSize = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => m >= raw);
    const ticks = [];
    for (let t = Math.ceil(lo / stepSize) * stepSize; t <= hi + 1e-12; t += stepSize) ticks.push(+t.toPrecision(12));
    return ticks;
  }

  function renderTimeline() {
    const box = $("timeline");
    const idx = state.index;
    const steps = idx.steps;
    const width = Math.max(320, box.clientWidth || 800);
    const height = 170;
    const m = { l: 52, r: 24, t: 24, b: 34 };
    const maxStep = Math.max(...steps, ...(idx.val_loss_curve || []).map((p) => p.step));
    const x = xScaleLog(maxStep, m.l, width - m.r);
    const curve = (idx.val_loss_curve || []).filter((p) => Number.isFinite(p.step)).sort((a, b) => a.step - b.step);
    const losses = curve.map((p) => p.loss).filter((v) => v !== null && Number.isFinite(v));
    const lo = losses.length ? Math.min(...losses) : 0;
    const hi = losses.length ? Math.max(...losses) : 1;
    const pad = (hi - lo) * 0.08 || 0.5;
    const y = (v) => m.t + (height - m.t - m.b) * (1 - (v - (lo - pad)) / (hi - lo + 2 * pad));
    const lossAt = new Map(curve.map((p) => [p.step, p.loss]));

    const root = svg("svg", { width, height, role: "img", "aria-label": "Validation loss over training, with the trace checkpoints marked" });
    for (const t of niceTicks(lo - pad, hi + pad)) {
      root.append(svg("line", { class: "gridline", x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }),
        svg("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, fmt(t, 2)));
    }
    root.append(svg("line", { class: "axis", x1: m.l, x2: width - m.r, y1: height - m.b, y2: height - m.b }));
    for (const t of logTicks(maxStep)) {
      root.append(svg("text", { x: x(t), y: height - m.b + 14, "text-anchor": "middle" }, String(t)));
    }
    root.append(svg("text", { x: width - m.r, y: height - 4, "text-anchor": "end" }, "training step (log scale)"),
      svg("text", { x: 4, y: 12 }, losses.length ? "validation loss (nats)" : ""));

    if (curve.length) {
      let d = "";
      let pen = false;
      for (const p of curve) {
        if (p.loss === null || !Number.isFinite(p.loss)) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`;
        pen = true;
      }
      root.append(svg("path", { d, fill: "none", stroke: cssVar("--series-1"), "stroke-width": 2, "stroke-linejoin": "round" }));
    }

    const current = step();
    const cx = x(current);
    root.append(svg("line", { x1: cx, x2: cx, y1: m.t, y2: height - m.b, stroke: cssVar("--text-secondary"), "stroke-dasharray": "3 3" }));
    steps.forEach((s, i) => {
      const v = lossAt.get(s);
      const cy = v !== undefined && v !== null && Number.isFinite(v) ? y(v) : height - m.b;
      const isCurrent = i === state.stepIndex;
      const label = `step ${s}${v !== undefined && v !== null ? ` · val loss ${fmt(v)}` : " · no val loss recorded"}`;
      const g = svg("g", { style: "cursor:pointer", onclick: () => selectStep(i) },
        svg("circle", { cx: x(s), cy, r: 10, fill: "transparent" }),
        svg("circle", {
          cx: x(s), cy, r: isCurrent ? 6 : 4,
          fill: isCurrent ? cssVar("--accent") : cssVar("--surface-1"),
          stroke: isCurrent ? cssVar("--surface-1") : cssVar("--accent"), "stroke-width": 2,
        }),
        svg("title", {}, label));
      root.append(g);
    });
    clear(box, root);
    if (!curve.length) box.append(el("div", { class: "note", text: "The index has no validation-loss curve; checkpoints are shown on the step axis." }));
    const v = lossAt.get(current);
    $("step-label").textContent = `step ${current} (checkpoint ${state.stepIndex + 1} of ${steps.length})${v !== undefined && v !== null ? ` · val loss ${fmt(v)}` : ""}`;
    $("step-slider").value = String(state.stepIndex);
  }

  // ---------------------------------------------------------------- probe text and per-token loss

  async function renderProbe() {
    const atStep = step();
    const seq = state.seq;
    const meter = $("loss-meter");
    const text = $("probe-text");
    clear($("seq-buttons"), el("span", { text: "Sequence" }), state.index.probe.map((_, b) => el("button", {
      type: "button", "aria-pressed": String(b === seq), text: String(b), onclick: () => { state.seq = b; changed({ seqChanged: true }); },
    })));
    let data;
    try {
      data = await payload("per_token_loss", atStep);
    } catch (error) {
      if (atStep !== step()) return;
      clear(meter, meterRow("bad", "per-token loss unavailable", error.message));
      drawProbe(null);
      return;
    }
    if (atStep !== step() || seq !== state.seq) return;
    drawProbe(data.loss[seq]);
    renderLossMeter(data, atStep);
  }

  function drawProbe(losses) {
    const tokens = probeRow();
    const vocabSize = state.index.vocab ? state.index.vocab.length : null;
    const finite = losses ? losses.filter((v) => v !== null) : [];
    const top = vocabSize ? Math.log(vocabSize) : Math.max(1, ...finite);
    const color = rampFor("orange");
    const spans = tokens.map((id, t) => {
      const loss = losses ? losses[t] : undefined;
      const word = tokenText(id);
      const span = el("span", { class: "tok", "data-pos": t, text: word });
      if (word.startsWith("<") && word.endsWith(">")) span.classList.add("special");
      if (loss === null) span.classList.add("blank");
      else if (loss !== undefined) {
        const rgb = color(loss / top);
        span.style.background = rgbCss(rgb);
        span.style.color = inkFor(rgb);
      }
      if (t === state.pos) span.classList.add("sel");
      span.addEventListener("click", () => { state.pos = t; changed({ posChanged: true }); });
      span.addEventListener("mousemove", (event) => showTip(event, [
        `position ${t} · “${word}” (id ${id})`,
        loss === null ? "loss: blank, no next token inside the window" : loss === undefined ? "loss not loaded" : `loss predicting the next token: ${fmt(loss)} nats`,
        t + 1 < tokens.length ? `next token: “${tokenText(tokens[t + 1])}”` : "",
      ].filter(Boolean)));
      span.addEventListener("mouseleave", hideTip);
      return span;
    });
    clear($("probe-text"), spans);
    clear($("loss-legend"),
      el("span", {}, "loss on the next token: ", rampLegend("orange", 0, vocabSize ? `ln ${vocabSize} = ${fmt(top, 2)}` : fmt(top, 2), "nats")),
      el("span", {}, el("span", { class: "sw", style: { backgroundImage: "repeating-linear-gradient(135deg, var(--border) 0 2px, transparent 2px 6px)" } }), "blank: no next token in the window"),
      el("span", { text: "Click a token to choose its position for the lens and the attribution." }));
  }

  async function renderLossMeter(data, atStep) {
    const meter = $("loss-meter");
    const rows = [];
    const found = findEvent(atStep, "logits");
    const m = data.meter || {};
    if (m.kind !== "hash_check") rows.push(meterRow("bad", "unexpected meter", `kind ${JSON.stringify(m.kind)}, expected hash_check`));
    let bad = false;
    if (found.error) {
      rows.push(meterRow("bad", "logits not in the trace", found.error));
      bad = true;
    } else if (found.event.sha256 === m.tensor_sha256) {
      rows.push(meterRow("ok", "built from the traced logits", `the view names logits ${short(m.tensor_sha256)}, the trace's current logits event for step ${atStep}`));
    } else {
      rows.push(meterRow("bad", "BUILT FROM OTHER LOGITS", `the view names ${short(m.tensor_sha256)}, the trace's current logits event is ${short(found.event.sha256)}. Rebuild the views.`));
      bad = true;
    }
    clear(meter, rows, found.error ? null : meterRow("info", "hashing the logits shard…", ""));
    setPanelBad("probe-panel", bad);
    if (found.error) return;
    try {
      const loaded = await loadTensor(found.event, { keepValues: false });
      if (atStep !== step()) return;
      clear(meter, rows, hashRow(loaded, "logits shard"));
      setPanelBad("probe-panel", bad || loaded.status === "bad");
    } catch (error) {
      if (atStep !== step()) return;
      clear(meter, rows, meterRow("bad", "logits shard unreadable", error.message));
      setPanelBad("probe-panel", true);
    }
  }

  // ---------------------------------------------------------------- logit lens

  async function renderLens() {
    const atStep = step();
    const { seq, pos } = state;
    const body = $("lens-body");
    $("lens-sub").textContent = `sequence ${seq}, position ${pos}`;
    let data;
    try {
      data = await payload("logit_lens", atStep);
    } catch (error) {
      if (atStep !== step()) return;
      clear($("lens-meter"), meterRow("bad", "logit lens unavailable", error.message));
      clear($("lens-badge"));
      clear(body);
      return;
    }
    if (atStep !== step() || seq !== state.seq || pos !== state.pos) return;
    const meter = Array.isArray(data.meter) ? data.meter : [];
    const last = meter[meter.length - 1];
    clear($("lens-meter"),
      meterRow("info", "meter", "per reading, over every probe position: KL from the model's own prediction (nats) and top-1 agreement with it; columns at right"),
      last ? meterRow("info", "last reading", `KL ${fmt(last.kl_mean)} nats · top-1 agreement ${(100 * last.top1_agreement).toFixed(1)}%. The last reading is the model's own computation.`) : meterRow("bad", "no per-layer meter in the view", ""));
    clear($("lens-badge"), data.badge ? badge(data.badge) : errorBox("This view carries no badge."));

    const tokens = probeRow();
    const target = pos + 1 < tokens.length ? tokens[pos + 1] : null;
    const color = rampFor("blue");
    const head = el("tr", {}, el("th", { text: "reading" }),
      Array.from({ length: data.top_tokens[0]?.[0]?.[0]?.length ?? 0 }, (_, k) => el("th", { text: `#${k + 1}` })),
      el("th", { class: "num", text: "KL (nats)" }), el("th", { class: "num", text: "top-1" }));
    const rows = data.layers.map((layerName, l) => {
      const ids = data.top_tokens[l][seq][pos];
      const probs = data.top_probs[l][seq][pos];
      return el("tr", {},
        el("td", { class: "mono", title: layerName, text: lensLabel(layerName, l) }),
        ids.map((id, k) => {
          const rgb = color(probs[k]);
          return el("td", {
            class: `cell${id === target ? " target" : ""}`,
            style: { background: rgbCss(rgb), color: inkFor(rgb) },
            title: `token “${tokenText(id)}” (id ${id}) · p = ${probs[k]}${id === target ? " · the actual next token" : ""}`,
          }, `${tokenText(id)} `, el("span", { style: { opacity: 0.8 }, text: (probs[k] * 100).toFixed(1) + "%" }));
        }),
        el("td", { class: "num", text: meter[l] ? fmt(meter[l].kl_mean) : "?" }),
        el("td", { class: "num", text: meter[l] ? `${(100 * meter[l].top1_agreement).toFixed(1)}%` : "?" }));
    });
    clear(body,
      el("div", { class: "note" }, `Top tokens if the model stopped at each residual stream, at “${tokenText(tokens[pos])}”. `,
        target === null ? "This is the last position, so there's no next token to outline." : `The actual next token, “${tokenText(target)}”, is outlined where it appears.`),
      el("div", { class: "scrollx" }, el("table", {}, el("thead", {}, head), el("tbody", {}, rows))),
      el("div", { class: "legend" }, el("span", {}, "probability ", rampLegend("blue", "0", "1", ""))));
  }

  function lensLabel(name, l) {
    const match = /^blocks\.(\d+)\.hook_resid_(pre|post)$/.exec(name);
    if (!match) return name;
    return match[2] === "pre" && match[1] === "0" ? "embeddings" : `after block ${match[1]}`;
  }

  // ---------------------------------------------------------------- attribution

  async function renderAttribution() {
    const atStep = step();
    const { seq, pos } = state;
    const body = $("attr-body");
    let data;
    try {
      data = await payload("attribution", atStep);
    } catch (error) {
      if (atStep !== step()) return;
      clear($("attr-meter"), meterRow("bad", "attribution unavailable", error.message));
      clear($("attr-badge"));
      clear(body);
      return;
    }
    if (atStep !== step() || seq !== state.seq || pos !== state.pos) return;
    const tokens = probeRow();
    const target = pos + 1 < tokens.length ? tokens[pos + 1] : null;
    $("attr-sub").textContent = target === null ? `sequence ${seq}, position ${pos}` : `sequence ${seq}, position ${pos}, on the logit of “${tokenText(target)}”`;
    const m = data.meter || {};
    clear($("attr-meter"), m.kind === "decomposition_error"
      ? meterRow("info", "decomposition error", `largest gap over the probe between the summed parts and the traced logit of the next token: ${fmt(m.max_abs)}`)
      : meterRow("bad", "unexpected meter", JSON.stringify(m)));
    clear($("attr-badge"), data.badge ? badge(data.badge) : errorBox("This view carries no badge."));

    const values = data.parts.map((_, p) => data.contributions[p][seq][pos]);
    if (values.every((v) => v === null)) {
      clear(body, el("div", { class: "note", text: "Blank: the last position has no next token, so there's no logit to attribute." }));
      return;
    }
    const sum = values.reduce((a, v) => a + (v ?? 0), 0);
    const extent = Math.max(1e-12, ...values.map((v) => Math.abs(v ?? 0)));
    const width = Math.max(300, (body.clientWidth || 520));
    const labelW = Math.min(190, width * 0.38);
    const valueW = 64;
    const barArea = width - labelW - valueW;
    const zero = labelW + barArea / 2;
    const rowH = 22;
    const height = rowH * values.length + 26;
    const root = svg("svg", { width, height, role: "img", "aria-label": "Each part's contribution to the next token's logit" });
    root.append(svg("line", { class: "axis", x1: zero, x2: zero, y1: 0, y2: rowH * values.length }));
    data.parts.forEach((part, p) => {
      const v = values[p];
      const yy = p * rowH;
      const len = v === null ? 0 : (Math.abs(v) / extent) * (barArea / 2 - 6);
      const x0 = v !== null && v < 0 ? zero - len : zero;
      root.append(svg("text", { x: labelW - 8, y: yy + 15, "text-anchor": "end" }, partLabel(part)));
      root.append(svg("rect", {
        x: x0, y: yy + 4, width: Math.max(len, v ? 1 : 0), height: rowH - 8, rx: 3,
        fill: v !== null && v < 0 ? cssVar("--negative") : cssVar("--positive"),
      }, svg("title", {}, `${part}: ${v}`)));
      root.append(svg("text", { x: width - 4, y: yy + 15, "text-anchor": "end" }, fmt(v)));
    });
    root.append(svg("text", { x: labelW - 8, y: rowH * values.length + 18, "text-anchor": "end", style: "font-weight:600" }, "sum of parts"),
      svg("text", { x: width - 4, y: rowH * values.length + 18, "text-anchor": "end", style: "font-weight:600" }, fmt(sum)));
    clear(body, root, el("div", { class: "legend" },
      el("span", {}, el("span", { class: "sw", style: { background: "var(--positive)" } }), "pushes the next token up"),
      el("span", {}, el("span", { class: "sw", style: { background: "var(--negative)" } }), "pushes it down")));
  }

  function partLabel(part) {
    const match = /^blocks\.(\d+)\.hook_(attn|mlp)_out$/.exec(part);
    if (match) return `block ${match[1]} ${match[2] === "attn" ? "attention" : "MLP"}`;
    return { hook_embed: "token embedding", hook_pos_embed: "position embedding", ln_bias: "final norm bias" }[part] || part;
  }

  // ---------------------------------------------------------------- attention pattern

  async function renderAttention() {
    const atStep = step();
    const { seq, layer } = state;
    const body = $("attn-body");
    const tap = `blocks.${layer}.attn.hook_pattern`;
    $("attn-sub").textContent = `readout of ${tap}, sequence ${seq}`;
    const found = findEvent(atStep, tap);
    if (found.error) {
      clear($("attn-meter"), meterRow("bad", "not in the trace", found.error));
      setPanelBad("attn-panel", true);
      clear(body);
      return;
    }
    clear($("attn-meter"), meterRow("info", "loading and hashing the shard…", ""));
    let loaded;
    try {
      loaded = await loadTensor(found.event);
    } catch (error) {
      if (atStep !== step() || layer !== state.layer) return;
      clear($("attn-meter"), meterRow("bad", "shard unreadable", error.message));
      setPanelBad("attn-panel", true);
      clear(body);
      return;
    }
    if (atStep !== step() || layer !== state.layer || seq !== state.seq) return;
    clear($("attn-meter"), hashRow(loaded, tap));
    setPanelBad("attn-panel", loaded.status === "bad");
    const [B, H, Q, K] = loaded.shape;
    if (loaded.shape.length !== 4 || seq >= B) {
      clear(body, errorBox(`expected (Batch, Head, QPos, KPos) with sequence ${seq}, got shape [${loaded.shape}]`));
      return;
    }
    if (state.nHeads !== H) {
      state.nHeads = H;
      fillSelect($("attn-head"), H, (h) => `${h}`);
      if (state.head >= H) state.head = 0;
      $("attn-head").value = String(state.head);
    }
    const head = state.head;
    const values = loaded.values;
    const base = (seq * H + head) * Q * K;
    const sqrt = $("attn-sqrt").checked;
    const color = rampFor("blue");
    const mask = hexToRgb(cssVar("--mask"));
    const image = new ImageData(K, Q);
    for (let q = 0; q < Q; q++) {
      for (let k = 0; k < K; k++) {
        const v = values[base + q * K + k];
        const rgb = k > q ? mask : color(sqrt ? Math.sqrt(Math.max(0, v)) : v);
        const o = (q * K + k) * 4;
        image.data[o] = rgb[0];
        image.data[o + 1] = rgb[1];
        image.data[o + 2] = rgb[2];
        image.data[o + 3] = 255;
      }
    }
    const scale = Math.max(1, Math.floor(Math.min(body.clientWidth || 512, 540) / K));
    const source = document.createElement("canvas");
    source.width = K;
    source.height = Q;
    source.getContext("2d").putImageData(image, 0, 0);
    const canvas = el("canvas", { width: K * scale, height: Q * scale, "aria-label": `attention pattern, layer ${layer}, head ${head}` });
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(source, 0, 0, K * scale, Q * scale);
    if (state.pos < Q) {
      ctx.strokeStyle = cssVar("--text-primary");
      ctx.lineWidth = 2;
      ctx.strokeRect(1, state.pos * scale + 1, K * scale - 2, Math.max(1, scale - 2));
    }
    const tokens = probeRow();
    const cell = (event) => {
      const r = canvas.getBoundingClientRect();
      const k = Math.floor(((event.clientX - r.left) / r.width) * K);
      const q = Math.floor(((event.clientY - r.top) / r.height) * Q);
      return [Math.min(Q - 1, Math.max(0, q)), Math.min(K - 1, Math.max(0, k))];
    };
    canvas.addEventListener("mousemove", (event) => {
      const [q, k] = cell(event);
      showTip(event, [`query ${q} “${tokenText(tokens[q])}” → key ${k} “${tokenText(tokens[k])}”`,
        k > q ? "masked: a token can't attend ahead" : `weight ${fmt(values[base + q * K + k], 4)}`]);
    });
    canvas.addEventListener("mouseleave", hideTip);
    canvas.addEventListener("click", (event) => { state.pos = cell(event)[0]; changed({ posChanged: true }); });
    const rowWeights = state.pos < Q ? Array.from({ length: state.pos + 1 }, (_, k) => [k, values[base + state.pos * K + k]]) : [];
    const topKeys = rowWeights.sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, 3);
    clear(body,
      el("div", { class: "heat-wrap" }, canvas),
      el("div", { class: "legend" },
        el("span", {}, "weight ", rampLegend("blue", "0", "1", sqrt ? "(square-root scale)" : "")),
        el("span", {}, el("span", { class: "sw", style: { background: "var(--mask)", border: "1px solid var(--border)" } }), "masked"),
        el("span", { text: "rows: query position; columns: key position. Click a row to choose its position." })),
      topKeys.length ? el("div", { class: "note" }, `From position ${state.pos} “${tokenText(tokens[state.pos])}”, head ${head} attends most to: `,
        topKeys.map(([k, w], i) => `${i ? ", " : ""}${k} “${tokenText(tokens[k])}” (${fmt(w)})`).join("")) : null);
  }

  function fillSelect(select, count, label) {
    clear(select, Array.from({ length: count }, (_, i) => el("option", { value: String(i), text: label(i) })));
  }

  // ---------------------------------------------------------------- residual stream norms

  function residTaps(n) {
    const taps = [];
    for (let l = 0; l < n; l++) taps.push(`blocks.${l}.hook_resid_pre`);
    taps.push(`blocks.${n - 1}.hook_resid_post`);
    return taps;
  }

  async function renderResidual() {
    const atStep = step();
    const seq = state.seq;
    const body = $("resid-body");
    const taps = residTaps(state.index.n_layers);
    $("resid-sub").textContent = `readout of ${taps.length} hook_resid taps, sequence ${seq}`;
    const found = taps.map((tap) => findEvent(atStep, tap));
    const missing = found.filter((f) => f.error);
    if (missing.length) {
      clear($("resid-meter"), missing.map((f) => meterRow("bad", "not in the trace", f.error)));
      setPanelBad("resid-panel", true);
      clear(body);
      return;
    }
    clear($("resid-meter"), meterRow("info", `loading and hashing ${taps.length} shards…`, ""));
    let loaded;
    try {
      loaded = await Promise.all(found.map((f) => loadTensor(f.event)));
    } catch (error) {
      if (atStep !== step()) return;
      clear($("resid-meter"), meterRow("bad", "shard unreadable", error.message));
      setPanelBad("resid-panel", true);
      clear(body);
      return;
    }
    if (atStep !== step() || seq !== state.seq) return;
    const statuses = loaded.map((t) => t.status);
    const summary = statuses.every((s) => s === "ok")
      ? meterRow("ok", "hash check passed", `all ${loaded.length} residual stream shards hash to their trace events`)
      : null;
    clear($("resid-meter"), summary || loaded.filter((t) => t.status !== "ok").map((t) => hashRow(t, t.event.tap)));
    if (summary) summary.title = loaded.map((t) => `${t.event.tap}: ${t.event.sha256}`).join("\n");
    setPanelBad("resid-panel", statuses.includes("bad"));

    const [B, T, D] = loaded[0].shape;
    if (loaded.some((t) => t.shape.length !== 3 || t.shape[1] !== T || t.shape[2] !== D) || seq >= B) {
      clear(body, errorBox(`expected (Batch, Pos, Embed) tensors with sequence ${seq}, got ${loaded.map((t) => `[${t.shape}]`).join(" ")}`));
      return;
    }
    const norms = loaded.map((t) => {
      const row = new Float64Array(T);
      for (let p = 0; p < T; p++) {
        let s = 0;
        const o = (seq * T + p) * D;
        for (let d = 0; d < D; d++) s += t.values[o + d] * t.values[o + d];
        row[p] = Math.sqrt(s);
      }
      return row;
    });
    const top = Math.max(...norms.map((r) => Math.max(...r)));
    const color = rampFor("blue");
    const cellW = Math.max(1, Math.floor(((body.clientWidth || 600) - 130) / T));
    const cellH = 20;
    const canvas = el("canvas", { width: T * cellW, height: norms.length * cellH, "aria-label": "residual stream norm per position and layer" });
    const ctx = canvas.getContext("2d");
    norms.forEach((row, r) => row.forEach((v, p) => {
      ctx.fillStyle = rgbCss(color(v / top));
      ctx.fillRect(p * cellW, r * cellH, cellW, cellH - 1);
    }));
    if (state.pos < T) {
      ctx.strokeStyle = cssVar("--text-primary");
      ctx.lineWidth = 1;
      ctx.strokeRect(state.pos * cellW + 0.5, 0.5, Math.max(1, cellW - 1), norms.length * cellH - 1);
    }
    const labels = taps.map((tap, i) => (i < taps.length - 1 ? (i === 0 ? "into block 0" : `into block ${i}`) : `after block ${taps.length - 2}`));
    const tokens = probeRow();
    const cell = (event) => {
      const r = canvas.getBoundingClientRect();
      const p = Math.floor(((event.clientX - r.left) / r.width) * T);
      const row = Math.floor(((event.clientY - r.top) / r.height) * norms.length);
      return [Math.min(norms.length - 1, Math.max(0, row)), Math.min(T - 1, Math.max(0, p))];
    };
    canvas.addEventListener("mousemove", (event) => {
      const [row, p] = cell(event);
      showTip(event, [`${taps[row]} · position ${p} “${tokenText(tokens[p])}”`, `L2 norm ${fmt(norms[row][p])}`]);
    });
    canvas.addEventListener("mouseleave", hideTip);
    canvas.addEventListener("click", (event) => { state.pos = cell(event)[1]; changed({ posChanged: true }); });
    const labelCol = el("div", { style: { display: "flex", flexDirection: "column", flex: "none", width: "122px" } },
      labels.map((label, i) => el("div", { class: "mono secondary", title: taps[i], style: { height: `${cellH}px`, lineHeight: `${cellH}px`, fontSize: "12px" }, text: label })));
    clear(body,
      el("div", { style: { display: "flex", gap: "8px", alignItems: "flex-start" } }, labelCol, el("div", { class: "scrollx" }, canvas)),
      el("div", { class: "legend" }, el("span", {}, "L2 norm over Embed ", rampLegend("blue", "0", fmt(top, 2), "")),
        el("span", { text: "columns: positions 0 to " + (T - 1) + ". Click a column to choose its position." })),
      state.pos < T ? el("div", { class: "note", text: `At position ${state.pos} “${tokenText(tokens[state.pos])}”: ${norms.map((r, i) => `${labels[i]} ${fmt(r[state.pos], 2)}`).join(" · ")}` }) : null);
  }

  // ---------------------------------------------------------------- layer structure, across checkpoints

  async function loadLayerStructure() {
    const steps = state.index.steps;
    let next = 0;
    const worker = async () => {
      while (next < steps.length) {
        const s = steps[next++];
        try {
          state.layerStructure.set(s, await payload("layer_structure", s));
        } catch (error) {
          state.layerStructureErrors.push(`step ${s}: ${error.message}`);
        }
        renderLayerStructure();
      }
    };
    await Promise.all([worker(), worker(), worker(), worker()]);
    state.layerStructureDone = true;
    renderLayerStructure();
  }

  function renderLayerStructure() {
    const meterBox = $("layers-meter");
    const body = $("layers-body");
    const steps = state.index.steps.filter((s) => state.layerStructure.has(s));
    const total = state.index.steps.length;
    const rows = [];
    const sample = state.layerStructure.get(step()) || state.layerStructure.get(steps[0]);
    if (sample) {
      const m = sample.meter || {};
      rows.push(meterRow("info", "composite meter", Object.entries(m).filter(([k]) => k !== "kind")
        .map(([k, v]) => `${k}: ${v === "hash_check" ? "hash check (computed from hash-checked trace tensors)" : v === "logit_lens" ? "the logit lens meter" : v}`).join(" · ")));
    }
    if (!state.layerStructureDone) rows.push(meterRow("info", `loaded ${steps.length} of ${total} checkpoints`, ""));
    for (const error of state.layerStructureErrors) rows.push(meterRow("bad", "missing", error));
    clear(meterBox, rows);
    if (!steps.length) {
      clear(body);
      return;
    }
    const nStreams = state.layerStructure.get(steps[0]).residual_norm_mean.length;
    const streamNames = Array.from({ length: nStreams }, (_, i) => (i < nStreams - 1 ? `into block ${i}` : `after block ${nStreams - 2}`));
    const lensNames = Array.from({ length: state.layerStructure.get(steps[0]).lens_kl.length },
      (_, i) => (i === 0 ? "embeddings" : `after block ${i - 1}`));
    const charts = [
      lineChart("Mean residual stream norm", "L2 norm", streamNames.map((name, i) => ({ name, points: steps.map((s) => [s, state.layerStructure.get(s).residual_norm_mean[i]]) }))),
      lineChart("Logit lens KL from the model (probe mean)", "nats", lensNames.map((name, i) => ({ name, points: steps.map((s) => [s, state.layerStructure.get(s).lens_kl[i]]) }))),
    ];
    const entropy0 = state.layerStructure.get(steps[0]).attention_entropy;
    const entropyCharts = entropy0.map((heads, l) => lineChart(`Attention entropy, layer ${l}`, "nats",
      heads.map((_, h) => ({ name: `head ${h}`, points: steps.map((s) => [s, state.layerStructure.get(s).attention_entropy[l][h]]) }))));
    clear(body, el("div", { class: "small-multiples" }, charts), el("div", { class: "small-multiples", style: { marginTop: "8px" } }, entropyCharts),
      el("div", { class: "note", text: "Attention entropy is the mean over sequences and query positions of −Σ p ln p over keys. The dashed line is the current checkpoint." }));
  }

  function lineChart(title, unit, series) {
    const width = 340;
    const height = 190;
    const m = { l: 46, r: 18, t: 22, b: 26 };
    const allSteps = state.index.steps;
    const maxStep = Math.max(...allSteps);
    const x = xScaleLog(maxStep, m.l, width - m.r);
    const values = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => v !== null && Number.isFinite(v));
    let lo = Math.min(0, ...values);
    let hi = Math.max(...values, lo + 1e-9);
    const pad = (hi - lo) * 0.05;
    hi += pad;
    const y = (v) => m.t + (height - m.t - m.b) * (1 - (v - lo) / (hi - lo));
    const colors = SERIES();
    const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, style: "max-width:100%;height:auto", role: "img", "aria-label": title });
    root.append(svg("text", { x: 0, y: 12, style: "font-weight:600" }, `${title}`), svg("text", { x: width, y: 12, "text-anchor": "end" }, unit));
    for (const t of niceTicks(lo, hi, 3)) {
      root.append(svg("line", { class: "gridline", x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }),
        svg("text", { x: m.l - 4, y: y(t) + 4, "text-anchor": "end" }, fmt(t, 2)));
    }
    root.append(svg("line", { class: "axis", x1: m.l, x2: width - m.r, y1: height - m.b, y2: height - m.b }));
    for (const t of logTicks(maxStep, 0.16)) root.append(svg("text", { x: x(t), y: height - m.b + 13, "text-anchor": "middle" }, String(t)));
    const cx = x(step());
    root.append(svg("line", { x1: cx, x2: cx, y1: m.t, y2: height - m.b, stroke: cssVar("--text-secondary"), "stroke-dasharray": "3 3" }));
    series.forEach((s, i) => {
      const colour = colors[i % colors.length];
      let d = "";
      let pen = false;
      for (const [st, v] of s.points) {
        if (v === null || !Number.isFinite(v)) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${x(st).toFixed(1)},${y(v).toFixed(1)}`;
        pen = true;
      }
      root.append(svg("path", { d, fill: "none", stroke: colour, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": i >= colors.length ? "4 3" : null }));
      for (const [st, v] of s.points) {
        if (v === null || !Number.isFinite(v)) continue;
        const dot = svg("circle", { cx: x(st), cy: y(v), r: st === step() ? 4 : 2.5, fill: colour, stroke: cssVar("--surface-1"), "stroke-width": 1 });
        dot.append(svg("title", {}, `${s.name} · step ${st}: ${fmt(v, 4)} ${unit}`));
        root.append(dot);
      }
    });
    const legend = el("div", { class: "legend" }, series.map((s, i) => el("span", {},
      el("span", { class: "sw", style: { background: colors[i % colors.length] } }), s.name)));
    const current = series.map((s) => s.points.find((p) => p[0] === step())).map((p, i) => (p ? `${series[i].name} ${fmt(p[1], 3)}` : null)).filter(Boolean);
    return el("div", {}, root, legend, el("div", { class: "note", text: `At step ${step()}: ${current.join(" · ")}` }));
  }

  // ---------------------------------------------------------------- orchestration

  // A panel that throws says so in its meter rather than sitting on "loading".
  function report(promise, meterId) {
    promise.catch((error) => {
      console.error(error);
      if (meterId) clear($(meterId), meterRow("bad", "this panel failed to render", String(error.message || error)));
    });
  }

  function renderStepPanels() {
    renderTimeline();
    report(renderProbe(), "loss-meter");
    report(renderLens(), "lens-meter");
    report(renderAttribution(), "attr-meter");
    report(renderAttention(), "attn-meter");
    report(renderResidual(), "resid-meter");
    if (state.layerStructure.size) renderLayerStructure();
  }

  function selectStep(i) {
    const n = state.index.steps.length;
    i = Math.max(0, Math.min(n - 1, i));
    if (i === state.stepIndex && state.started) return;
    state.stepIndex = i;
    state.started = true;
    writeHash();
    renderStepPanels();
  }

  function changed({ seqChanged = false, posChanged = false, layerChanged = false, headChanged = false } = {}) {
    const T = probeRow().length;
    state.pos = Math.max(0, Math.min(T - 1, state.pos));
    writeHash();
    if (seqChanged) {
      report(renderProbe(), "loss-meter");
      report(renderResidual(), "resid-meter");
    }
    if (posChanged && !seqChanged) {
      for (const span of document.querySelectorAll("#probe-text .tok")) span.classList.toggle("sel", Number(span.dataset.pos) === state.pos);
      report(renderResidual(), "resid-meter");
    }
    if (seqChanged || posChanged) {
      report(renderLens(), "lens-meter");
      report(renderAttribution(), "attr-meter");
    }
    if (seqChanged || posChanged || layerChanged || headChanged) report(renderAttention(), "attn-meter");
  }

  function wireControls() {
    const slider = $("step-slider");
    slider.max = String(state.index.steps.length - 1);
    slider.addEventListener("input", () => selectStep(Number(slider.value)));
    $("prev-step").addEventListener("click", () => selectStep(state.stepIndex - 1));
    $("next-step").addEventListener("click", () => selectStep(state.stepIndex + 1));
    document.addEventListener("keydown", (event) => {
      if (event.target.closest && event.target.closest("input, select, textarea")) return;
      if (event.key === "ArrowLeft") selectStep(state.stepIndex - 1);
      else if (event.key === "ArrowRight") selectStep(state.stepIndex + 1);
    });
    fillSelect($("attn-layer"), state.index.n_layers, (l) => `${l}`);
    $("attn-layer").value = String(state.layer);
    $("attn-layer").addEventListener("change", (event) => { state.layer = Number(event.target.value); changed({ layerChanged: true }); });
    // Heads come from the pattern's shape; until it loads, offer head 0 or the one in the address.
    fillSelect($("attn-head"), state.head + 1, (h) => `${h}`);
    $("attn-head").value = String(state.head);
    $("attn-head").addEventListener("change", (event) => { state.head = Number(event.target.value); changed({ headChanged: true }); });
    $("attn-sqrt").addEventListener("change", () => changed({ headChanged: true }));
    let resizeTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(renderStepPanels, 200);
    });
    const scheme = window.matchMedia("(prefers-color-scheme: dark)");
    if (scheme.addEventListener) scheme.addEventListener("change", renderStepPanels);
  }

  async function probeBatchHash(probe) {
    const B = probe.length;
    const T = probe[0].length;
    const bytes = new Uint8Array(B * T * 4);
    const view = new DataView(bytes.buffer);
    probe.forEach((row, b) => row.forEach((id, t) => view.setInt32((b * T + t) * 4, id, true)));
    return sha256Hex(bytes);
  }

  // Reads sequence, position, layer and head from the address into state; returns the step's index, or -1.
  function applyHash() {
    const index = state.index;
    const saved = readHash();
    state.seq = saved.seq !== null && saved.seq >= 0 && saved.seq < index.probe.length ? saved.seq : 0;
    state.pos = saved.pos !== null && saved.pos >= 0 && saved.pos < index.probe[0].length ? saved.pos : 0;
    state.layer = saved.layer !== null && saved.layer >= 0 && saved.layer < index.n_layers ? saved.layer : 0;
    state.head = saved.head !== null && saved.head >= 0 ? saved.head : 0;
    return saved.step === null ? -1 : index.steps.indexOf(saved.step);
  }

  function fatal(message) {
    clear($("facts"), el("span", { class: "error", text: message }));
    clear($("app"), el("section", { class: "panel bad" }, el("h2", { text: "The viewer can't start" }), errorBox(message),
      el("p", { class: "note" }, `It reads the view index at ${VIEWS_DIR}index.json and the trace store at ${TRACE_DIR}events.jsonl, relative to this page. `,
        "Serve the repo root with python3 -m http.server and open /viewer/ through localhost, or point ?views= and ?trace= at other relative folders.")));
  }

  async function main() {
    let index;
    let eventsText;
    try {
      [index, eventsText] = await Promise.all([
        fetchJSON(`${VIEWS_DIR}index.json`, "view index"),
        fetchText(`${TRACE_DIR}events.jsonl`, "trace events"),
      ]);
    } catch (error) {
      fatal(error.message);
      return;
    }
    for (const field of ["run_id", "steps", "views", "files", "probe", "n_layers"]) {
      if (!(field in index)) {
        fatal(`index.json has no "${field}"; it doesn't follow build.py's contract`);
        return;
      }
    }
    if (!index.steps.length || !index.probe.length) {
      fatal("index.json lists no steps or no probe sequences");
      return;
    }
    state.index = index;
    try {
      indexEvents(eventsText);
    } catch (error) {
      fatal(error.message);
      return;
    }
    state.probeHash = await probeBatchHash(index.probe);
    if (state.probeHash === null) {
      // Without Web Crypto the probe can't be matched by hash; fall back to the events' own batch, and say so.
      const hashes = new Set([...trace.byKey.values()].flat().map((e) => e.batch_hash));
      state.probeHash = hashes.size === 1 ? [...hashes][0] : null;
    }
    const savedStep = applyHash();
    renderFacts();
    if (!HASH_AVAILABLE) {
      $("facts").append(el("span", { class: "error", text: "Web Crypto is unavailable here, so no hash is checked. Open the page through localhost." }));
    }
    wireControls();
    selectStep(savedStep >= 0 ? savedStep : 0);
    // An address edited by hand moves the page to it.
    window.addEventListener("hashchange", () => {
      const at = applyHash();
      $("attn-layer").value = String(state.layer);
      if (state.nHeads !== null && state.head >= state.nHeads) state.head = 0;
      if (state.nHeads === null) fillSelect($("attn-head"), state.head + 1, (h) => `${h}`);
      $("attn-head").value = String(state.head);
      state.started = false;
      selectStep(at >= 0 ? at : state.stepIndex);
    });
    report(loadLayerStructure(), "layers-meter");
  }

  main();
})();
