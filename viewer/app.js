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
// * When the index lists `jlens`, a J-lens panel sits beside the logit lens. It shows the
//   same table of top tokens per reading, its meter and badge, and per reading its
//   Jensen-Shannon divergence and top-1 agreement with the logit lens.
// * Both lens tables show, per reading, the actual next token's rank and probability from
//   `target_rank` and `target_prob`, so the next token is visible even outside the top 5.
// * The layer-structure lens-KL chart uses a log scale, so readings near zero stay
//   distinguishable.
// * In the J-lens table, the "vs logit lens" columns come directly after each reading's
//   name, before the top tokens, so the comparison D21 records is visible without
//   scrolling sideways.
// * When `../runs/phase3/interventions/index.json` exists, or the folder `?interventions=`
//   names, a compare panel lists its interventions by spec hash and step. Choosing one shows:
//   - the spec as readable lines, one per edit
//   - the probe text colored by per-token KL and by the next token's logit change, with a
//     switch between them
//   - each requested tap's change per position
//   - the effect's exactness meter, naming its identities and citing G3.1
//   A payload whose step differs from the timeline's current step says so, rather than
//   mixing two steps' numbers. The panel computes nothing: every number comes from the
//   payload.
// * When the index has `curves`, the timeline draws each as its own row on the step axis. Each
//   row has its name and unit, a line through its points on its own vertical scale, with nulls
//   left as gaps, and its value at the current step. With neither `curves` nor
//   `val_loss_curve`, it says so.
// * When the index has `next_token`, each probe sequence's last position is scored like any
//   other. The probe text colors it by its loss, and the lens tables outline its next token,
//   decoded with the index's vocab.
// * Model-level badges: it reads `../runs/phase4/badges.json`, or the file `?badges=` names,
//   which G4.6 writes.
//   - Each view's panel shows its subject's badge under its D14 badge or meter: the probe text
//     `per_token_loss`; the lens panels `logit_lens` and `jlens`; then `attribution`,
//     `attention_pattern`, `residual_norm` and `layer_structure`. The compare panel shows the
//     badge of each operation in the chosen spec.
//   - A badge shows its status, each fact as its label, its value in its unit and its gate, and
//     its notes.
//   - Units: a fraction as a percentage; a fraction removed to three decimals; a rank, 0-based
//     in the file, as #n counted from 1, so a median of 50 reads #51, followed by "of <the
//     tokens fact>" when the badge has one; a step as an integer; a logit signed, to three
//     decimals.
//   - A missing file at the default path hides the badges. A named file that is missing or
//     unreadable says so. Nothing is computed: every value is the file's.
// * Compare mode: when the payload has `top_tokens` and `top_probs`, the panel shows both
//   passes' top five tokens at the chosen position, with their probabilities, decoded with the
//   index's vocab, side by side. A payload without them says so.
// * Phase 5 badges: the badges file may also hold `topk`, `staircase`, `transcoder` and `dpatch`,
//   which G5.6 writes beside Phase 4's subjects into `runs/phase5/badges.json`. Their units add to
//   the badges': a score and a ratio to three decimals, edges and a count as integers. The compare
//   panel shows `dpatch`'s badge for a spec with a dpatch edit, and a dpatch edit reads as a patch
//   through dictionary <hash>'s code from batch <hash> at its tap, naming its latents when it has
//   them, each hash shortened with the full hash on hover.
// * Features: when the index has `features`, a features panel lists them by kind, taps and site.
//   Choosing one fetches its payload, whose shape is in src/nablaframe/views/features.py, and
//   shows:
//   - above it, the payload's `meters`, each value with its name, citing their gate and path on
//     hover; the payload's `meter`, the reconstruction's FVU on the probe; and the badge of the
//     payload's kind
//   - at the page's sequence and position, the probe's active latents, largest code first: each
//     latent's number, its code, its frequency and its three largest effect tokens
//   - for one latent, chosen from that list or by number: its frequency; its top contexts, each as
//     its held-out window's tokens from 8 before the position to 1 after, the position outlined,
//     with its code and, when the payload has `variables`, that window's values; and its effect's
//     top and bottom tokens with their logits, under the payload's effect label
//   The panel computes nothing: every number is the payload's. A payload whose step isn't the
//   timeline's says so, as the compare panel does, and offers to move there. A payload without a
//   meter isn't drawn. The chosen payload and latent live in the address's #fragment.
// * Phase 6 badges: the badges file may also hold `name`, `edge`, `open`, `generalize` and `dsteer`,
//   which G6.7 writes beside Phase 4's and Phase 5's subjects into `runs/phase6/badges.json`. Their
//   units add to the badges': parts as an integer, or to one decimal where a median falls between
//   two; a span, a `[first, last]` pair of steps, as "first to last"; a flag as yes or no; a label
//   as the file holds it; and a null in any of them as "none". The compare panel shows `dsteer`'s
//   badge for a spec with a dsteer edit, and a dsteer edit reads as moving dictionary <hash>'s
//   latents at its tap and positions, removing rows <hash> and adding rows <hash> at amounts <hash>,
//   each hash shortened with the full hash on hover.
// * Gestures: `?gestures=<dir>` names a run's records folder, where src/nablaframe/gestures/records.py
//   keeps the records and src/nablaframe/views/gestures.py writes `panel.json`. Without it the
//   gestures panel is hidden. The panel reads `panel.json`, whose shape is in views/gestures.py, and
//   lists its records by kind, in the order name, edge, open, generalize, each entry reading its
//   request in words with its meter first, and a filter narrows the list by any word of an entry.
//   Choosing one fetches its file, hashes it with crypto.subtle.digest("SHA-256"), and draws it only
//   when the hash is its entry's and the file's key and request are its entry's; a mismatch is
//   shown, never hidden. Above it sits its kind's badge, placed and drawn as every model-level badge
//   is. Then, every number being the record's:
//   - a name: its label, named or weak, and its name, a token value decoded with the index's vocab;
//     steering's follows beside the null's, with unchanged and ablated beside them, or, where no
//     causal test applies, the KL beside the null's, with the record's note; the name's F1 and its
//     set's; coverage at every description; the set, each latent with its F1, and where the pointed
//     latent ranks; the set's effect, its top and bottom tokens, with the value's rank from each
//     end, and the pointed latent's own effect beside it, labelled as that latent's; and the top
//     candidates. A latent with no name says it fires on no row of its class.
//   - an edge: follows beside its null and kept beside its null, then the label, beside follows
//     through the residual alone and the unfrozen swap; the output's follows beside its null; the
//     single latents beside the sets; and its two places, its pairs and its instances
//   - open-this-up: its target; the error curve, the median error at each number of parts for the
//     absolute, signed and random orders, on a log-scaled axis of parts with the 0.1 line marked; the
//     reading and the share within 128 parts; and, one measured instance at a time, its window and
//     position and its top parts, each with its position, latent, attribution and name
//   - generalize-this: the slices and the seeds as tables, each seed with its cosine; and its
//     checkpoints on the timeline. While the record is chosen, the timeline gains rows for its F1,
//     follows, the null's follows, the probe and the accuracy, with its envelope's runs shaded and
//     its gap's checkpoints marked, on the timeline's step axis; a step the views' index doesn't
//     have is drawn all the same. An edge's rows are follows, kept and their nulls.
//   The panel computes nothing. The fragment gains gesture=<key>.
// * Features link to names: in the features panel, an active latent with entries in the gestures
//   panel's `names`, under the payload's `<dictionary>-<site>`, shows its name records' names and
//   labels, and so does the chosen latent; choosing one opens it in the gestures panel. A chosen
//   latent with none says so.
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
// * The J-lens panel is hidden unless the index's `views` lists "jlens". Its rows are the
//   block outputs it reads; each one's comparison is the payload's `comparison` entry for
//   that reading, a mean over every probe position, against the logit lens at the same
//   block output. A payload without `comparison` says so in the panel's meter. Its table
//   is the logit lens's with the "vs logit lens" group moved up front, between the reading
//   and the top tokens; the logit lens's own table keeps its order.
// * Ranks are shown 1-based, as #1 to #5 head the top-token columns, so the outlined next
//   token in column #k reads rank #k. A view built without `target_rank` and
//   `target_prob` says so beside its table rather than leaving the columns empty.
// * On the log-scale chart, a reading at or below zero can't be placed; it is drawn as an
//   open marker on the bottom edge, and the chart's note says how many there are.
// * State lives in the address's #fragment (step, sequence, position, layer, head, and the
//   chosen intervention as compare=<spec hash>-<step>), so a view can be reopened as it was.
// * Compare mode reads the index src/nablaframe/interventions/cli.py writes last, then the
//   payload file each entry names, relative to the same folder. Without `?interventions=`,
//   a folder with no index.json (a 404) hides the panel; a named folder without one shows
//   the panel with the error. A payload without a "meter", or whose step, spec hash or array
//   shapes don't match its index entry and its own tokens, isn't drawn: the panel says why.
// * The spec's lines are its JSON's edits, in order, in words; hashes are shortened, with
//   the full hash on hover; alpha, seeds, steps and ranges are printed as the JSON holds them
//   (a range stays half-open, [a, b)).
// * The colored text is the payload's own `tokens`, decoded with the index's vocab, for the
//   page's chosen sequence and position; when they aren't the index's probe, it says so. KL
//   uses a sequential scale and the logit change a diverging one, centred on zero; each
//   scale's end is the largest magnitude in the payload, all sequences, and the legend
//   prints that value. The last position's logit change is null and drawn blank. The tap
//   changes are drawn like the residual norms, one row per requested tap, on one scale.
// * The exactness meter prints the payload's `identities` and `checked_by`; whether the
//   identities held is in that gate's report, so the panel never says they passed. With
//   the steps matched it also prints `kl_mean`.
// * On a step mismatch, the spec and the meter stay, the effect's numbers are hidden, and a
//   button moves the timeline to the payload's step when the index has it.
// * The timeline's rows share one log-scaled step axis: the validation loss first, when the
//   index has a non-empty `val_loss_curve`, then each of `curves` in the index's order. Every
//   row draws its finite points, lone ones as dots, and puts each checkpoint's marker on its
//   line where it has a value there; the step axis carries every checkpoint's marker too, so
//   a step no row has a value at stays reachable. A click anywhere on the timeline moves to
//   the nearest checkpoint. A row's value at the current step reads "not in this curve" when
//   the curve has no point there and "blank" when the point is null. A curve that doesn't
//   follow build.py's form isn't drawn: the timeline says which and why.
// * `next_token` must be one int per probe sequence; otherwise the header says so and the
//   page reads it as absent. The token after each window is shown after the probe text,
//   dashed, as the last position's target. The lenses, the attribution and the compare
//   panel's top tokens take the last position's next token from it; a view built without it
//   still has nulls there, and they read as such.
// * Badges are loaded once and placed under each panel's D14 badge, or under its meter when
//   it has none. A file that isn't {"badges": {...}} is unreadable; a subject the file has no
//   badge for, or a badge that isn't {"status", "facts", "notes"} with each fact {"label",
//   "value", "unit", "gate", "path"}, says so in its panel. Each fact's gate shows, on
//   hover, the path to its value in that gate's measurements and the report's commit, time
//   and SHA-256 from the file's "reports". A judged pass is marked as passing, a judged
//   fail as failing, "not tested" as a warning; a value that isn't a number in a numeric
//   unit is shown as the file holds it and marked. A fraction reads to one decimal place of
//   its percentage, and a rank's #n keeps the median's half. The compare panel shows one
//   badge per distinct operation, in the spec's order, whatever the step. So that a jswap
//   spec's badge doesn't sit beside an "unknown operation", a jswap edit reads as a swap of
//   tokens a's and b's J-lens coordinates at its tap, with J's hash shortened, and a zero
//   with a head names it, as the induction demo zeroes several heads of one tap.
// * The compare panel's top tokens sit between the colored text and the tap changes, at the
//   page's position, as one table: each rank's token and probability in the unintervened
//   pass, then the intervened. The probe's next token is outlined in both when the payload
//   ran on the probe, and a line names each pass's top token, or says it didn't change.
//   Top tokens shaped other than (Batch, Pos, k) with k the same in both passes are an error
//   in that section only.
// * Phase 5's units: a score and a ratio read to three decimals, edges and a count as plain
//   integers; a value that isn't a whole number in an integer unit is shown as the file holds it
//   and marked, as a step's is. Without `?badges=` the page still reads Phase 4's file, which has
//   no Phase 5 subject, so the features panel and a dpatch spec say their badge is missing; G5.6
//   prints addresses naming runs/phase5/badges.json.
// * A dpatch edit reads "patch through dictionary <hash>'s code from batch <hash> at <tap>,
//   <positions>", then "; latents 3, 5 only" when it lists latents, as the JSON holds them, or
//   "; no latent (an empty latents list, so it changes nothing)" when the list is empty.
// * The features panel sits below the compare panel and is hidden unless the index has
//   `features`; an empty list shows it with a note. Its entries are grouped by kind, in the
//   index's order, each a button reading its site and its taps (one tap when both are the same,
//   else input → output), with the dictionary's hash, the file and the step on hover. An entry
//   that doesn't follow build.py's form, or repeats another's dictionary and site, isn't listed,
//   and the panel says which and why. Nothing is chosen until a button is pressed.
// * A features payload is drawn only when it has its meter, its step, dictionary, kind, site and
//   taps match its index entry, and its lists have features.py's shapes: one entry per latent of
//   its width, contexts inside its held-out windows, one value per window for each variable, and
//   probe codes per sequence and position. Otherwise the panel says why. Probe codes shaped other
//   than the index's probe are an error in the active latents' section only.
// * Above the payload: one meter row for its copied meters, each value by name, numbers to three
//   decimals and ints as ints, the value as the report holds it on hover, and on the row's hover
//   the gate and path, with that report's commit, time and SHA-256 when the badges file describes
//   it; one row for the reconstruction meter, its FVU, its probe positions and its two taps; then
//   its kind's badge, placed and drawn as every model-level badge is.
// * On a step mismatch the meters and the badge stay, the latents and the chosen latent are
//   hidden, and a button moves the timeline to the payload's step when the index has it. The page
//   still opens at the address's step or the first checkpoint, so a payload built at the final
//   checkpoint first asks to move there.
// * The active latents are a table in the payload's order: each latent's number, its code, its
//   frequency and its first three `effect.top` tokens. Clicking a row chooses that latent. A
//   position with no active latent says every code is 0 there.
// * One latent is chosen by a row or by the number box under the table, whose range is the site's
//   width; a number outside it, or anything but a whole number, says so. It shows its frequency;
//   its contexts as a table of rank, window, position, code, one column per variable in the
//   payload's order (the int as held, and on hover the token it names in the index's vocab), and
//   the window's tokens from 8 before the position to 1 after, clipped to the window, "…" where
//   the window goes on; then the effect's label, verbatim, over two tables, top and bottom, of
//   each token and its logit signed to three decimals. A latent with no contexts says it never
//   fires on the held-out windows.
// * A frequency reads as a percentage to two decimals, or with an exponent below 0.01%; a code
//   to three decimals, as values elsewhere on the page.
// * The fragment gains features=<dictionary>-<site> and latent=<n>. Choosing another payload
//   clears the latent, since a latent's number names another latent in another dictionary.
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
  const INTERVENTIONS_DIR = dirParam("interventions", "../runs/phase3/interventions/");
  const INTERVENTIONS_NAMED = new URLSearchParams(location.search).has("interventions");
  // Model-level badges (D38): a file, not a folder.
  const BADGES_NAMED = !!new URLSearchParams(location.search).get("badges");
  const BADGES_PATH = new URLSearchParams(location.search).get("badges") || "../runs/phase4/badges.json";

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
      detail ? el("span", { class: "detail" }, detail) : null);
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
    violet: { light: ramp(["#f5f4fb", "#dcd8f3", "#b0a8e4", "#7d70cf", "#4a3aa7", "#2a1f6b"]),
              dark: ramp(["#1f1d2a", "#352d6b", "#5247a8", "#7d70cf", "#b0a8e4", "#dcd8f3"]) },
    red: { light: ramp(["#fdf3f3", "#f8cfcf", "#f09a9a", "#e34948", "#b02a2a", "#6e1414"]),
           dark: ramp(["#2a1c1c", "#6b1f1f", "#a82e2e", "#e34948", "#f09a9a", "#fbd0d0"]) },
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

  // Signed values: red below zero, blue above, both receding to the surface at zero.
  function diverging(t) {
    if (!Number.isFinite(t)) t = 0;
    return t < 0 ? rampFor("red")(-t) : rampFor("blue")(t);
  }

  function divergingLegend(lo, hi, unit) {
    const stops = [-1, -0.5, 0, 0.5, 1].map((t, i) => `${rgbCss(diverging(t))} ${i * 25}%`).join(",");
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
    // Compare mode: the intervention index's entries, the chosen one as `${spec_hash}-${step}`,
    // the probe's coloring ("kl" or "logit"), and loaded payloads by file name.
    compare: { entries: null, chosen: null, color: "kl", payloads: new Map() },
    // The index's next_token, when it's one int per probe sequence; otherwise null.
    nextToken: null,
    // Model-level badges: "loading", "hidden" (no file at the default path), "error" or "ok".
    modelBadges: { status: "loading", error: null, badges: null, reports: null },
    // Features (D45): the index's well-formed entries (null without `features`), why any others
    // aren't listed, the chosen one as `${dictionary}-${site}`, the chosen latent, and loaded
    // payloads by file name.
    features: { entries: null, problems: [], chosen: null, latent: null, payloads: new Map() },
  };

  const step = () => state.index.steps[state.stepIndex];

  function tokenText(id) {
    const vocab = state.index.vocab;
    return vocab && id >= 0 && id < vocab.length ? vocab[id] : `#${id}`;
  }
  const probeRow = () => state.index.probe[state.seq];

  // The token a probe position predicts: the next one in the window, or after the last position,
  // the index's next_token for that sequence. Null when there is none.
  function targetAt(seq, pos) {
    const tokens = state.index.probe[seq];
    if (pos + 1 < tokens.length) return tokens[pos + 1];
    return state.nextToken ? state.nextToken[seq] : null;
  }

  function readHash() {
    const values = new URLSearchParams(location.hash.slice(1));
    const int = (name) => (values.has(name) && /^-?\d+$/.test(values.get(name)) ? Number(values.get(name)) : null);
    const compare = values.get("compare");
    const features = values.get("features");
    const latent = int("latent");
    return {
      step: int("step"), seq: int("seq"), pos: int("pos"), layer: int("layer"), head: int("head"),
      compare: compare && /^[0-9a-f]{64}-(0|[1-9]\d*)$/.test(compare) ? compare : null,
      features: features && /^[0-9a-f]{64}-(0|[1-9]\d*)$/.test(features) ? features : null,
      latent: latent !== null && latent >= 0 ? latent : null,
    };
  }

  function writeHash() {
    const values = new URLSearchParams({ step: step(), seq: state.seq, pos: state.pos, layer: state.layer, head: state.head });
    if (state.compare.chosen) values.set("compare", state.compare.chosen);
    if (state.features.chosen) values.set("features", state.features.chosen);
    if (state.features.chosen && Number.isInteger(state.features.latent)) values.set("latent", state.features.latent);
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
    if ("next_token" in idx && !state.nextToken) {
      parts.push(el("span", { class: "error", text: `the index's next_token isn't one int per probe sequence (${idx.probe.length}), so the last position has no target here` }));
    }
    const mb = state.modelBadges;
    if (mb.status === "ok") {
      parts.push(el("span", { title: modelBadgeReportsTitle() }, "model-level badges: ", el("code", { text: BADGES_PATH })));
    } else if (mb.status === "error") {
      parts.push(el("span", { class: "error", text: `model-level badges: ${mb.error}` }));
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

  // Why a curve doesn't follow build.py's form, or null when it does.
  function curveProblem(c) {
    if (!c || typeof c !== "object" || Array.isArray(c)) return "it isn't {\"name\", \"unit\", \"points\"}";
    if (typeof c.name !== "string" || !c.name) return "its name isn't a non-empty string";
    if (typeof c.unit !== "string" || !c.unit) return "its unit isn't a non-empty string";
    if (!Array.isArray(c.points)) return "its points aren't a list";
    const bad = c.points.findIndex((p) => !p || typeof p !== "object" || !Number.isInteger(p.step)
      || !(p.value === null || (typeof p.value === "number" && Number.isFinite(p.value))));
    return bad === -1 ? null : `point ${bad} isn't {"step": an int, "value": a finite number or null}`;
  }

  // The timeline's rows, each {name, unit, points: [{step, value}] sorted by step}: the validation
  // loss when the index has one, then each well-formed curve. Malformed curves come back as problems.
  function timelineRows() {
    const idx = state.index;
    const rows = [];
    const problems = [];
    const byStep = (a, b) => a.step - b.step;
    const val = Array.isArray(idx.val_loss_curve) ? idx.val_loss_curve : [];
    if (val.length) {
      rows.push({ name: "validation loss", unit: "nats",
        points: val.filter((p) => p && Number.isFinite(p.step)).map((p) => ({ step: p.step, value: p.loss })).sort(byStep) });
    }
    if ("curves" in idx) {
      if (!Array.isArray(idx.curves)) {
        problems.push("the index's curves aren't a list, so none is drawn");
      } else {
        idx.curves.forEach((c, i) => {
          const why = curveProblem(c);
          if (why) problems.push(`curve ${i}${c && typeof c.name === "string" && c.name ? ` “${c.name}”` : ""} isn't drawn: ${why}`);
          else rows.push({ name: c.name, unit: c.unit, points: c.points.map((p) => ({ step: p.step, value: p.value })).sort(byStep) });
        });
      }
    }
    return { rows, problems };
  }

  const finiteValue = (v) => v !== null && v !== undefined && Number.isFinite(v);

  function renderTimeline() {
    const box = $("timeline");
    const idx = state.index;
    const steps = idx.steps;
    const current = step();
    const { rows, problems } = timelineRows();
    const width = Math.max(320, box.clientWidth || 800);
    const m = { l: 52, r: 24 };
    const hereTexts = rows.map((row) => {
      const at = row.points.find((p) => p.step === current);
      return at === undefined ? `step ${current}: not in this curve` : at.value === null ? `step ${current}: blank`
        : `step ${current}: ${fmt(at.value)} ${row.unit}`;
    });
    // When a row's name and its value at the current step can't share a line (about 6.4px a
    // character at 11px), every row puts its value on a line of its own.
    const twoLines = rows.some((row, r) => (row.name.length + row.unit.length + 3 + hereTexts[r].length) * 6.4 + 24 > width - m.r - 4);
    const labelH = twoLines ? 30 : 16;
    const plotH = rows.length > 1 ? 64 : 112;
    const rowH = labelH + plotH + 18;
    const top = 4;
    const axisY = top + (rows.length ? rows.length * rowH : 40);
    const height = axisY + 34;
    const maxStep = Math.max(...steps, ...rows.flatMap((r) => r.points.map((p) => p.step)));
    const x = xScaleLog(maxStep, m.l, width - m.r);

    const root = svg("svg", { width, height, role: "img",
      "aria-label": `${rows.length ? rows.map((r) => r.name).join(", ") : "Checkpoints"} over training, with the trace checkpoints marked` });
    // A click anywhere moves to the nearest checkpoint; the markers on top take their own clicks.
    root.append(svg("rect", {
      x: 0, y: 0, width, height, fill: "transparent", style: "cursor:pointer",
      onclick: (event) => {
        const px = event.clientX - root.getBoundingClientRect().left;
        let best = 0;
        steps.forEach((s, i) => { if (Math.abs(x(s) - px) < Math.abs(x(steps[best]) - px)) best = i; });
        selectStep(best);
      },
    }));
    const marker = (s, i, cx, cy, label, small = false) => {
      const isCurrent = i === state.stepIndex;
      const r = small ? (isCurrent ? 5 : 3.5) : (isCurrent ? 6 : 4);
      return svg("g", { style: "cursor:pointer", onclick: () => selectStep(i) },
        svg("circle", { cx, cy, r: 10, fill: "transparent" }),
        svg("circle", {
          cx, cy, r,
          fill: isCurrent ? cssVar("--accent") : cssVar("--surface-1"),
          stroke: isCurrent ? cssVar("--surface-1") : cssVar("--accent"), "stroke-width": 2,
        }),
        svg("title", {}, label));
    };

    rows.forEach((row, r) => {
      const y0 = top + r * rowH;
      const p0 = y0 + labelH;
      const p1 = p0 + plotH;
      const values = row.points.map((p) => p.value).filter(finiteValue);
      const lo = values.length ? Math.min(...values) : 0;
      const hi = values.length ? Math.max(...values) : 1;
      const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 0.5;
      const y = (v) => p0 + plotH * (1 - (v - (lo - pad)) / (hi - lo + 2 * pad));
      const at = new Map(row.points.map((p) => [p.step, p.value]));
      root.append(
        svg("text", { x: 4, y: y0 + 12, style: "font-weight:600" }, row.name, svg("tspan", { style: "font-weight:normal" }, ` (${row.unit})`)),
        svg("text", { x: width - m.r, y: y0 + (twoLines ? 26 : 12), "text-anchor": "end", style: "font-weight:600" }, hereTexts[r]));
      for (const t of niceTicks(lo - pad, hi + pad, plotH > 80 ? 4 : 2)) {
        if (y(t) < p0 - 0.5 || y(t) > p1 + 0.5) continue;
        root.append(svg("line", { class: "gridline", x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }),
          svg("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, fmt(t, 2)));
      }
      root.append(svg("line", { class: "axis", x1: m.l, x2: width - m.r, y1: p1, y2: p1 }));
      if (!values.length) {
        root.append(svg("text", { x: (m.l + width - m.r) / 2, y: (p0 + p1) / 2 + 4, "text-anchor": "middle" }, "every value in this curve is null"));
      }
      const colour = r === 0 && idx.val_loss_curve?.length ? cssVar("--series-1") : SERIES()[r % 8];
      let d = "";
      let pen = false;
      row.points.forEach((p, k) => {
        if (!finiteValue(p.value)) { pen = false; return; }
        d += `${pen ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.value).toFixed(1)}`;
        pen = true;
        // A point with no finite neighbour draws no line, so it gets a dot.
        const alone = !finiteValue(row.points[k - 1]?.value) && !finiteValue(row.points[k + 1]?.value);
        if (alone && !steps.includes(p.step)) {
          root.append(svg("circle", { cx: x(p.step), cy: y(p.value), r: 2.5, fill: colour },
            svg("title", {}, `${row.name} · step ${p.step}: ${fmt(p.value, 4)} ${row.unit}`)));
        }
      });
      if (d) root.append(svg("path", { d, fill: "none", stroke: colour, "stroke-width": 2, "stroke-linejoin": "round" }));
      steps.forEach((s, i) => {
        const v = at.get(s);
        if (finiteValue(v)) root.append(marker(s, i, x(s), y(v), `step ${s} · ${row.name} ${fmt(v, 4)} ${row.unit}`));
      });
    });

    // The current step's dashed line runs through each row's plot, not its label line.
    const cx = x(current);
    const spans = rows.length ? rows.map((_, r) => [top + r * rowH + labelH, top + (r + 1) * rowH]) : [[top, axisY]];
    spans[spans.length - 1][1] = axisY;
    for (const [y1, y2] of spans) {
      root.append(svg("line", { x1: cx, x2: cx, y1, y2, stroke: cssVar("--text-secondary"), "stroke-dasharray": "3 3", "pointer-events": "none" }));
    }
    root.append(svg("line", { class: "axis", x1: m.l, x2: width - m.r, y1: axisY, y2: axisY }));
    for (const t of logTicks(maxStep)) {
      root.append(svg("text", { x: x(t), y: axisY + 18, "text-anchor": "middle" }, String(t)));
    }
    root.append(svg("text", { x: width - m.r, y: height - 2, "text-anchor": "end" }, "training step (log scale)"));
    steps.forEach((s, i) => root.append(marker(s, i, x(s), axisY, `step ${s} (checkpoint ${i + 1} of ${steps.length})`, true)));

    clear(box, root);
    if (!rows.length) box.append(el("div", { class: "note", text: "The index has neither curves nor a validation-loss curve; checkpoints are shown on the step axis." }));
    for (const problem of problems) box.append(errorBox(problem));
    const val = (idx.val_loss_curve || []).find((p) => p && p.step === current);
    const v = val ? val.loss : undefined;
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
      const target = targetAt(state.seq, t);
      const last = t + 1 === tokens.length;
      span.addEventListener("mousemove", (event) => showTip(event, [
        `position ${t} · “${word}” (id ${id})`,
        loss === null
          ? (target === null ? "loss: blank, no next token inside the window" : "loss: blank in this view, though the index names a next token; was the view built without it?")
          : loss === undefined ? "loss not loaded" : `loss predicting the next token: ${fmt(loss)} nats`,
        target === null ? "" : `next token: “${tokenText(target)}”${last ? " (the token after the window, from the index's next_token)" : ""}`,
      ].filter(Boolean)));
      span.addEventListener("mouseleave", hideTip);
      return span;
    });
    // The token after the window, which the last position is scored against: shown, not clickable.
    const after = state.nextToken ? state.nextToken[state.seq] : null;
    const ghost = after === null ? null : el("span", {
      class: "tok after", title: `the token after the window, “${tokenText(after)}” (id ${after}): the last position's target, from the index's next_token`,
      text: tokenText(after),
    });
    clear($("probe-text"), spans, ghost);
    clear($("loss-legend"),
      el("span", {}, "loss on the next token: ", rampLegend("orange", 0, vocabSize ? `ln ${vocabSize} = ${fmt(top, 2)}` : fmt(top, 2), "nats")),
      el("span", {}, el("span", { class: "sw", style: { backgroundImage: "repeating-linear-gradient(135deg, var(--border) 0 2px, transparent 2px 6px)" } }),
        state.nextToken ? "blank: no loss in the view" : "blank: no next token in the window"),
      state.nextToken ? el("span", {}, el("span", { class: "sw", style: { border: "1px dashed var(--text-secondary)" } }),
        "after the window: the last position's next token, from the index") : null,
      el("span", { text: "Click a token to choose its position for the lenses and the attribution." }));
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

  const pct = (x) => (x === null || x === undefined || !Number.isFinite(x) ? "?" : `${(100 * x).toFixed(1)}%`);

  // The lens meter's two rows: what it measures, and its last reading.
  function lensMeterRows(meter, lastNote) {
    const last = meter[meter.length - 1];
    return [
      meterRow("info", "meter", "per reading, over every probe position: KL from the model's own prediction (nats) and top-1 agreement with it, in the “meter” columns"),
      last ? meterRow("info", "last reading", `KL ${fmt(last.kl_mean)} nats · top-1 agreement ${pct(last.top1_agreement)}. ${lastNote}`)
        : meterRow("bad", "no per-layer meter in the view", ""),
    ];
  }

  const topK = (data) => data.top_tokens?.[0]?.[0]?.[0]?.length ?? 0;

  // The actual next token's rank (1-based on screen) and probability, as two cells.
  function targetCells(data, l, seq, pos, target) {
    const rank = data.target_rank?.[l]?.[seq]?.[pos];
    const prob = data.target_prob?.[l]?.[seq]?.[pos];
    if (target === null) return [el("td", { class: "num vs muted", text: "none" }), el("td", { class: "num muted", text: "none" })];
    if (!Number.isInteger(rank)) {
      const why = rank === null ? "the view has no rank here; it may have been built without the index's next_token" : "the view has no rank here";
      return [el("td", { class: "num vs", title: why, text: "?" }), el("td", { class: "num", title: why, text: "?" })];
    }
    const word = tokenText(target);
    return [
      el("td", { class: `num vs${rank < topK(data) ? "" : " outside"}`, title: `“${word}” ranks #${rank + 1} of ${state.index.vocab?.length ?? "the vocabulary"} (0-based ${rank} in the view)`, text: `#${rank + 1}` }),
      el("td", { class: "num", title: `p(“${word}”) = ${prob}`, text: prob === null || prob === undefined ? "?" : (prob * 100 < 0.1 && prob > 0 ? fmt(prob) : pct(prob)) }),
    ];
  }

  // The table both lenses share: top tokens per reading, the next token's rank and probability,
  // then the meter's columns. Any group of columns the caller adds comes first, right after the
  // reading's name, so it stays on screen when the table scrolls sideways. Groups head the
  // columns, so their own headings stay short and the table fits beside its neighbour.
  function lensTable(data, seq, pos, { meter, extra = null }) {
    const tokens = probeRow();
    const target = targetAt(seq, pos);
    const last = pos + 1 === tokens.length;
    const color = rampFor("blue");
    const k = topK(data);
    const group = (text, span, title, cls = "") => el("th", { class: `group ${cls}`, colspan: String(span), title, text });
    // With a group up front, the top tokens open with a border of their own.
    const tokensVs = extra ? "vs" : "";
    const groups = el("tr", {}, el("th", {}),
      extra ? group(extra.label, extra.heads.length, extra.title, "vs") : null,
      group(`top ${k} tokens`, k, "each reading's most probable tokens, best first", tokensVs),
      group("next token", 2, "the actual next token under this reading", "vs"),
      group("meter", 2, "probe means: KL from the model's own prediction (nats) and top-1 agreement with it", "vs"));
    const head = el("tr", {}, el("th", { text: "reading" }),
      extra ? extra.heads : null,
      Array.from({ length: k }, (_, i) => el("th", { class: i === 0 ? tokensVs : "", text: `#${i + 1}` })),
      el("th", { class: "num vs", title: "the actual next token's rank under this reading; #1 is the top token", text: "rank" }),
      el("th", { class: "num", title: "the actual next token's probability under this reading", text: "p" }),
      el("th", { class: "num vs", title: "KL from the model's own prediction, nats, probe mean", text: "KL" }),
      el("th", { class: "num", title: "top-1 agreement with the model's own prediction, probe mean", text: "top-1" }));
    const rows = data.layers.map((layerName, l) => {
      const ids = data.top_tokens[l][seq][pos];
      const probs = data.top_probs[l][seq][pos];
      return el("tr", {},
        el("td", { class: "mono", title: layerName, text: lensLabel(layerName, l) }),
        extra ? extra.cells(l) : null,
        ids.map((id, i) => {
          const rgb = color(probs[i]);
          return el("td", {
            class: `cell${id === target ? " target" : ""}${i === 0 && extra ? " vs" : ""}`,
            style: { background: rgbCss(rgb), color: inkFor(rgb) },
            title: `token “${tokenText(id)}” (id ${id}) · p = ${probs[i]}${id === target ? " · the actual next token" : ""}`,
          }, el("div", { text: tokenText(id) }), el("div", { class: "p", text: (probs[i] * 100).toFixed(1) + "%" }));
        }),
        targetCells(data, l, seq, pos, target),
        el("td", { class: "num vs", text: meter[l] ? fmt(meter[l].kl_mean) : "?" }),
        el("td", { class: "num", text: meter[l] ? pct(meter[l].top1_agreement) : "?" }));
    });
    const missingRanks = !Array.isArray(data.target_rank) || !Array.isArray(data.target_prob);
    return [
      el("div", { class: "note" }, `Top tokens at “${tokenText(tokens[pos])}”. `,
        target === null ? "This is the last position, so there's no next token to outline or rank."
          : `The actual next token, “${tokenText(target)}”${last ? ", the token after the window from the index's next_token," : ""} is outlined where it appears, and its rank and probability follow the top ${k}.`),
      missingRanks ? errorBox("This view has no target_rank or target_prob, so the next token's rank isn't shown. Rebuild the views.") : null,
      el("div", { class: "scrollx" }, el("table", { class: "lens" }, el("thead", {}, groups, head), el("tbody", {}, rows))),
      el("div", { class: "legend" }, el("span", {}, "probability ", rampLegend("blue", "0", "1", "")),
        el("span", {}, el("span", { class: "outside", text: "#n" }), ` next token outside the top ${k}`)),
    ];
  }

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
    clear($("lens-meter"), lensMeterRows(meter, "The last reading is the model's own computation."));
    clear($("lens-badge"), data.badge ? badge(data.badge) : errorBox("This view carries no badge."));
    clear(body, el("div", { class: "note", text: "What the model would predict if it stopped at each residual stream." }),
      lensTable(data, seq, pos, { meter }));
  }

  // ---------------------------------------------------------------- J-lens

  const hasJLens = () => Array.isArray(state.index?.views) && state.index.views.includes("jlens");

  async function renderJLens() {
    if (!hasJLens()) return;
    const atStep = step();
    const { seq, pos } = state;
    const body = $("jlens-body");
    $("jlens-sub").textContent = `sequence ${seq}, position ${pos}`;
    let data;
    try {
      data = await payload("jlens", atStep);
    } catch (error) {
      if (atStep !== step()) return;
      clear($("jlens-meter"), meterRow("bad", "J-lens unavailable", error.message));
      clear($("jlens-badge"));
      clear(body);
      return;
    }
    if (atStep !== step() || seq !== state.seq || pos !== state.pos) return;
    const meter = Array.isArray(data.meter) ? data.meter : [];
    const comparison = Array.isArray(data.comparison) ? data.comparison : null;
    const rows = lensMeterRows(meter, "The last block's J is the identity, so the last reading is the logit lens's.");
    if (comparison) {
      rows.push(meterRow("info", "vs logit lens", "per reading, over every probe position: Jensen-Shannon divergence (nats) and top-1 agreement with the logit lens at the same block output, in the “vs logit lens” columns"));
    } else {
      rows.push(meterRow("bad", "no comparison in the view", "the payload has no comparison with the logit lens, so its columns read ?"));
    }
    clear($("jlens-meter"), rows);
    clear($("jlens-badge"), data.badge ? badge(data.badge) : errorBox("This view carries no badge."));
    const byLayer = (l) => comparison?.find((c) => c.layer === l);
    clear(body, el("div", { class: "note", text: "What the model would predict from each block's output, carried to the last block's output by that block's summed Jacobian J (D17), then read as the logit lens reads it." }),
      lensTable(data, seq, pos, {
        meter,
        extra: {
          label: "vs logit lens",
          title: "probe means, against the logit lens at the same block output: Jensen-Shannon divergence (nats) and top-1 agreement",
          heads: [
            el("th", { class: "num vs", title: "Jensen-Shannon divergence from the logit lens at the same block output, nats, probe mean", text: "JS" }),
            el("th", { class: "num", title: "top-1 agreement with the logit lens at the same block output, probe mean", text: "top-1" }),
          ],
          cells: (l) => {
            const c = byLayer(l);
            return [el("td", { class: "num vs", text: c ? fmt(c.js_mean) : "?" }), el("td", { class: "num", text: c ? pct(c.top1_agreement) : "?" })];
          },
        },
      }));
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
    const target = targetAt(seq, pos);
    $("attr-sub").textContent = target === null ? `sequence ${seq}, position ${pos}` : `sequence ${seq}, position ${pos}, on the logit of “${tokenText(target)}”`;
    const m = data.meter || {};
    clear($("attr-meter"), m.kind === "decomposition_error"
      ? meterRow("info", "decomposition error", `largest gap over the probe between the summed parts and the traced logit of the next token: ${fmt(m.max_abs)}`)
      : meterRow("bad", "unexpected meter", JSON.stringify(m)));
    clear($("attr-badge"), data.badge ? badge(data.badge) : errorBox("This view carries no badge."));

    const values = data.parts.map((_, p) => data.contributions[p][seq][pos]);
    if (values.every((v) => v === null)) {
      clear(body, el("div", { class: "note", text: target === null ? "Blank: the last position has no next token, so there's no logit to attribute."
        : "Blank: this view has no contributions here, though the index names a next token; it may have been built without it." }));
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
      lineChart("Logit lens KL from the model", "nats", lensNames.map((name, i) => ({ name, points: steps.map((s) => [s, state.layerStructure.get(s).lens_kl[i]]) })), { log: true }),
    ];
    const entropy0 = state.layerStructure.get(steps[0]).attention_entropy;
    const entropyCharts = entropy0.map((heads, l) => lineChart(`Attention entropy, layer ${l}`, "nats",
      heads.map((_, h) => ({ name: `head ${h}`, points: steps.map((s) => [s, state.layerStructure.get(s).attention_entropy[l][h]]) }))));
    clear(body, el("div", { class: "small-multiples" }, charts), el("div", { class: "small-multiples", style: { marginTop: "8px" } }, entropyCharts),
      el("div", { class: "note", text: "The logit lens KL is its meter's mean over the probe, on a log scale so readings near zero stay apart. Attention entropy is the mean over sequences and query positions of −Σ p ln p over keys. The dashed line is the current checkpoint." }));
  }

  // Decades from 10^lo to 10^hi, thinned so at most about four are labelled.
  function decadeTicks(loExp, hiExp) {
    const every = Math.max(1, Math.ceil((hiExp - loExp) / 4));
    const ticks = [];
    for (let e = hiExp; e >= loExp; e -= every) ticks.push(e);
    return ticks.reverse();
  }

  const fmtPow10 = (e) => (e >= -2 && e <= 3 ? String(+(10 ** e).toPrecision(1)) : `1e${e}`);

  function lineChart(title, unit, series, { log = false } = {}) {
    const width = 340;
    const height = 190;
    const m = { l: 46, r: 18, t: 22, b: 26 };
    const allSteps = state.index.steps;
    const maxStep = Math.max(...allSteps);
    const x = xScaleLog(maxStep, m.l, width - m.r);
    const values = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => v !== null && Number.isFinite(v));
    const plotTop = m.t;
    const plotBottom = height - m.b;
    // On a log scale only positive values have a place; the axis spans whole decades around them.
    const placeable = (v) => v !== null && Number.isFinite(v) && (!log || v > 0);
    let y;
    let ticks;
    if (log) {
      const positive = values.filter((v) => v > 0);
      const loExp = positive.length ? Math.floor(Math.log10(Math.min(...positive))) : -1;
      let hiExp = positive.length ? Math.ceil(Math.log10(Math.max(...positive))) : 0;
      if (hiExp <= loExp) hiExp = loExp + 1;
      y = (v) => plotTop + (plotBottom - plotTop) * (1 - (Math.log10(v) - loExp) / (hiExp - loExp));
      ticks = decadeTicks(loExp, hiExp).map((e) => [10 ** e, fmtPow10(e)]);
    } else {
      const lo = Math.min(0, ...values);
      let hi = Math.max(...values, lo + 1e-9);
      hi += (hi - lo) * 0.05;
      y = (v) => plotTop + (plotBottom - plotTop) * (1 - (v - lo) / (hi - lo));
      ticks = niceTicks(lo, hi, 3).map((t) => [t, fmt(t, 2)]);
    }
    const colors = SERIES();
    const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, style: "max-width:100%;height:auto", role: "img", "aria-label": title });
    root.append(svg("text", { x: 0, y: 12, style: "font-weight:600" }, `${title}`), svg("text", { x: width, y: 12, "text-anchor": "end" }, log ? `${unit}, log scale` : unit));
    for (const [t, label] of ticks) {
      root.append(svg("line", { class: "gridline", x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }),
        svg("text", { x: m.l - 4, y: y(t) + 4, "text-anchor": "end" }, label));
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
        if (!placeable(v)) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${x(st).toFixed(1)},${y(v).toFixed(1)}`;
        pen = true;
      }
      root.append(svg("path", { d, fill: "none", stroke: colour, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": i >= colors.length ? "4 3" : null }));
      for (const [st, v] of s.points) {
        if (v === null || !Number.isFinite(v)) continue;
        const r = st === step() ? 4 : 2.5;
        // A value a log axis can't place: an open marker on the bottom edge.
        const dot = placeable(v)
          ? svg("circle", { cx: x(st), cy: y(v), r, fill: colour, stroke: cssVar("--surface-1"), "stroke-width": 1 })
          : svg("circle", { cx: x(st), cy: plotBottom, r: r + 0.5, fill: cssVar("--surface-1"), stroke: colour, "stroke-width": 1.5 });
        dot.append(svg("title", {}, `${s.name} · step ${st}: ${fmt(v, 4)} ${unit}${placeable(v) ? "" : " (at or below zero, so off the log axis; drawn on its bottom edge)"}`));
        root.append(dot);
      }
    });
    const legend = el("div", { class: "legend" }, series.map((s, i) => el("span", {},
      el("span", { class: "sw", style: { background: colors[i % colors.length] } }), s.name)));
    const current = series.map((s) => s.points.find((p) => p[0] === step())).map((p, i) => (p ? `${series[i].name} ${fmt(p[1], 3)}` : null)).filter(Boolean);
    const offAxis = log ? values.filter((v) => v <= 0).length : 0;
    return el("div", {}, root, legend, el("div", { class: "note", text: `At step ${step()}: ${current.join(" · ")}` }),
      offAxis ? el("div", { class: "note", text: `${offAxis} value${offAxis === 1 ? " is" : "s are"} at or below zero, which a log scale can't place: open markers on the bottom edge.` }) : null);
  }

  // ---------------------------------------------------------------- model-level badges (D38)

  // Each view's subject and the slot under its D14 badge, or under its meter when it has none.
  const MODEL_BADGE_SLOTS = [
    ["per_token_loss", "loss-model-badge"], ["logit_lens", "lens-model-badge"], ["jlens", "jlens-model-badge"],
    ["attribution", "attr-model-badge"], ["attention_pattern", "attn-model-badge"],
    ["residual_norm", "resid-model-badge"], ["layer_structure", "layers-model-badge"],
  ];

  // Reads the badges G4.6 writes, if there are any. Without ?badges=, a 404 hides them.
  async function loadModelBadges() {
    const mb = state.modelBadges;
    const settle = (status, error = null) => {
      mb.status = status;
      mb.error = error;
      renderModelBadges();
      renderFacts();
      report(renderCompare(), "compare-meter");
      report(renderFeatures(), "features-meter");
    };
    if (!BADGES_NAMED) {
      let head;
      try {
        head = await fetch(localUrl(BADGES_PATH), { method: "HEAD", cache: "no-cache" });
      } catch (error) {
        settle("hidden");
        return;
      }
      if (head.status === 404) {
        settle("hidden");
        return;
      }
    }
    let data;
    try {
      data = await fetchJSON(BADGES_PATH, "model-level badges");
    } catch (error) {
      settle("error", `${BADGES_PATH} is missing or unreadable (${error.message})`);
      return;
    }
    const isMap = (v) => !!v && typeof v === "object" && !Array.isArray(v);
    if (!isMap(data) || !isMap(data.badges)) {
      settle("error", `${BADGES_PATH} is unreadable: it isn't {"badges": {...}}, the form G4.6 writes`);
      return;
    }
    mb.badges = data.badges;
    mb.reports = isMap(data.reports) ? data.reports : null;
    settle("ok");
  }

  function modelBadgeReportsTitle() {
    const reports = state.modelBadges.reports;
    if (!reports) return "the file names no reports";
    return Object.entries(reports).map(([gate, r]) => `${gate}: ${reportProvenance(r)}`).join("\n");
  }

  function reportProvenance(r) {
    if (!r || typeof r !== "object") return "not described in the file";
    return `commit ${r.git_commit ?? "?"}, ${r.timestamp_utc ?? "?"}, report sha256 ${r.sha256 ?? "?"}`;
  }

  // Why a badge doesn't follow badges.py's form, or null when it does.
  function badgeProblem(b) {
    if (!b || typeof b !== "object" || Array.isArray(b)) return "it isn't an object";
    if (typeof b.status !== "string") return "its status isn't a string";
    if (!Array.isArray(b.facts)) return "its facts aren't a list";
    if (!Array.isArray(b.notes) || !b.notes.every((n) => typeof n === "string")) return "its notes aren't a list of strings";
    const bad = b.facts.findIndex((f) => !f || typeof f !== "object" || typeof f.label !== "string" || !f.label
      || typeof f.unit !== "string" || typeof f.gate !== "string" || !Array.isArray(f.path) || !("value" in f));
    return bad === -1 ? null : `fact ${bad} isn't {"label", "value", "unit", "gate", "path"}`;
  }

  // A fact's value in its unit, as {text, odd}: odd when the file's value doesn't fit the unit,
  // in which case the text is the value as the file holds it. Display only; nothing is computed.
  function factValue(fact, facts) {
    const v = fact.value;
    const asHeld = (why) => ({ text: `${JSON.stringify(v)} ${fact.unit}`, odd: why });
    if (typeof v !== "number" || !Number.isFinite(v)) return asHeld(`not a number, in a badge whose unit is ${fact.unit}`);
    switch (fact.unit) {
      case "fraction":
        return { text: `${(100 * v).toFixed(1)}%` };
      case "fraction removed":
        return { text: v.toFixed(3) };
      case "rank": {
        const tokens = facts.find((f) => f.unit === "tokens" && typeof f.value === "number");
        return { text: `#${+(v + 1).toPrecision(12)}${tokens ? ` of ${tokens.value}` : ""}` };
      }
      case "step":
        return Number.isInteger(v) ? { text: String(v) } : asHeld("not an integer step");
      case "tokens":
        return Number.isInteger(v) ? { text: `${v} tokens` } : asHeld("not a whole count of tokens");
      case "logit":
        return { text: `${v >= 0 ? "+" : ""}${v.toFixed(3)}` };
      case "score":
      case "ratio":
        return { text: v.toFixed(3) };
      case "edges":
        return Number.isInteger(v) ? { text: String(v) } : asHeld("not a whole number of edges");
      case "count":
        return Number.isInteger(v) ? { text: String(v) } : asHeld("not a whole count");
      default:
        return asHeld(`a unit the viewer doesn't know, ${JSON.stringify(fact.unit)}`);
    }
  }

  const BADGE_STATUS = {
    "judged: pass": ["ok", "judged: pass", "judged by a gate, which passed"],
    "judged: fail": ["bad", "judged: FAIL", "judged by a gate, which didn't pass"],
    recorded: ["info", "recorded", "recorded by the gates, not judged pass or fail"],
    "not tested": ["warn", "not tested", "no Phase 4 test exercises it"],
    readout: ["info", "readout", "an exact readout, with no model-level structure to recover"],
  };

  // The subject's model-level badge as a block, or null while there is nothing to show.
  function modelBadgeBox(subject) {
    const mb = state.modelBadges;
    if (mb.status === "hidden" || mb.status === "loading") return null;
    if (mb.status === "error") return el("div", { class: "model-badge bad" }, el("b", { text: "Model-level badge (D38) unavailable: " }), mb.error);
    const b = mb.badges[subject];
    if (b === undefined) return el("div", { class: "model-badge bad" }, el("b", { text: "No model-level badge: " }), `${BADGES_PATH} has none for `, el("code", { text: subject }));
    const problem = badgeProblem(b);
    if (problem) return el("div", { class: "model-badge bad" }, el("b", { text: "Model-level badge unreadable: " }), `the badge for `, el("code", { text: subject }), ` in ${BADGES_PATH}: ${problem}`);
    const [kind, statusText, statusTitle] = BADGE_STATUS[b.status] || ["bad", `unknown status ${JSON.stringify(b.status)}`, "a status badges.py doesn't write"];
    const reports = mb.reports || {};
    const facts = b.facts.map((fact) => {
      const { text, odd } = factValue(fact, b.facts);
      const gateTitle = `${fact.gate}'s measurements at ${JSON.stringify(fact.path)}\n${fact.gate} report: ${reportProvenance(reports[fact.gate])}`;
      return el("li", {},
        `${fact.label}: `,
        el("b", { class: `value${odd ? " odd" : ""}`, title: `${JSON.stringify(fact.value)} in the file, unit ${fact.unit}${odd ? `\n${odd}` : ""}`, text }),
        odd ? el("span", { class: "odd", text: ` (${odd})` }) : null,
        " ", el("span", { class: "gate", title: gateTitle, text: fact.gate }));
    });
    return el("details", { class: `model-badge ${kind}`, open: true },
      el("summary", {},
        el("b", { text: "Model-level badge (D38): " }),
        el("span", { class: "status", title: statusTitle, text: statusText }),
        " · ", el("code", { text: subject }),
        b.facts.length ? el("span", { class: "muted", text: ` · ${b.facts.length} ${b.facts.length === 1 ? "fact" : "facts"} from the reports` }) : null),
      facts.length ? el("ul", {}, facts) : null,
      b.notes.map((note) => el("div", { class: "note", text: note })));
  }

  function renderModelBadges() {
    for (const [subject, slot] of MODEL_BADGE_SLOTS) clear($(slot), modelBadgeBox(subject));
  }

  // One badge per distinct operation in the spec, in its order.
  function compareModelBadges(data) {
    if (!data || !data.spec || !Array.isArray(data.spec.edits)) return null;
    const ops = [...new Set(data.spec.edits.map((e) => e && e.op).filter((op) => typeof op === "string"))];
    return ops.map((op) => modelBadgeBox(op));
  }

  // ---------------------------------------------------------------- compare mode (Phase 3, D27)

  const compareKey = (entry) => `${entry.spec_hash}-${entry.step}`;
  const isInt = (v) => Number.isInteger(v) && v >= 0;

  // Reads the intervention index, if there is one. Without ?interventions=, a 404 hides the panel.
  async function loadInterventions() {
    const indexPath = `${INTERVENTIONS_DIR}index.json`;
    if (!INTERVENTIONS_NAMED) {
      let head;
      try {
        head = await fetch(localUrl(indexPath), { method: "HEAD", cache: "no-cache" });
      } catch (error) {
        return;
      }
      if (head.status === 404) return;
    }
    $("compare-panel").hidden = false;
    $("compare-sub").textContent = `from ${INTERVENTIONS_DIR}`;
    clear($("compare-meter"), meterRow("info", "loading the intervention index…", ""));
    let index;
    try {
      index = await fetchJSON(indexPath, "intervention index");
    } catch (error) {
      clear($("compare-meter"), meterRow("bad", "intervention index unavailable", error.message));
      setPanelBad("compare-panel", true);
      return;
    }
    const entries = index && Array.isArray(index.interventions) ? index.interventions : null;
    const bad = entries === null ? -1 : entries.findIndex((e) => !e || typeof e !== "object" || !/^[0-9a-f]{64}$/.test(e.spec_hash)
      || !isInt(e.step) || typeof e.file !== "string" || !e.file);
    if (bad !== -1) {
      const why = entries === null ? 'it has no "interventions" list' : `entry ${bad} isn't {"spec_hash", "step", "file"}`;
      clear($("compare-meter"), meterRow("bad", "intervention index unreadable", `${indexPath}: ${why}, so it doesn't follow cli.py's contract`));
      setPanelBad("compare-panel", true);
      return;
    }
    const c = state.compare;
    c.entries = entries;
    if (c.chosen === null) {
      const here = entries.find((e) => e.step === step());
      if (here) c.chosen = compareKey(here);
      writeHash();
    }
    await renderCompare();
  }

  function comparePayload(entry) {
    const c = state.compare;
    if (!c.payloads.has(entry.file)) {
      const promise = fetchJSON(INTERVENTIONS_DIR + entry.file, `intervention ${short(entry.spec_hash)} · step ${entry.step}`)
        .then((data) => checkComparePayload(entry, data));
      promise.catch(() => c.payloads.delete(entry.file));
      c.payloads.set(entry.file, promise);
      if (c.payloads.size > 24) c.payloads.delete(c.payloads.keys().next().value);
    }
    return c.payloads.get(entry.file);
  }

  // A payload is drawn only when it has its meter and its arrays agree with its own tokens.
  function checkComparePayload(entry, data) {
    const name = entry.file;
    if (!data || typeof data !== "object" || !("meter" in data)) throw new Error(`${name} has no meter, so it isn't shown`);
    if (data.step !== entry.step) throw new Error(`the index lists ${name} at step ${entry.step}, but it holds step ${data.step}`);
    if (data.spec_hash !== entry.spec_hash) throw new Error(`the index lists ${name} under spec ${short(entry.spec_hash)}, but it holds spec ${short(data.spec_hash)}`);
    if (!data.spec || !Array.isArray(data.spec.edits)) throw new Error(`${name} has no spec with a list of edits`);
    const tokens = data.tokens;
    if (!Array.isArray(tokens) || !tokens.length || !tokens.every((row) => Array.isArray(row) && row.length === tokens[0].length)) {
      throw new Error(`${name}'s tokens aren't a (Batch, Pos) list`);
    }
    const sameShape = (array) => Array.isArray(array) && array.length === tokens.length
      && array.every((row, b) => Array.isArray(row) && row.length === tokens[b].length);
    for (const field of ["kl", "delta_logit", "delta_loss"]) {
      if (!sameShape(data[field])) throw new Error(`${name}'s ${field} isn't shaped like its tokens, (${tokens.length}, ${tokens[0].length})`);
    }
    if (!Array.isArray(data.taps) || !data.tap_changes || typeof data.tap_changes !== "object") {
      throw new Error(`${name} has no taps list and tap_changes`);
    }
    for (const tap of data.taps) {
      if (!sameShape(data.tap_changes[tap])) throw new Error(`${name}'s change for ${tap} is missing or isn't shaped like its tokens`);
    }
    return data;
  }

  function renderCompareList() {
    const c = state.compare;
    const here = step();
    clear($("compare-list"),
      el("span", { text: "Interventions" }),
      c.entries.length ? c.entries.map((e) => el("button", {
        type: "button",
        class: e.step === here ? "here" : null,
        "aria-pressed": String(compareKey(e) === c.chosen),
        title: `spec ${e.spec_hash}\nstep ${e.step}${e.step === here ? " (the timeline's current step)" : ""}\n${e.file}`,
        text: `${short(e.spec_hash)} · step ${e.step}`,
        onclick: () => {
          c.chosen = compareKey(e);
          writeHash();
          report(renderCompare(), "compare-meter");
        },
      })) : el("span", { class: "muted", text: "none listed" }),
      c.entries.some((e) => e.step === here) && c.entries.some((e) => e.step !== here)
        ? el("span", { class: "muted", text: `bold: at step ${here}, the timeline's` }) : null);
  }

  const hashCode = (hash) => el("code", { title: hash, text: typeof hash === "string" ? short(hash) : JSON.stringify(hash) });
  const tapCode = (tap) => el("code", { text: tap });

  function positionsText(positions) {
    if (positions === "all") return "every position";
    if (Array.isArray(positions)) {
      if (!positions.length) return "no position (a null edit)";
      return `${positions.length === 1 ? "position" : "positions"} ${positions.join(", ")}`;
    }
    if (positions && Array.isArray(positions.range) && positions.range.length === 2) {
      const [a, b] = positions.range;
      return a === b ? `no position (the empty range [${a}, ${b}))` : `positions [${a}, ${b}) (half-open)`;
    }
    return `positions ${JSON.stringify(positions)}`;
  }

  function componentText(component) {
    if (component === "embeddings") return "the embeddings and the tied output layer, D26";
    if (component === "ln_final") return "the final layer norm";
    const match = /^blocks\.(\d+)(\.attn|\.mlp)?$/.exec(component);
    if (!match) return null;
    return match[2] === ".attn" ? `block ${match[1]}'s attention` : match[2] === ".mlp" ? `block ${match[1]}'s MLP` : `block ${match[1]}, whole`;
  }

  // One edit as a readable line: its operation, where it applies, and its arguments as the JSON holds them.
  function editLine(e) {
    const at = positionsText(e.positions);
    const args = e.args && typeof e.args === "object" ? e.args : {};
    switch (e.op) {
      case "zero":
        return ["zero ", "head" in args ? `head ${JSON.stringify(args.head)} of ` : "", tapCode(e.tap), ` at ${at}`];
      case "mean":
        return ["replace ", tapCode(e.tap), ` at ${at} with its per-position mean over reference batch `, hashCode(args.reference)];
      case "resample":
        return ["resample ", tapCode(e.tap), ` at ${at} from reference batch `, hashCode(args.reference), `, seed ${JSON.stringify(args.seed)}`];
      case "patch":
        return ["patch ", tapCode(e.tap), ` at ${at} from source batch `, hashCode(args.source)];
      case "add":
        return [`add ${JSON.stringify(args.alpha)} × vector `, hashCode(args.vector), " to ", tapCode(e.tap), ` at ${at}`];
      case "swap": {
        const what = componentText(e.tap);
        return ["swap ", tapCode(e.tap), what ? ` (${what})` : "", ` with its weights from step ${JSON.stringify(args.step)}`,
          e.positions === "all" ? "" : `, at ${at}`];
      }
      case "jswap":
        return ["swap the J-lens coordinates of tokens ", tokenWithId(args.a), " and ", tokenWithId(args.b), " in ", tapCode(e.tap),
          ` at ${at}, with J `, hashCode(args.J)];
      case "dpatch":
        return ["patch through dictionary ", hashCode(args.dictionary), "'s code from batch ", hashCode(args.source), " at ",
          tapCode(e.tap), `, ${at}`, latentsText(args)];
      default:
        return [el("span", { class: "error", text: `unknown operation ${JSON.stringify(e.op)}: ` }), tapCode(String(e.tap)), ` at ${at}, arguments ${JSON.stringify(args)}`];
    }
  }

  // A dpatch edit's latents, when it names them: only their change counts (D45).
  function latentsText(args) {
    if (!("latents" in args)) return "";
    const latents = args.latents;
    if (!Array.isArray(latents)) return `; latents ${JSON.stringify(latents)}`;
    if (!latents.length) return "; no latent (an empty latents list, so it changes nothing)";
    return `; ${latents.length === 1 ? "latent" : "latents"} ${latents.map((l) => JSON.stringify(l)).join(", ")} only`;
  }

  function specBlock(data) {
    const edits = data.spec.edits;
    return el("div", { class: "spec" },
      el("div", { class: "note" }, `Spec `, hashCode(data.spec_hash), edits.length
        ? `, ${edits.length} ${edits.length === 1 ? "edit" : "edits"}. Swaps apply first, then the other edits, each in this order.`
        : ": no edits, so this is the unintervened pass."),
      edits.length ? el("ol", {}, edits.map((e) => el("li", {}, editLine(e)))) : null);
  }

  function exactnessRows(data, matched) {
    const m = data.meter || {};
    const rows = [];
    if (m.kind !== "exactness") {
      rows.push(meterRow("bad", "unexpected meter", `kind ${JSON.stringify(m.kind)}, expected exactness`));
    } else {
      const identities = Array.isArray(m.identities) ? m.identities : [];
      rows.push(identities.length && m.checked_by
        ? meterRow("info", "exactness", `the D25 identities ${identities.join(", ")}, each checked bit for bit by ${m.checked_by}; whether they held is in ${m.checked_by}'s report`)
        : meterRow("bad", "exactness meter incomplete", `identities ${JSON.stringify(m.identities)}, checked by ${JSON.stringify(m.checked_by)}`));
    }
    if (matched) rows.push(meterRow("info", "mean KL", `${fmt(data.kl_mean)} nats, the unintervened prediction against the intervened one, over every sequence and position`));
    return rows;
  }

  // The largest magnitude among a (Batch, Pos) list's numbers, skipping nulls: a color scale's end.
  function largest(rows) {
    let top = 0;
    for (const row of rows) for (const v of row) if (v !== null && Number.isFinite(v) && Math.abs(v) > top) top = Math.abs(v);
    return top;
  }

  const sameTokens = (a, b) => a.length === b.length && a.every((row, i) => row.length === b[i].length && row.every((v, j) => v === b[i][j]));

  async function renderCompare() {
    const c = state.compare;
    if (!c.entries) return;
    renderCompareList();
    const meter = $("compare-meter");
    const body = $("compare-body");
    const entry = c.entries.find((e) => compareKey(e) === c.chosen);
    if (!entry) {
      $("compare-sub").textContent = `from ${INTERVENTIONS_DIR}`;
      clear(meter, c.chosen ? meterRow("warn", "not in the index", `the address names intervention ${short(c.chosen.split("-")[0])} at step ${c.chosen.split("-")[1]}, which the index doesn't list`) : null);
      clear($("compare-model-badges"));
      clear(body, el("div", { class: "note", text: c.entries.length ? "Choose an intervention to see its spec and its effect." : "The intervention index lists no interventions." }));
      setPanelBad("compare-panel", false);
      return;
    }
    const { chosen } = c;
    const atStep = step();
    const { seq, pos } = state;
    $("compare-sub").textContent = `spec ${short(entry.spec_hash)} at step ${entry.step}`;
    let data;
    try {
      data = await comparePayload(entry);
    } catch (error) {
      if (chosen !== c.chosen) return;
      clear(meter, meterRow("bad", "intervention unavailable", error.message));
      clear($("compare-model-badges"));
      clear(body);
      setPanelBad("compare-panel", true);
      return;
    }
    if (chosen !== c.chosen || atStep !== step() || seq !== state.seq || pos !== state.pos) return;
    setPanelBad("compare-panel", false);
    const matched = data.step === atStep;
    clear(meter, exactnessRows(data, matched));
    clear($("compare-model-badges"), compareModelBadges(data));
    if (!matched) {
      const onTimeline = state.index.steps.includes(data.step);
      clear(body, specBlock(data), el("div", { class: "mismatch" },
        el("b", { text: `This intervention ran at step ${data.step}; the timeline is at step ${atStep}. ` }),
        "Its effect isn't shown beside another step's views. ",
        onTimeline
          ? el("button", { type: "button", text: `Move the timeline to step ${data.step}`, onclick: () => selectStep(state.index.steps.indexOf(data.step)) })
          : el("span", { text: `Step ${data.step} isn't one of the view index's checkpoints, so the timeline can't reach it.` })));
      return;
    }
    if (seq >= data.tokens.length) {
      clear(body, specBlock(data), errorBox(`The payload holds ${data.tokens.length} sequence(s); sequence ${seq} isn't one of them.`));
      return;
    }
    clear(body,
      specBlock(data),
      sameTokens(data.tokens, state.index.probe) ? null
        : el("div", { class: "mismatch", text: "This intervention ran on a batch other than the view index's probe: the text below is its own tokens, and the other panels' positions don't line up with it." }),
      el("h3", { text: `Effect on sequence ${seq}` }),
      el("div", { class: "controls" }, el("span", { text: "Color by" }),
        [["kl", "per-token KL"], ["logit", "next token's logit change"]].map(([mode, label]) => el("button", {
          type: "button", "aria-pressed": String(c.color === mode), text: label,
          onclick: () => { c.color = mode; report(renderCompare(), "compare-meter"); },
        }))),
      compareProbe(data, seq),
      compareTopTokens(data, seq, pos),
      el("h3", { text: "Requested taps' change" }),
      tapChanges(data, seq));
  }

  function compareProbe(data, seq) {
    const mode = state.compare.color;
    const field = mode === "kl" ? data.kl : data.delta_logit;
    const top = largest(field);
    const tokens = data.tokens[seq];
    const color = rampFor("violet");
    const spans = tokens.map((id, t) => {
      const value = field[seq][t];
      const word = tokenText(id);
      const span = el("span", { class: "tok", "data-pos": t, text: word });
      if (word.startsWith("<") && word.endsWith(">")) span.classList.add("special");
      if (value === null) span.classList.add("blank");
      else {
        const scaled = top > 0 ? value / top : 0;
        const rgb = mode === "kl" ? color(scaled) : diverging(scaled);
        span.style.background = rgbCss(rgb);
        span.style.color = inkFor(rgb);
      }
      if (t === state.pos) span.classList.add("sel");
      span.addEventListener("click", () => { state.pos = t; changed({ posChanged: true }); });
      span.addEventListener("mousemove", (event) => {
        const kl = data.kl[seq][t];
        const dl = data.delta_logit[seq][t];
        const dloss = data.delta_loss[seq][t];
        showTip(event, [
          `position ${t} · “${word}” (id ${id})`,
          `KL, unintervened ‖ intervened: ${kl === null ? "blank" : `${fmt(kl)} nats`}`,
          dl === null ? "next token's logit change: blank, no next token in the window" : `next token's logit change: ${fmt(dl)}`,
          dloss === null ? "" : `next token's loss change: ${fmt(dloss)} nats`,
          t + 1 < tokens.length ? `next token: “${tokenText(tokens[t + 1])}”` : "",
        ].filter(Boolean));
      });
      span.addEventListener("mouseleave", hideTip);
      return span;
    });
    const legend = mode === "kl"
      ? el("span", {}, "KL of the intervened prediction from the unintervened one: ", rampLegend("violet", 0, fmt(top), "nats"))
      : el("span", {}, "change in the actual next token's logit: ", divergingLegend(fmt(-top), fmt(top), ""), " blue raises it, red lowers it");
    return [
      el("div", { class: "probe", id: "compare-text" }, spans),
      el("div", { class: "legend" }, legend,
        top === 0 ? el("span", { text: "every value in this payload is 0" }) : el("span", { text: "scale: the payload's largest magnitude" }),
        mode === "logit" ? el("span", {}, el("span", { class: "sw", style: { backgroundImage: "repeating-linear-gradient(135deg, var(--border) 0 2px, transparent 2px 6px)" } }), "blank: no next token in the window") : null),
    ];
  }

  const tokenWithId = (id) => (Number.isInteger(id) ? `“${tokenText(id)}” (id ${id})` : JSON.stringify(id));

  // Why the payload's top tokens can't be drawn, or null. Each pass's tokens and probabilities are
  // (Batch, Pos, k), shaped like the payload's tokens, with one k throughout.
  function topTokensProblem(data) {
    const tokens = data.tokens;
    let k = null;
    for (const field of ["top_tokens", "top_probs"]) {
      const value = data[field];
      if (!value || typeof value !== "object" || Array.isArray(value)) return `its ${field} isn't {"unintervened", "intervened"}`;
      for (const pass of ["unintervened", "intervened"]) {
        const rows = value[pass];
        const ok = Array.isArray(rows) && rows.length === tokens.length && rows.every((row, b) => Array.isArray(row)
          && row.length === tokens[b].length && row.every((cell) => {
            if (!Array.isArray(cell) || !cell.length) return false;
            if (k === null) k = cell.length;
            return cell.length === k && cell.every((v) => (field === "top_tokens" ? Number.isInteger(v) : typeof v === "number"));
          }));
        if (!ok) return `its ${field}.${pass} isn't (Batch, Pos, k) like its tokens, (${tokens.length}, ${tokens[0].length}), with one k`;
      }
    }
    return null;
  }

  // Both passes' top tokens at the page's position, side by side. Every number is the payload's.
  function compareTopTokens(data, seq, pos) {
    const heading = el("h3", { text: "What each pass predicts" });
    if (!("top_tokens" in data) && !("top_probs" in data)) {
      return [heading, el("div", { class: "note", text: "This payload has no top_tokens or top_probs, so what each pass predicts isn't shown. The command line writes them since Phase 4's delta 4; rewrite the payload to see them." })];
    }
    const problem = topTokensProblem(data);
    if (problem) return [heading, errorBox(`The payload's top tokens can't be shown: ${problem}.`)];
    const tokens = data.tokens[seq];
    if (pos >= tokens.length) return [heading, errorBox(`The payload's sequences have ${tokens.length} positions; position ${pos} isn't one of them.`)];
    const onProbe = sameTokens(data.tokens, state.index.probe);
    const target = onProbe ? targetAt(seq, pos) : null;
    const passes = ["unintervened", "intervened"];
    const ids = passes.map((pass) => data.top_tokens[pass][seq][pos]);
    const probs = passes.map((pass) => data.top_probs[pass][seq][pos]);
    const k = ids[0].length;
    const color = rampFor("blue");
    const cell = (id, p, pass, first) => {
      const rgb = color(p);
      return [
        el("td", {
          class: `cell${id === target ? " target" : ""}${first ? "" : " vs"}`,
          style: { background: rgbCss(rgb), color: inkFor(rgb) },
          title: `${pass}: token “${tokenText(id)}” (id ${id}) · p = ${p}${id === target ? " · the probe's next token" : ""}`,
          text: tokenText(id),
        }),
        el("td", { class: "num", title: `p = ${p}`, text: p * 100 < 0.1 && p > 0 ? fmt(p) : pct(p) }),
      ];
    };
    const table = el("table", { class: "lens top-compare" },
      el("thead", {},
        el("tr", {}, el("th", {}), el("th", { class: "group", colspan: "2", text: "unintervened" }), el("th", { class: "group vs", colspan: "2", text: "intervened" })),
        el("tr", {}, el("th", { text: "rank" }), el("th", { text: "token" }), el("th", { class: "num", text: "p" }),
          el("th", { class: "vs", text: "token" }), el("th", { class: "num", text: "p" }))),
      el("tbody", {}, Array.from({ length: k }, (_, i) => el("tr", {},
        el("td", { class: "num", text: `#${i + 1}` }),
        cell(ids[0][i], probs[0][i], "unintervened", true),
        cell(ids[1][i], probs[1][i], "intervened", false)))));
    const [before, after] = [ids[0][0], ids[1][0]];
    return [
      heading,
      el("div", { class: "note" }, `Top ${k} tokens at position ${pos} “${tokenText(tokens[pos])}”, best first. `,
        before === after ? `The top token is “${tokenText(before)}” in both passes.`
          : `The top token changes from “${tokenText(before)}” to “${tokenText(after)}”.`,
        target === null ? (onProbe ? " There's no next token here to outline." : " This payload didn't run on the probe, so no next token is outlined.")
          : ` The probe's next token, “${tokenText(target)}”, is outlined where it appears.`),
      el("div", { class: "scrollx" }, table),
      el("div", { class: "legend" }, el("span", {}, "probability ", rampLegend("blue", "0", "1", ""))),
    ];
  }

  function tapChanges(data, seq) {
    const taps = data.taps;
    if (!taps.length) return el("div", { class: "note", text: "No taps were requested when this payload was written (the command line's --taps), so there are no tap changes to show." });
    const rows = taps.map((tap) => data.tap_changes[tap][seq]);
    const top = largest(taps.map((tap) => data.tap_changes[tap]).flat());
    const T = rows[0].length;
    const tokens = data.tokens[seq];
    const color = rampFor("orange");
    const box = $("compare-body");
    const cellW = Math.max(1, Math.floor(((box.clientWidth || 600) - 230) / T));
    const cellH = 20;
    const canvas = el("canvas", { width: T * cellW, height: rows.length * cellH, "aria-label": "each requested tap's change per position" });
    const ctx = canvas.getContext("2d");
    rows.forEach((row, r) => row.forEach((v, p) => {
      ctx.fillStyle = rgbCss(color(top > 0 && v !== null ? v / top : 0));
      ctx.fillRect(p * cellW, r * cellH, cellW, cellH - 1);
    }));
    if (state.pos < T) {
      ctx.strokeStyle = cssVar("--text-primary");
      ctx.lineWidth = 1;
      ctx.strokeRect(state.pos * cellW + 0.5, 0.5, Math.max(1, cellW - 1), rows.length * cellH - 1);
    }
    const cell = (event) => {
      const rect = canvas.getBoundingClientRect();
      const p = Math.floor(((event.clientX - rect.left) / rect.width) * T);
      const r = Math.floor(((event.clientY - rect.top) / rect.height) * rows.length);
      return [Math.min(rows.length - 1, Math.max(0, r)), Math.min(T - 1, Math.max(0, p))];
    };
    canvas.addEventListener("mousemove", (event) => {
      const [r, p] = cell(event);
      showTip(event, [`${taps[r]} · position ${p} “${tokenText(tokens[p])}”`, `L2 norm of the change: ${fmt(rows[r][p])}`]);
    });
    canvas.addEventListener("mouseleave", hideTip);
    canvas.addEventListener("click", (event) => { state.pos = cell(event)[1]; changed({ posChanged: true }); });
    const labelCol = el("div", { class: "tap-labels" },
      taps.map((tap) => el("div", { class: "mono secondary", title: tap, style: { height: `${cellH}px`, lineHeight: `${cellH}px` }, text: tap })));
    return [
      el("div", { style: { display: "flex", gap: "8px", alignItems: "flex-start" } }, labelCol, el("div", { class: "scrollx" }, canvas)),
      el("div", { class: "legend" }, el("span", {}, "L2 norm over the tap's trailing axes of intervened minus unintervened ", rampLegend("orange", "0", fmt(top), "")),
        el("span", { text: `one scale for every tap; columns: positions 0 to ${T - 1}` })),
      state.pos < T ? el("div", { class: "note", text: `At position ${state.pos} “${tokenText(tokens[state.pos])}”: ${taps.map((tap, r) => `${tap} ${fmt(rows[r][state.pos])}`).join(" · ")}` }) : null,
    ];
  }

  // ---------------------------------------------------------------- features (Phase 5, D45)

  const KIND_NAMES = { topk: "TopK SAE", staircase: "Staircase SAE", transcoder: "transcoder" };
  const kindName = (kind) => KIND_NAMES[kind] || kind;
  const featureKey = (entry) => `${entry.dictionary}-${entry.site}`;
  const isNumber = (v) => typeof v === "number" && Number.isFinite(v);
  const isMapping = (v) => !!v && typeof v === "object" && !Array.isArray(v);

  // Why a features entry of the index doesn't follow build.py's form, or null when it does.
  function featureEntryProblem(e) {
    if (!isMapping(e)) return "it isn't an object";
    if (typeof e.dictionary !== "string" || !/^[0-9a-f]{64}$/.test(e.dictionary)) return "its dictionary isn't a content hash";
    if (typeof e.kind !== "string" || !e.kind) return "its kind isn't a non-empty string";
    if (!isInt(e.site)) return "its site isn't a non-negative int";
    if (!Array.isArray(e.taps) || e.taps.length !== 2 || !e.taps.every((t) => typeof t === "string" && t)) return "its taps aren't [input tap, output tap]";
    if (!isInt(e.step)) return "its step isn't a non-negative int";
    if (typeof e.file !== "string" || !e.file) return "its file isn't a non-empty string";
    return null;
  }

  // Reads the index's `features`, if it has them: the panel shows only then.
  function readFeatureIndex() {
    const f = state.features;
    if (!("features" in state.index)) return;
    $("features-panel").hidden = false;
    f.entries = [];
    const list = state.index.features;
    if (!Array.isArray(list)) {
      f.problems.push("the index's features aren't a list, so none is listed");
      return;
    }
    const seen = new Map();
    list.forEach((e, i) => {
      const why = featureEntryProblem(e) ?? (seen.has(featureKey(e)) ? `it repeats entry ${seen.get(featureKey(e))}'s dictionary and site` : null);
      if (why) f.problems.push(`features entry ${i} isn't listed: ${why}`);
      else {
        seen.set(featureKey(e), i);
        f.entries.push(e);
      }
    });
  }

  function featuresPayload(entry) {
    const f = state.features;
    if (!f.payloads.has(entry.file)) {
      const promise = fetchJSON(VIEWS_DIR + entry.file, `features · ${kindName(entry.kind)} site ${entry.site}`)
        .then((data) => checkFeaturesPayload(entry, data));
      promise.catch(() => f.payloads.delete(entry.file));
      f.payloads.set(entry.file, promise);
      if (f.payloads.size > 6) f.payloads.delete(f.payloads.keys().next().value);
    }
    return f.payloads.get(entry.file);
  }

  // A payload is drawn only when it has its meter, matches its index entry, and holds its lists
  // in features.py's shapes. Nothing is computed here; this only checks what the panel reads.
  function checkFeaturesPayload(entry, data) {
    const fail = (why) => { throw new Error(`${entry.file} isn't drawn: ${why}`); };
    if (!isMapping(data) || !("meter" in data)) fail("it has no meter");
    for (const key of ["step", "dictionary", "kind", "site"]) {
      if (data[key] !== entry[key]) fail(`the index lists its ${key} as ${JSON.stringify(entry[key])}, but it holds ${JSON.stringify(data[key])}`);
    }
    if (!Array.isArray(data.taps) || data.taps.length !== 2 || data.taps[0] !== entry.taps[0] || data.taps[1] !== entry.taps[1]) {
      fail(`the index lists its taps as ${JSON.stringify(entry.taps)}, but it holds ${JSON.stringify(data.taps)}`);
    }
    const width = data.width;
    if (!Number.isInteger(width) || width < 1) fail("its width isn't a positive int");
    if (!Number.isInteger(data.k) || data.k < 1) fail("its k isn't a positive int");
    const perLatent = (v) => Array.isArray(v) && v.length === width;
    if (!perLatent(data.frequency) || !data.frequency.every(isNumber)) fail(`its frequency isn't one number per latent (${width})`);
    const held = data.held_out;
    if (!Array.isArray(held) || !held.length || !held.every((row) => Array.isArray(row) && row.length && row.every(Number.isInteger))) {
      fail("its held_out isn't a list of windows of token ids");
    }
    const inHeld = (c) => Array.isArray(c) && c.length === 3 && isInt(c[0]) && c[0] < held.length && isInt(c[1]) && c[1] < held[c[0]].length && isNumber(c[2]);
    if (!perLatent(data.contexts) || !data.contexts.every((list) => Array.isArray(list) && list.every(inHeld))) {
      fail("its contexts aren't, per latent, a list of [window, position, code] inside its held-out windows");
    }
    const effect = data.effect;
    const tokenPairs = (v) => perLatent(v) && v.every((list) => Array.isArray(list) && list.every((p) => Array.isArray(p) && p.length === 2 && Number.isInteger(p[0]) && isNumber(p[1])));
    if (!isMapping(effect) || typeof effect.label !== "string" || !effect.label || !tokenPairs(effect.top) || !tokenPairs(effect.bottom)) {
      fail('its effect isn\'t {"label", "top", "bottom"}, with a list of [token, logit] per latent');
    }
    if (!isMapping(data.variables) || !Object.values(data.variables).every((v) => Array.isArray(v) && v.length === held.length && v.every(Number.isInteger))) {
      fail(`its variables aren't {name: one int per held-out window (${held.length})}`);
    }
    const probe = data.probe;
    const cellsOk = (ids, values) => Array.isArray(ids) && Array.isArray(values) && ids.length === values.length
      && ids.every((i) => isInt(i) && i < width) && values.every(isNumber);
    if (!isMapping(probe) || !Array.isArray(probe.latents) || !Array.isArray(probe.values) || probe.latents.length !== probe.values.length
      || !probe.latents.every((row, b) => Array.isArray(row) && Array.isArray(probe.values[b]) && row.length === probe.values[b].length
        && row.every((ids, t) => cellsOk(ids, probe.values[b][t])))) {
      fail('its probe isn\'t {"latents", "values"}, (Batch, Pos) lists of latents and their codes');
    }
    return data;
  }

  function renderFeaturesList() {
    const f = state.features;
    const kinds = [...new Set(f.entries.map((e) => e.kind))];
    clear($("features-list"),
      kinds.map((kind) => el("div", { class: "controls" },
        el("span", { class: "kind", text: kindName(kind) }),
        f.entries.filter((e) => e.kind === kind).map((e) => el("button", {
          type: "button",
          "aria-pressed": String(featureKey(e) === f.chosen),
          title: `${kindName(e.kind)}, site ${e.site}\ndictionary ${e.dictionary}\nreads ${e.taps[0]}, reconstructs ${e.taps[1]}\nstep ${e.step}\n${e.file}`,
          text: `site ${e.site} · ${e.taps[0] === e.taps[1] ? e.taps[0] : `${e.taps[0]} → ${e.taps[1]}`}`,
          onclick: () => chooseFeatures(e),
        })))),
      f.entries.length ? null : el("div", { class: "note", text: "The index lists no features payloads." }),
      f.problems.map((problem) => errorBox(problem)));
  }

  function chooseFeatures(entry) {
    const f = state.features;
    if (featureKey(entry) !== f.chosen) f.latent = null; // a latent's number means another latent in another dictionary
    f.chosen = featureKey(entry);
    writeHash();
    report(renderFeatures(), "features-meter");
  }

  function chooseLatent(latent) {
    state.features.latent = latent;
    writeHash();
    report(renderFeatures(), "features-meter");
  }

  // A meter value as the report holds it: ints as ints, other numbers to three decimals.
  const meterValue = (v) => (Number.isInteger(v) ? String(v) : isNumber(v) ? fmt(v) : JSON.stringify(v));

  // The payload's meters, copied from its gate's report, then its own reconstruction meter.
  function featuresMeterRows(data) {
    const rows = [];
    const ms = data.meters;
    if (!isMapping(ms) || typeof ms.gate !== "string" || !Array.isArray(ms.path) || !isMapping(ms.values)) {
      rows.push(meterRow("bad", "meters unreadable", `the payload's meters aren't {"gate", "path", "values"}: ${JSON.stringify(ms)}`));
    } else {
      const reports = state.modelBadges.reports;
      const described = reports && reports[ms.gate] ? `\n${ms.gate} report: ${reportProvenance(reports[ms.gate])}` : "";
      const values = Object.entries(ms.values);
      rows.push(meterRow("info", `meters, from ${ms.gate}'s report`,
        values.length ? values.map(([name, v], i) => [i ? " · " : null, `${name} `, el("span", { title: `${name}: ${JSON.stringify(v)} in the report`, text: meterValue(v) })])
          : "the report's entry holds no values",
        `${ms.gate}'s measurements at ${JSON.stringify(ms.path)}${described}`));
    }
    const m = data.meter;
    if (!isMapping(m) || m.kind !== "reconstruction") {
      rows.push(meterRow("bad", "unexpected meter", `kind ${JSON.stringify(m && m.kind)}, expected reconstruction`));
    } else {
      rows.push(meterRow("info", "reconstruction on the probe",
        `FVU ${isNumber(m.fvu) ? fmt(m.fvu) : JSON.stringify(m.fvu)} over ${JSON.stringify(m.positions)} probe positions, from ${data.taps[0]} to ${data.taps[1]}`));
    }
    return rows;
  }

  // A share of held-out positions as a percentage; one too small for two decimals keeps its digits.
  function fmtShare(v) {
    const percent = 100 * v;
    return percent > 0 && percent < 0.01 ? `${percent.toExponential(1)}%` : `${percent.toFixed(2)}%`;
  }

  const signedLogit = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;

  async function renderFeatures() {
    const f = state.features;
    if (!f.entries) return;
    renderFeaturesList();
    const meter = $("features-meter");
    const body = $("features-body");
    const section = $("features-latent-section");
    const entry = f.entries.find((e) => featureKey(e) === f.chosen);
    if (!entry) {
      $("features-sub").textContent = "";
      const [dictionary, site] = f.chosen ? f.chosen.split("-") : [];
      clear(meter, f.chosen ? meterRow("warn", "not in the index", `the address names site ${site} of dictionary ${short(dictionary)}, which the index doesn't list`) : null);
      clear($("features-model-badge"));
      clear(body, f.entries.length ? el("div", { class: "note", text: "Choose a dictionary's site to see its latents." }) : null);
      section.hidden = true;
      setPanelBad("features-panel", false);
      return;
    }
    const { chosen, latent } = f;
    const atStep = step();
    const { seq, pos } = state;
    $("features-sub").textContent = `${kindName(entry.kind)}, site ${entry.site}, dictionary ${short(entry.dictionary)}, step ${entry.step}`;
    clear($("features-model-badge"), modelBadgeBox(entry.kind));
    let data;
    try {
      const pending = featuresPayload(entry);
      clear(meter, meterRow("info", "loading the features payload…", ""));
      data = await pending;
    } catch (error) {
      if (chosen !== f.chosen) return;
      clear(meter, meterRow("bad", "features payload unavailable", error.message));
      clear(body);
      section.hidden = true;
      setPanelBad("features-panel", true);
      return;
    }
    if (chosen !== f.chosen || latent !== f.latent || atStep !== step() || seq !== state.seq || pos !== state.pos) return;
    setPanelBad("features-panel", false);
    clear(meter, featuresMeterRows(data));
    if (data.step !== atStep) {
      const onTimeline = state.index.steps.includes(data.step);
      clear(body, el("div", { class: "mismatch" },
        el("b", { text: `This payload was built at step ${data.step}; the timeline is at step ${atStep}. ` }),
        "Its latents aren't shown beside another step's views. ",
        onTimeline
          ? el("button", { type: "button", text: `Move the timeline to step ${data.step}`, onclick: () => selectStep(state.index.steps.indexOf(data.step)) })
          : el("span", { text: `Step ${data.step} isn't one of the view index's checkpoints, so the timeline can't reach it.` })));
      section.hidden = true;
      return;
    }
    clear(body, activeLatents(data, seq, pos));
    const input = $("features-latent");
    input.max = String(data.width - 1);
    if (document.activeElement !== input) input.value = latent === null ? "" : String(latent);
    $("features-latent-range").textContent = `0 to ${data.width - 1}`;
    section.hidden = false;
    clear($("features-latent-body"), latentDetail(data, latent));
  }

  // The probe's active latents at the page's sequence and position, as the payload lists them.
  function activeLatents(data, seq, pos) {
    const tokens = probeRow();
    const heading = el("h3", { text: `Active latents at sequence ${seq}, position ${pos} “${tokenText(tokens[pos])}”` });
    const probe = state.index.probe;
    const latents = data.probe.latents;
    if (latents.length !== probe.length || latents.some((row, b) => row.length !== probe[b].length)) {
      return [heading, errorBox(`The payload's probe codes are for ${latents.length} sequence(s) of ${latents[0]?.length ?? 0} positions, the view index's probe is ${probe.length} of ${tokens.length}: it wasn't built on this probe, so its latents can't be placed at the page's position.`)];
    }
    const ids = latents[seq][pos];
    const codes = data.probe.values[seq][pos];
    if (!ids.length) return [heading, el("div", { class: "note", text: `No latent is active here: every one of the ${data.width} codes is 0.` })];
    const rows = ids.map((id, i) => el("tr", {
      class: id === state.features.latent ? "chosen" : null,
      title: `latent ${id}: choose it to see its contexts and its effect`,
      onclick: () => chooseLatent(id),
    },
    el("td", { class: "num", text: String(id) }),
    el("td", { class: "num", title: `code ${codes[i]}`, text: fmt(codes[i]) }),
    el("td", { class: "num", title: `frequency ${data.frequency[id]}`, text: fmtShare(data.frequency[id]) }),
    el("td", {}, data.effect.top[id].slice(0, 3).map(([token, value], j) => [j ? " " : null,
      el("span", { class: "eff", title: `“${tokenText(token)}” (id ${token}): ${value}`, text: tokenText(token) })]))));
    return [
      heading,
      el("div", { class: "note", text: `${ids.length} of the site's ${data.width} latents are active here (k = ${data.k}), largest code first. Choose one to see its contexts and its effect.` }),
      el("div", { class: "scrollx" }, el("table", { class: "features" },
        el("thead", {}, el("tr", {},
          el("th", { class: "num", text: "latent" }),
          el("th", { class: "num", text: "code" }),
          el("th", { class: "num", title: "the share of held-out positions at which it is active", text: "frequency" }),
          el("th", { title: "its three largest direct effects on the logits, largest first (see the effect's label below)", text: "largest effect tokens" }))),
        el("tbody", {}, rows))),
    ];
  }

  // One latent: its frequency, its top held-out contexts, and its effect, all as the payload holds them.
  function latentDetail(data, latent) {
    if (latent === null) return el("div", { class: "note", text: "Choose a latent from the list above, or type its number." });
    if (!Number.isInteger(latent)) return errorBox(`“${latent}” isn't a latent's number.`);
    if (latent >= data.width) return errorBox(`Latent ${latent} isn't one of this site's ${data.width} latents, 0 to ${data.width - 1}.`);
    const contexts = data.contexts[latent];
    const names = Object.keys(data.variables);
    const ctxRows = contexts.map(([w, p, code], i) => {
      const window = data.held_out[w];
      const lo = Math.max(0, p - 8);
      const hi = Math.min(window.length - 1, p + 1);
      const spans = [];
      if (lo > 0) spans.push(el("span", { class: "muted", text: "… " }));
      for (let t = lo; t <= hi; t++) {
        const word = tokenText(window[t]);
        const span = el("span", { class: `tok${t === p ? " sel" : ""}`, title: `window ${w}, position ${t}: “${word}” (id ${window[t]})`, text: word });
        if (word.startsWith("<") && word.endsWith(">")) span.classList.add("special");
        spans.push(span);
      }
      if (hi < window.length - 1) spans.push(el("span", { class: "muted", text: " …" }));
      return el("tr", {},
        el("td", { class: "num", text: `#${i + 1}` }),
        el("td", { class: "num", text: String(w) }),
        el("td", { class: "num", text: String(p) }),
        el("td", { class: "num", title: `code ${code}`, text: fmt(code) }),
        names.map((name, j) => {
          const v = data.variables[name][w];
          const asToken = state.index.vocab && v >= 0 && v < state.index.vocab.length ? `; as a token, “${tokenText(v)}”` : "";
          return el("td", { class: `num${j === 0 ? " vs" : ""}`, title: `${name} of window ${w}: ${v}${asToken}`, text: String(v) });
        }),
        el("td", { class: `ctx${names.length ? " vs" : ""}` }, spans));
    });
    const effectTable = (heading, pairs) => el("table", { class: "effect" },
      el("thead", {}, el("tr", {}, el("th", { text: heading }), el("th", { class: "num", text: "logit" }))),
      el("tbody", {}, pairs.map(([token, value]) => el("tr", {},
        el("td", { title: `id ${token}`, text: tokenText(token) }),
        el("td", { class: "num", title: String(value), text: signedLogit(value) })))));
    return [
      el("h3", { text: `Latent ${latent}` }),
      el("div", { class: "note" }, "Frequency ", el("b", { title: String(data.frequency[latent]), text: fmtShare(data.frequency[latent]) }),
        ": the share of held-out positions at which it is active."),
      el("h3", { text: "Top contexts" }),
      contexts.length
        ? [el("div", { class: "note", text: `Its ${contexts.length} largest codes on the held-out windows, largest first, each shown from 8 tokens before its position to 1 after, the position outlined.${names.length ? " The variables are the window's own." : ""}` }),
          el("div", { class: "scrollx" }, el("table", { class: "features" },
            el("thead", {},
              names.length ? el("tr", {}, el("th", { colspan: "4" }), el("th", { class: "group vs", colspan: String(names.length), text: "variables" }), el("th", { class: "vs" })) : null,
              el("tr", {},
                el("th", { class: "num", text: "" }), el("th", { class: "num", text: "window" }), el("th", { class: "num", text: "position" }),
                el("th", { class: "num", text: "code" }),
                names.map((name, j) => el("th", { class: `num${j === 0 ? " vs" : ""}`, text: name })),
                el("th", { class: names.length ? "vs" : null, text: "context" }))),
            el("tbody", {}, ctxRows)))]
        : el("div", { class: "note", text: "It never fires on the held-out windows, so it has no contexts." }),
      el("h3", { text: "Direct effect on the logits" }),
      el("div", { class: "note effect-label", text: data.effect.label }),
      el("div", { class: "effects" },
        effectTable(`top ${data.effect.top[latent].length}`, data.effect.top[latent]),
        effectTable(`bottom ${data.effect.bottom[latent].length}`, data.effect.bottom[latent])),
    ];
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
    report(renderJLens(), "jlens-meter");
    report(renderAttribution(), "attr-meter");
    report(renderAttention(), "attn-meter");
    report(renderResidual(), "resid-meter");
    report(renderCompare(), "compare-meter");
    report(renderFeatures(), "features-meter");
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
      report(renderJLens(), "jlens-meter");
      report(renderAttribution(), "attr-meter");
      report(renderCompare(), "compare-meter");
      report(renderFeatures(), "features-meter");
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
    $("features-latent").addEventListener("change", (event) => {
      const text = event.target.value.trim();
      chooseLatent(text === "" ? null : /^\d+$/.test(text) ? Number(text) : text);
    });
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
    state.compare.chosen = saved.compare;
    state.features.chosen = saved.features;
    state.features.latent = saved.features ? saved.latent : null;
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
    const next = index.next_token;
    state.nextToken = Array.isArray(next) && next.length === index.probe.length && next.every((id) => Number.isInteger(id)) ? next : null;
    $("jlens-panel").hidden = !hasJLens();
    readFeatureIndex();
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
    report(loadInterventions(), "compare-meter");
    loadModelBadges().catch((error) => console.error(error));
  }

  main();
})();
