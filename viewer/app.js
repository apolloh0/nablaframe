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
// Stub: the viewer thread implements this page.
"use strict";
