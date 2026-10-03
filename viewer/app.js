// Nablaframe viewer (Phase 2, D13). Contract for the implementing thread:
//
// * Served by `python3 -m http.server` from the repo root. Every path it fetches is
//   relative; it contacts no other host, and loads no library from outside the repo.
// * Readouts come straight from the trace store: it parses events.jsonl, and for each
//   tensor it shows, fetches the shard and slices the "tensor" entry's bytes exactly as
//   src/nablaframe/views/shards.py does. Before displaying a tensor, it hashes that
//   slice with crypto.subtle.digest("SHA-256") and compares it with the event's
//   sha256. A mismatch is shown, never hidden. That check is a readout's meter.
// * Derived views come from the JSON files export_view writes; every one carries a
//   "meter" entry, and the page shows it beside the view.
// * The timeline lists the trace's steps on the reference run's validation-loss curve;
//   every panel follows it. Each tensor load shows progress.
//
// Stub: the gates' companion delta implements this page.
"use strict";
