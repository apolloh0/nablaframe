"""Phase 7 gate helpers: settings, bars, the clock's measurements, set's identities, requests and judgements (D54 to D58).

Contract: gates/phase7/_common.py. The settings are the spec's (D54, D57), so the gates retrace feasibility. The tests
that read a model use phase7_fixtures.py's toy models and need the jax lane.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT


def _common():
    gates = REPO_ROOT / "gates" / "phase7"
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location("phase7_common", gates / "_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # registered, as an import would, so its dataclasses resolve their annotations
    spec.loader.exec_module(module)
    return module


def plain(value):
    return json.loads(json.dumps(value)) == value


def test_settings_and_bars_are_fixed():
    c = _common()
    assert c.MODEL == {"vocab": 114, "d_model": 32, "n_layers": 1, "n_heads": 4, "d_mlp": 128, "seq": 3}
    assert c.TRAINING == {"batch": 512, "seq": 3, "lr": 1e-3, "warmup_steps": 10, "total_steps": 10000,
                          "min_lr_ratio": 0.1, "weight_decay": 1.0, "data_seed": 0, "param_seed": 0}
    assert (c.EVAL_SEED, c.PROBE_WINDOWS) == (2, 64)
    assert c.DICTIONARY == {"steps": 3000, "batch": 4096}
    assert c.NAMES == ("token(p+1)", "token(p-2)", "token(p-1)")
    assert c.SEED == 50 and c.IDENTITY_WINDOWS == 64
    assert c.JUDGED == (("induction", 1, "attention", "mechanism", "copied", "table"),
                        ("two_hop", 3, "write", "query", "hop", "table"),
                        ("modular_addition", 0, "write", "answer", "sum", "table"),
                        ("modular_addition", 0, "write", "answer", "sum", "fourier"))
    assert c.RECORDED == (("modular_addition", 0, "mlp", "answer", "sum and operands", "additive"),
                          ("modular_addition", 0, "attention", "answer", "operands", "additive"),
                          ("two_hop", 3, "attention", "query", "hop", "table"))
    assert c.BARS == {"held_accuracy": 0.99, "control_ratio": 2.0, "frequencies": 8, "logit_r2": 0.8, "kl": 0.25,
                      "ablation": 1.0, "wrong": 1.0, "reads_agreement": 0.8, "answer_agreement": 0.1,
                      "twin_within": 0.05}
    assert c.RUN_DIR == REPO_ROOT / "runs" / "phase7" and c.RECORD_DIR == c.RUN_DIR / "gestures"
    assert c.REPORT_DIR == REPO_ROOT / "reports" / "phase7"


def test_configs_checkpoints_and_probe():
    c = _common()
    from nablaframe.zoo.tasks import ModularAdditionTask

    steps = c.checkpoint_steps()
    assert steps == list(range(0, 3001, 100)) + list(range(3500, 10001, 500)) and len(steps) == 45
    assert all(type(s) is int for s in steps)
    for task in (ModularAdditionTask(), ModularAdditionTask(p=11)):
        held = task.held_out()
        got = c.probe(task)
        want = held[np.random.default_rng(2).choice(len(held), 64, replace=False)]
        assert got.shape == (64, 4) and np.array_equal(got, want), "the held-out pairs D54 draws, in that order"
    pytest.importorskip("jax")
    model, train = c.configs()
    assert {k: getattr(model, k) for k in c.MODEL} == c.MODEL
    assert {k: getattr(train, k) for k in c.TRAINING} == c.TRAINING
    assert (train.betas, train.eps, train.clip) == ((0.9, 0.95), 1e-8, 1.0), "every other field at its default"


def test_probe_passes_the_bitwise_oracle():
    """D60: G7.1's trace checks the probe with D6's bitwise oracle first, so the probe must pass it at the specimen's size."""
    pytest.importorskip("jax")
    c = _common()
    from nablaframe.model.gpt2 import GPT2Model, init_params
    from nablaframe.oracle.mode_equality import check_mode_equality
    from nablaframe.trace.registry import tap_names
    from nablaframe.zoo.tasks import ModularAdditionTask

    config, _ = c.configs()
    tokens = np.ascontiguousarray(c.probe(ModularAdditionTask())[:, :config.seq], dtype=np.int32)
    for seed in (0, 1):
        result = check_mode_equality(GPT2Model(config), init_params(config, seed=seed), tokens, tap_names(1), "bitwise")
        assert result.passed, f"seed {seed}: {result.reason}"


def _embedding(p, columns, rng, d_extra=2):
    """Number-token rows whose column j is ``r_j cos(2π k_j v / p + φ_j)``, so frequency k holds Σ r_j² over its columns."""
    v = np.arange(p)
    cols = [r * np.cos(2 * np.pi * k * v / p + rng.uniform(0, 2 * np.pi)) for k, r in columns]
    cols += [np.full(p, 3.0)] * d_extra  # a constant: no frequency's energy
    return np.stack(cols, axis=1)


def test_fewest_frequencies():
    c = _common()
    from nablaframe.gestures import crystallize

    rng = np.random.default_rng(0)
    E = _embedding(23, [(7, np.sqrt(5)), (2, np.sqrt(3)), (9, 1.0), (4, 1.0)], rng)
    got = c.fewest_frequencies(E, 23)
    assert got == {"count": 2, "held": pytest.approx(0.8)} and type(got["count"]) is int and plain(got)
    assert got["count"] == len(crystallize.frequencies(E, 23)["k"])
    assert got["held"] == pytest.approx(crystallize.frequencies(E, 23)["held"], rel=1e-12)

    flat = _embedding(29, [(k, 1.0) for k in range(1, 12)], rng)
    got = c.fewest_frequencies(flat, 29)
    assert got["count"] == 9 and got["held"] == pytest.approx(9 / 11), "no cut at MAX_FREQUENCIES"
    assert len(crystallize.frequencies(flat, 29)["k"]) == 8

    trained = rng.standard_normal((113, 32))
    want = crystallize.frequencies(trained, 113)
    got = c.fewest_frequencies(trained, 113)
    assert got["count"] > 8 and got["held"] >= 0.75, "a random embedding needs many: the twin's 39"
    for bad in ((E, 22), (E[:-1], 23), (E, 2)):
        with pytest.raises(ValueError):
            c.fewest_frequencies(*bad)
    assert want["held"] < 0.75


def _logits(windows, p, vocab, k, alpha, rng, *, noise=0.0):
    """Logits at position 2: Σ αₖ cos(2πk(a + b − c)/p) at each number token c, plus a constant per pair."""
    a, b = windows[:, 0].astype(np.int64), windows[:, 1].astype(np.int64)
    c = np.arange(p)
    out = rng.normal(0, 50, (len(windows), vocab))  # the "=" token's logit: nothing to fit
    phase = 2 * np.pi * ((a + b)[:, None] - c[None, :]) / p
    out[:, :p] = sum(al * np.cos(f * phase) for f, al in zip(k, alpha)) + rng.normal(0, 20, (len(windows), 1))
    out[:, :p] += noise * rng.standard_normal((len(windows), p))
    return out.astype(np.float32)


def test_logit_r2():
    c = _common()
    from nablaframe.zoo.tasks import ModularAdditionTask

    task = ModularAdditionTask(p=11, train_fraction=0.5)
    p, vocab = task.p, len(task.vocab)
    fit, held = task.training(), task.held_out()
    rng = np.random.default_rng(1)
    k, alpha = [2, 5], [3.0, -1.5]
    fit_logits, held_logits = _logits(fit, p, vocab, k, alpha, rng), _logits(held, p, vocab, k, alpha, rng)
    r2 = c.logit_r2(fit_logits, fit, held_logits, held, p, k)
    assert type(r2) is float and r2 > 0.999999, "the logits are the clock's, up to a constant per pair"
    assert c.logit_r2(fit_logits, fit, held_logits, held, p, [5, 2]) == pytest.approx(r2, abs=1e-9)
    noisy = c.logit_r2(fit_logits, fit, _logits(held, p, vocab, k, alpha, rng, noise=1.0), held, p, k)
    assert 0.5 < noisy < 0.95
    assert abs(c.logit_r2(fit_logits, fit, held_logits, held, p, [3])) < 0.05, "another frequency explains nothing"
    flipped = _logits(held, p, vocab, k, [-3.0, 1.5], rng)
    assert c.logit_r2(fit_logits, fit, flipped, held, p, k) == pytest.approx(-3.0, abs=1e-4), \
        "α is fitted on the fitting pairs and judged on the held-out pairs"
    shifted = held.copy()
    shifted[:, 1] = (held[:, 1] + 1) % p
    assert c.logit_r2(fit_logits, fit, held_logits, shifted, p, k) < 0.5, "a and b are the window's tokens at 0 and 1"
    flat = np.tile(np.arange(vocab, dtype=np.float32) * 0 + 2.0, (len(held), 1))
    assert c.logit_r2(fit_logits, fit, flat, held, p, k) is None, "no variance over c to explain"
    for bad in ((fit_logits, fit, held_logits, held, p, []), (fit_logits[:-1], fit, held_logits, held, p, k),
                (fit_logits, fit, held_logits[:, :p - 1], held, p, k)):
        with pytest.raises(ValueError):
            c.logit_r2(*bad)


class Spy:
    """The model, noting the taps each forward pass records; and, given ``spoil``, changing what one edit's pass gives."""

    def __init__(self, model, spoil=None, what="logits"):
        self.model, self.spoil, self.what, self.calls = model, spoil, what, []

    def instrumented(self, params, tokens, taps, interventions=None):
        import jax

        self.calls.append((tuple(taps), sorted(interventions or {})))
        logits, recorded = self.model.instrumented(params, tokens, taps, interventions)
        if self.spoil is not None and self.spoil in (interventions or {}):
            nudge = lambda tree: jax.tree_util.tree_map(lambda x: x * 1.0001 + 1e-6, tree)  # noqa: E731
            if self.what == "logits":
                logits = nudge(logits)
            else:
                recorded = dict(recorded, **{self.what: nudge(recorded[self.what])})
        return logits, recorded


def test_set_identities():
    pytest.importorskip("jax")
    c = _common()
    import phase7_fixtures as F
    from nablaframe.interventions.apply import editable_taps
    from nablaframe.trace.registry import tap_names

    w = F.world("modular_addition")
    tokens = w.held[:16, :3]
    want = [t for t in editable_taps(1) if not t.endswith(".attn.hook_z")]
    spy = Spy(w.model)
    got = c.set_identities(spy, w.params, tokens, n_layers=1)
    assert list(got) == want and all(v is True for v in got.values()), "set to its own values changes nothing"
    canonical = set(tap_names(1))
    assert all(set(taps) == canonical for taps, edits in spy.calls if edits), "every edited pass records every tap"
    assert any(set(taps) == canonical for taps, edits in spy.calls if not edits), "and so does the pass it is held to"
    assert {tuple(edits) for _, edits in spy.calls if edits} == {(t,) for t in want}, "one set at a time"

    for what in ("logits", "blocks.0.hook_resid_post"):
        spoiled = c.set_identities(Spy(w.model, spoil="blocks.0.hook_mlp_out", what=what), w.params, tokens, n_layers=1)
        assert spoiled == {t: t != "blocks.0.hook_mlp_out" for t in want}, f"a change in the {what} is seen"
    induction = F.world("induction")
    got = c.set_identities(induction.model, induction.params, induction.held[:8, :induction.config.seq], n_layers=3)
    assert len(got) == len(editable_taps(3)) - 3 and all(got.values())


def test_requests_and_their_hashes():
    c = _common()
    from nablaframe.gestures.records import request_hash

    a, b = "a" * 64, "b" * 64
    got = c.requests("e" * 64, [0, 100, 10000], (a, b))
    assert len(got) == 64 and got == c.requests("e" * 64, [0, 100, 10000], [a, b])
    assert all(r.kind == "crystallize" and r.instances == 8938 and r.run == "e" * 64 for r in got)
    assert [(r.step, list(r.args["cites"]), r.seed, r.args["block"]) for r in got[:16]] == [
        (step, cites, seed, block) for step in (0, 10000) for cites in ([], [a, b]) for seed in (50, 51)
        for block in (0, 1)]
    bases = [{k: v for k, v in got[16 * i].args.items() if k not in ("cites", "block")} for i in range(4)]
    assert bases == [{"part": "write", "class": "answer", "function": "sum", "form": "table"},
                     {"part": "write", "class": "answer", "function": "sum", "form": "fourier"},
                     {"part": "mlp", "class": "answer", "function": "sum and operands", "form": "additive"},
                     {"part": "attention", "class": "answer", "function": "operands", "form": "additive"}]
    assert c.request_hashes(got) == {"checked": 64, "distinct": True, "stable": True}
    assert c.request_hashes(got + got[:1])["distinct"] is False
    assert len({request_hash(r) for r in got}) == 64


def _results(*, kl=0.02, ablation=4.1, wrong=(5.8, 5.9), cf=None, twin_kl=0.5, twin_wrong=(0.5005, 0.498),
             rows=((0, "inert"), (100, "holds"), (200, "holds")), twin=True):
    return {
        "meter": {"label": "holds", "kl": kl, "ablation": ablation, "removed": 0.99, "agreement": 0.98,
                  "loss_delta": -0.03},
        "wrong": [{"key": f"k{i}", "kl": x, "agreement": 0.1} for i, x in enumerate(wrong)],
        "counterfactual": cf,
        "twin": {"meter": {"label": "inert", "kl": twin_kl, "ablation": 0.4}, "wrong": [
            {"key": f"k{i}", "kl": x, "agreement": 0.5} for i, x in enumerate(twin_wrong)], "counterfactual": None}
        if twin else None,
        "checkpoints": None if rows is None else [{"step": s, "kl": 0.1, "ablation": 2.0, "label": label,
                                                   "accuracy": 0.5} for s, label in rows],
    }


def test_crystal_checks_and_judged():
    c = _common()
    cf = {"name": "broken", "rows": 24, "reads": {"kl": 0.01, "agreement": 0.94},
          "answer": {"kl": 9.3, "agreement": 0.008}}
    got = c.crystal_checks(_results())
    assert got == {"kl": 0.02, "ablation": 4.1, "wrong_min": 5.8, "reads_agreement": None, "answer_agreement": None,
                   "twin_within": pytest.approx(0.002), "final_holds": True, "step0_outside": True} and plain(got)
    got = c.crystal_checks(_results(cf=cf, twin_wrong=(0.46, 0.51), wrong=(6.9, 6.3)))
    assert (got["reads_agreement"], got["answer_agreement"]) == (0.94, 0.008)
    assert got["twin_within"] == pytest.approx(0.04) and got["wrong_min"] == 6.3, "the greatest distance, either way"
    assert c.crystal_checks(_results(rows=((0, "inert"), (100, "holds"), (200, "doesn't hold"))))["final_holds"] is False
    assert c.crystal_checks(_results(rows=((0, "holds"), (100, "holds"))))["step0_outside"] is False
    assert c.crystal_checks(_results(rows=((100, "inert"), (200, "holds"))))["step0_outside"] is False, \
        "step 0 must be read to be outside"
    assert c.crystal_checks(_results(rows=((0, "doesn't hold"), (200, "holds"))))["step0_outside"] is True
    none = c.crystal_checks(_results(wrong=(), twin_wrong=()))
    assert none["wrong_min"] is None and none["twin_within"] is None
    for bad in (_results(twin=False), _results(rows=None), _results(rows=())):
        with pytest.raises(ValueError):
            c.crystal_checks(bad)

    checks = [c.crystal_checks(_results(kl=0.134, ablation=4.58, wrong=(4.61, 4.62), cf=cf)),
              c.crystal_checks(_results(kl=0.0, ablation=6.94, wrong=(6.73,),
                                        cf=dict(cf, reads={"kl": 0.0, "agreement": 1.0},
                                                answer={"kl": 13.5, "agreement": 0.0}))),
              c.crystal_checks(_results(kl=0.021, ablation=4.14, wrong=(5.77, 5.94, 5.9), twin_wrong=(0.503, 0.499)))]
    got = c.judged(checks)
    assert got == {"kl_max": 0.134, "ablation_min": 4.14, "wrong_min": 4.61, "reads_agreement_min": 0.94,
                   "answer_agreement_max": 0.008, "twin_within_max": pytest.approx(0.003), "envelope_ok": True}
    assert plain(got) and type(got["kl_max"]) is float
    assert c.judged(checks[2:])["reads_agreement_min"] is None, "none with a counterfactual: nothing to judge"
    failing = c.crystal_checks(_results(rows=((0, "holds"), (100, "holds"))))
    assert c.judged(checks + [failing])["envelope_ok"] is False
    assert c.judged(checks + [c.crystal_checks(_results(wrong=(), twin_wrong=()))])["wrong_min"] is None, \
        "a judged reading that wasn't made fails its bar"
    with pytest.raises(ValueError):
        c.judged([])
