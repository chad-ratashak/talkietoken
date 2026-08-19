"""Cross-model encoding-penalty matrix: every tokenizer as target, every
other as reference.

Why this exists
---------------
`results/reconstitution-penalty.txt` reports one number — talkie's normalised
encoding penalty on Python — computed for one target against one reference
(o200k). The predictor the whole H1b argument rests on is a single cell of a
matrix nobody had drawn. Two things it cannot tell you on its own:

1. **Is the number a property of talkie, or of the reference?** If talkie's
   penalty moves when o200k is swapped for another modern tokenizer, the
   published figure is a coordinate, not a measurement.

2. **Does the metric discriminate at all?** A high penalty is supposed to
   mean "this model must spell the capability out, because its corpus lacked
   it." Run the same statistic on a model that unambiguously *has* Python in
   its training data. If that model scores as high as talkie, the statistic
   is measuring tokenizer design, not corpus absence.

That second run is the negative control the predictor never had.
METHODS.md § 8 calls cross-model "the design H1b is really for, and the
reason the harness is model-agnostic"; this is that design, restricted to
the tokenizer half, which needs no generation and no network.

No generation, no code execution, no API spend: this is tokenization of a
few hundred characters, 30 pairs deep.

    uv run python reconstitution/run_matrix.py
"""

import os

# Offline by default, set before transformers is imported anywhere. The
# tokenizers this loads are already-cached data files; nothing here should
# reach the network, and a silent download would break that claim without
# anyone noticing. --allow-download opts out.
if "--allow-download" not in os.sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reconstitution"))

import capabilities as C  # noqa: E402
import curve as V  # noqa: E402
import penalty as P  # noqa: E402

# The six tokenizers, oldest era first, talkie last as the model under test.
#
# Model-origin policy (tokenizer-only/baselines.py): U.S./allied vendors
# only. gpt2/cl100k/o200k are OpenAI; gemma-2-2b is Google; Llama-3.2 is
# Meta, all U.S. vendors, all run locally. Other tokenizers (Qwen, DeepSeek,
# GLM, Yi) are excluded under the model-origin rule (US/allied allowlist).
#
# The unsloth mirrors are byte-identical re-uploads of the gated originals
# over the core vocabulary; talkietoken-urdu.py verifies that by sha256
# across three mirrors each.
SPECS = [
    ("gpt2", "tiktoken:gpt2"),
    ("cl100k", "tiktoken:cl100k_base"),
    ("o200k", "tiktoken:o200k_base"),
    ("gemma-2-2b", "hf:unsloth/gemma-2-2b"),
    ("Llama-3.2-3B", "hf:unsloth/Llama-3.2-3B"),
    ("talkie-1930", "talkie"),
]

# The model whose corpus is in question.
UNDER_TEST = "talkie-1930"

# Modern, code-fluent tokenizers whose training data unambiguously contained
# Python. If the statistic is measuring corpus absence they should score
# ~1.0; whatever they actually score is the floor any claim about talkie has
# to clear. gpt2 is excluded — it is the positive control, a tokenizer that
# really is worse at Python.
NEG_CONTROLS = ("cl100k", "o200k", "gemma-2-2b", "Llama-3.2-3B")

# What each vocab_size actually counts. run_curve.py prints all three under
# the label "merges"; none of them is a merge count, and with HF in the set
# there are now three different definitions in one table.
VOCAB_MEANING = {
    "talkie": "mergeable ranks, incl. the EOT special",
    "tiktoken:": "n_vocab: max rank + 1, incl. specials",
    "hf:": "len(tokenizer): base vocab + added tokens",
}


def vocab_note(spec):
    for prefix, note in VOCAB_MEANING.items():
        if spec == prefix or spec.startswith(prefix):
            return note
    return "?"


def load_all(emit):
    """Load every tokenizer, failing with a message that names the fix."""
    toks = {}
    for name, spec in SPECS:
        try:
            toks[name] = P.load_tokenizer(spec)
        except Exception as exc:  # noqa: BLE001 — the message is the point
            raise SystemExit(
                f"\ncould not load {name} ({spec}):\n  {type(exc).__name__}: {exc}\n\n"
                "This run is offline by design. If the repo is not in the "
                "local Hugging Face cache, fetch it once with:\n"
                f"  uv run python reconstitution/run_matrix.py --allow-download\n"
            ) from exc
        emit(f"  {name:<14} {toks[name].vocab_size:>8,}  {spec}")
    return toks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capability", default="python")
    ap.add_argument("--control", default="period-prose", choices=sorted(P.CONTROLS))
    ap.add_argument("--allow-download", action="store_true",
                    help="permit Hugging Face downloads (default: offline)")
    ap.add_argument("--out", default="results/penalty-matrix")
    a = ap.parse_args()

    out = []

    def emit(s=""):
        out.append(s)

    cap = C.load_capability(a.capability)
    control = P.CONTROLS[a.control]
    dialects = list(cap.dialects)
    preserving = [d for d in dialects if d in C.CONTENT_PRESERVING]

    emit("=== CROSS-MODEL ENCODING PENALTY MATRIX ===")
    emit("Every tokenizer as target, every other as reference. The published")
    emit("figure in results/reconstitution-penalty.txt is one cell of this.")
    emit()
    emit("The question is not 'what is talkie's penalty'. It is whether that")
    emit("number is a property of talkie or an artefact of picking o200k, and")
    emit("whether a model that demonstrably HAS Python scores any lower.")
    emit()

    emit("--- provenance ---")
    stamp = V.git_stamp()
    emit(f"  generated        {V.time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    emit(f"  git              {stamp['commit']} on {stamp['branch']}"
         f"{' (dirty)' if stamp['dirty'] else ''}")
    emit(f"  capability       {cap.name}")
    emit(f"  control prose    {a.control}")
    emit(f"  dialects         {len(dialects)} ({', '.join(dialects)})")
    emit(f"  content-preserv. {len(preserving)} ({', '.join(preserving)})")
    emit(f"  network          {'ALLOWED' if a.allow_download else 'off (HF_HUB_OFFLINE=1)'}")
    emit("  generation       none. Tokenizer measurement only.")
    emit()

    emit("--- tokenizers ---")
    emit(f"  {'name':<14} {'vocab':>8}  spec")
    toks = load_all(emit)
    emit()
    emit("  'vocab' is not one quantity across families:")
    for name, spec in SPECS:
        emit(f"    {name:<14} {vocab_note(spec)}")
    emit()

    names = [n for n, _ in SPECS]

    # --- the measurements ------------------------------------------------
    # cells[(target, ref)][dialect] = Penalty
    cells, norm_mean, per_dialect = {}, {}, {}
    for tname in names:
        for rname in names:
            if tname == rname:
                continue
            pens = {d: P.penalty_of(toks[tname], toks[rname],
                                    cap.penalty_material(d), control, d)
                    for d in dialects}
            cells[(tname, rname)] = pens
            per_dialect[f"{tname}|{rname}"] = {
                d: {"target_tokens": p.material.target_tokens,
                    "ref_tokens": p.material.ref_tokens,
                    "chars": p.material.chars,
                    "ratio": p.material.ratio,
                    "normalized": p.normalized,
                    "tok_per_char": p.material.tok_per_char,
                    "tok_per_char_excess": p.tok_per_char_excess}
                for d, p in pens.items()}
            norm_mean[(tname, rname)] = (
                sum(pens[d].normalized for d in preserving) / len(preserving))

    # --- matrix ----------------------------------------------------------
    emit("--- NORMALISED PENALTY (mean over content-preserving dialects) ---")
    emit("target tokens per reference token on Python demos, divided by the")
    emit("same pair's ratio on control prose. 1.0 = 'costs what my own")
    emit("ordinary text costs'. Rows are the model under test.")
    emit()
    emit(f"  {'target \\ ref':<14}" + "".join(f"{n:>14}" for n in names))
    for tname in names:
        row = "".join(
            f"{'—':>14}" if tname == rname else f"{norm_mean[(tname, rname)]:>14.3f}"
            for rname in names)
        emit(f"  {tname:<14}{row}")
    emit()

    # --- reference sensitivity -------------------------------------------
    emit("--- REFERENCE SENSITIVITY (how much the reference choice moves it) ---")
    emit("A target whose row is flat has a penalty that is its own property.")
    emit("A target whose row swings is being described by its reference.")
    emit()
    emit(f"  {'target':<14}{'min':>9}{'max':>9}{'spread':>9}   argmin / argmax")
    spread = {}
    for tname in names:
        vals = {r: norm_mean[(tname, r)] for r in names if r != tname}
        lo_r, hi_r = min(vals, key=vals.get), max(vals, key=vals.get)
        spread[tname] = {"min": vals[lo_r], "max": vals[hi_r],
                         "argmin": lo_r, "argmax": hi_r,
                         "spread": vals[hi_r] - vals[lo_r]}
        emit(f"  {tname:<14}{vals[lo_r]:>9.3f}{vals[hi_r]:>9.3f}"
             f"{vals[hi_r] - vals[lo_r]:>9.3f}   {lo_r} / {hi_r}")
    emit()

    # --- reference-free measure ------------------------------------------
    # tok_per_char_excess is material tok/char over control tok/char, both
    # measured on the TARGET alone. It does not involve the reference at
    # all, so it is a column, not a matrix — that is exactly why METHODS.md
    # § 5 offers it for the case where the reference is also blind.
    emit("--- TOK/CHAR EXCESS (reference-free; a column, not a matrix) ---")
    emit("material tokens-per-char over control tokens-per-char, target only.")
    emit("Immune to the reference choice above. NOT on the same scale as the")
    emit("normalised penalty: material is code and control is prose, so their")
    emit("character densities differ for reasons unrelated to encoding.")
    emit()
    emit(f"  {'target':<14}" + "".join(f"{d.replace('py-', ''):>12}" for d in preserving))
    tpc = {}
    for tname in names:
        any_ref = next(r for r in names if r != tname)
        vals = {d: cells[(tname, any_ref)][d].tok_per_char_excess for d in preserving}
        tpc[tname] = vals
        emit(f"  {tname:<14}" + "".join(f"{vals[d]:>12.3f}" for d in preserving))
    emit()

    # --- the read ---------------------------------------------------------
    emit("--- HOW TO READ THIS ---")
    emit("  gpt2 is the POSITIVE control: a tokenizer genuinely worse at")
    emit("    Python (2019, 50k, web text). It should score high. If it does")
    emit("    not, the statistic is broken outright.")
    emit("  cl100k / Llama-3.2 / gemma-2-2b are NEGATIVE controls: current,")
    emit("    code-fluent, Python unambiguously in the training data. They")
    emit("    should score ~1.0 if the statistic measures corpus absence.")
    emit("  talkie-1930 is the model under test.")
    emit()
    emit("  The predictor discriminates only if talkie sits ABOVE the")
    emit("  negative-control band under a common reference. Read the talkie")
    emit("  row against the cl100k / Llama / gemma rows, one column at a time.")
    emit()

    # --- the pre-registered decision -------------------------------------
    # Fixed before the numbers were seen: under each reference, compare the
    # model under test against the WORST negative control. A separation
    # smaller than the spread among the negative controls themselves is not
    # a signal about talkie's corpus — it is the statistic's own noise.
    emit("--- DOES IT DISCRIMINATE? (pre-registered) ---")
    emit("Per reference: talkie's margin over the worst negative control,")
    emit("against the spread among the negative controls themselves. A margin")
    emit("smaller than that spread is not evidence about talkie's corpus.")
    emit()
    emit(f"  {'reference':<14}{'talkie':>9}{'neg ctrl band':>16}{'margin':>9}"
         f"{'band':>8}   verdict")
    verdict = {}
    for rname in names:
        if rname == UNDER_TEST:
            continue
        negs = {n: norm_mean[(n, rname)] for n in NEG_CONTROLS if n != rname}
        # Collapse controls that are not independent. Llama-3.2 and cl100k
        # return identical counts on this material, so a band computed
        # across both is one tokenizer measured twice, and comes out
        # near-zero — which would read as "separates" for the wrong reason.
        distinct, seen = {}, {}
        for n in negs:
            key = tuple(cells[(n, rname)][d].material.target_tokens for d in dialects)
            if key in seen:
                continue
            seen[key] = n
            distinct[n] = negs[n]
        dropped = [n for n in negs if n not in distinct]
        t = norm_mean[(UNDER_TEST, rname)]
        lo, hi = min(distinct.values()), max(distinct.values())
        margin = t - hi
        if len(distinct) < 2:
            ok, band, note = None, float("nan"), "n/a: <2 independent controls"
        else:
            band = hi - lo
            ok = margin > band
            note = "separates" if ok else "NOT separated"
        verdict[rname] = {"talkie": t, "neg_lo": lo, "neg_hi": hi,
                          "margin": margin, "band": band, "separates": ok,
                          "controls_used": sorted(distinct),
                          "controls_dropped_as_duplicates": sorted(dropped)}
        band_s = "     —" if ok is None else f"{band:>6.3f}"
        emit(f"  {rname:<14}{t:>9.3f}{f'{lo:.3f}-{hi:.3f}':>16}{margin:>+9.3f}"
             f"{band_s:>8}   {note}"
             + (f"  [dropped {', '.join(dropped)}]" if dropped else ""))
    emit()
    scored = [v for v in verdict.values() if v["separates"] is not None]
    n_ok = sum(v["separates"] for v in scored)
    emit(f"  separates under {n_ok} of {len(scored)} scorable references "
         f"({len(verdict) - len(scored)} unscorable for lack of independent controls).")
    emit()
    emit("  The row that decides it is o200k: that is the reference the")
    emit("  published figure uses. The other rows are robustness.")
    emit()
    emit("  Caveat on the gemma-2-2b row: a tokenizer cannot be its own")
    emit("  control, so gemma leaves the control set exactly when it is the")
    emit("  reference — and gemma is the control that varies most. What is")
    emit("  left (cl100k, o200k) happens to agree closely, so the band")
    emit("  collapses and the margin clears it. That row reads 'separates'")
    emit("  because the widest control was removed, not because talkie stood")
    emit("  out. Do not count it as corroboration.")
    emit()
    emit("  Note: Llama-3.2 and cl100k return identical counts on English and")
    emit("  Python — Llama-3's 128k vocab embeds cl100k's 100k tokens. They")
    emit("  are distinct tokenizers (they diverge on Urdu and Hindi), but as")
    emit("  a Python reference Llama adds no information cl100k did not.")
    emit("  Treat that column as a duplicate, not as corroboration.")
    emit()

    payload = {
        "meta": {
            "generated": V.time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "git": stamp,
            "capability": cap.name,
            "control": a.control,
            "dialects": dialects,
            "content_preserving": preserving,
            "network": "allowed" if a.allow_download else "offline",
            "generation": None,
            "tokenizers": {
                n: {"spec": s, "label": toks[n].label,
                    "vocab_size": toks[n].vocab_size,
                    "vocab_size_means": vocab_note(s)}
                for n, s in SPECS},
        },
        "normalized_mean": {f"{t}|{r}": v for (t, r), v in norm_mean.items()},
        "reference_sensitivity": spread,
        "tok_per_char_excess": tpc,
        "per_dialect": per_dialect,
    }

    txt = ROOT / f"{a.out}.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("\n".join(out) + "\n")
    V.dump_json(ROOT / f"{a.out}.json", payload)
    print("\n".join(out))
    print(f"\nwrote {a.out}.{{txt,json}}")


if __name__ == "__main__":
    main()
