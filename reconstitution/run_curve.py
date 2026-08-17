"""CLI: measure in-context capability reconstitution and write the results.

    # tokenizer only — no network, no code execution, always runnable
    uv run python reconstitution/run_curve.py --penalty-only

    # the harness validating its own maths against a known ground truth
    uv run python reconstitution/run_curve.py --backend synthetic --allow-exec

    # the real thing, against the only channel available without a GPU
    uv run python reconstitution/run_curve.py --backend talkie-chat \\
        --strata format-control,oneline,humaneval-style --allow-exec

    # what would actually settle the question
    uv run python reconstitution/run_curve.py \\
        --backend local:talkie-lm/talkie-1930-13b-base --allow-exec

Writes <out>.txt and <out>.json. The prose blocks the report prints live in
NOTES at the top of this file, apart from the layout code, so wording can be
edited without reading the formatting — if you are here to change how a
number is computed, it is in curve.py or statutils.py, not here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import capabilities as C  # noqa: E402
import curve as V  # noqa: E402
import penalty as P  # noqa: E402
import statutils as S  # noqa: E402
from backends import load_backend  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

NOTES = {
    "header": "=== IN-CONTEXT CAPABILITY RECONSTITUTION ===",
    "purpose": [
        "Measures whether a capability absent from a model's training data can",
        "be rebuilt from in-context demonstrations, as a curve over how much",
        "demonstration material is supplied — not as a single pass@k point.",
        "",
        "Why the shape matters rather than the level: methods that remove a",
        "capability from weights (data filtering; module ablation such as GRAM)",
        "claim the capability is gone. If context alone rebuilds it, removal from",
        "weights is necessary but not sufficient. The curve is what distinguishes",
        "a robust removal from a cosmetic one.",
        "",
        "talkie is the calibration case, not the point: it is the one model where",
        "ground truth about the corpus is checkable independently, because a BPE",
        "merge is a fossil of training data and in-context learning cannot create",
        "one. See RESULTS.md and results/python-leak.txt.",
    ],
    "penalty": [
        "The encoding penalty is the predictor. If the tokenizer has no merges",
        "for a capability's surface form, each demonstration costs more tokens,",
        "so a fixed context window holds fewer of them. Prediction: higher",
        "penalty, less recovery per token of context.",
        "",
        "'ratio' is target tokens per reference token, the raw Tier 4 measure.",
        "'normalised' divides that by the same model's ratio on control prose, so",
        "a small vocabulary is not mistaken for a capability-specific penalty.",
        "Dialects hold the lesson fixed and move only the token cost.",
    ],
    "curve": [
        "Every rate carries its raw numerator, its denominator, and an interval.",
        "pass@k is the Chen et al. unbiased estimator; its interval is a",
        "problem-level (cluster) bootstrap, because the k draws for one problem",
        "share a prompt and are not independent. 'solved>=1' is a plain",
        "proportion over problems, so that one gets a Wilson interval.",
        "",
        "pass@k is never extrapolated above the k actually sampled.",
    ],
    "format_control_ok": [
        "FORMAT CONTROL PASSED. The channel does transmit few-shot pattern",
        "induction, so a zero in a code stratum is about code.",
    ],
    "format_control_fail": [
        "*** FORMAT CONTROL FAILED ***",
        "The model could not complete a trivial English pattern from the same",
        "few-shot channel. Every code number below is therefore uninterpretable:",
        "a zero may mean the capability is absent OR that this channel does not",
        "carry few-shot demonstrations at all. Fix the channel before reading",
        "anything into the code strata.",
    ],
    "axis": [
        "Which x-axis do the dialect curves coincide on? The axis with the",
        "smallest mean spread across dialects is the currency reconstitution is",
        "actually bought with. Descriptive only — with a handful of dialects the",
        "differences are usually inside the points' own intervals.",
    ],
    "hypothesis": [
        "Two separable claims, tested apart because a single test across all",
        "dialects mixes them and returns noise:",
        "  H1a MECHANISM  at a matched token budget, demonstrations that cost",
        "                 more tokens each deliver fewer lessons, so recovery",
        "                 is lower. All dialects admissible.",
        "  H1b PROXY      the tokenizer-derived penalty — computable with no",
        "                 generation at all — predicts that difficulty. Only",
        "                 content-preserving dialects admissible: py-nodoc is",
        "                 cheap because it teaches less, not because it encodes",
        "                 better, and including it would let a content effect",
        "                 masquerade as an encoding effect.",
        "Spearman rho against a permutation null. The |rho| needed for",
        "significance at each group count is computed by exact enumeration and",
        "printed: at n=5 only a perfect ordering can reach p<=0.05, and at",
        "n<=4 nothing can. An unresolved test is reported as unresolved, never",
        "as a null.",
    ],
    "leverage": [
        "At a matched token budget that buys only that much difference in",
        "demonstrations delivered, so H1b on dialects alone is intrinsically",
        "low-powered. Higher-leverage designs, in order of preference:",
        "  cross-language — the Tier 5 spectrum (Python vs SQL vs J vs APL)",
        "    spans multiple-fold penalties, not fractions of one",
        "  cross-model — several filtered or ablated models with genuinely",
        "    different penalties for the same capability. This is what H1b is",
        "    really for, and why the harness is model-agnostic.",
    ],
}


def emit_to(out):
    def emit(s=""):
        print(s)
        out.append(s)

    return emit


def render(payload, emit):
    meta = payload["meta"]
    emit(NOTES["header"])
    for ln in NOTES["purpose"]:
        emit(ln)
    emit()
    emit("--- provenance ---")
    emit(f"  generated        {meta['generated']}")
    emit(f"  git              {meta['git']['commit']} on {meta['git']['branch']}"
         f"{' (dirty)' if meta['git']['dirty'] else ''}")
    emit(f"  capability       {meta['capability']}")
    emit(f"  target tokenizer {meta['target_tokenizer']['label']} "
         f"({meta['target_tokenizer']['vocab_size']:,} merges)")
    emit(f"  reference        {meta['reference_tokenizer']['label']} "
         f"({meta['reference_tokenizer']['vocab_size']:,} merges)")
    emit(f"  control prose    {meta.get('control', '-')}")
    if meta.get("humaneval_jsonl"):
        emit(f"  HumanEval set    {meta['humaneval_jsonl']} (real problems)")
    else:
        emit("  task set         hand-written, HumanEval-STYLE. Numbers here are")
        emit("                   NOT comparable to a published HumanEval pass@k;")
        emit("                   pass --humaneval-jsonl for that.")
    emit()
    emit("--- generation channel ---")
    emit(f"  backend          {meta['backend']}")
    emit(f"  variant          {meta['backend_variant']}")
    if meta.get("generations") is not None:
        emit(f"  generations      {meta['generations']}"
             + (f" ({meta['rate_limit_hits']} rate-limit backoffs)"
                if meta.get("rate_limit_hits") else ""))
        if meta.get("cache"):
            emit(f"  cache            {meta['cache']['hits']} hits, "
                 f"{meta['cache']['misses']} misses")
    emit("  CAVEAT:")
    for ln in _wrap(meta["CONTAMINATION"], 68):
        emit(f"    {ln}")
    emit()

    if payload.get("reference_point"):
        rp = payload["reference_point"]
        emit("--- what the model's authors report (primary source) ---")
        for ln in rp["lines"]:
            emit(f"  {ln}")
        emit()

    emit("--- ENCODING PENALTY (tokenizer only; no generation) ---")
    for ln in NOTES["penalty"]:
        emit(ln)
    emit()
    emit(f"  {'dialect':<14}{'demo tok':>9}{'ref tok':>9}{'ratio':>8}"
         f"{'normalised':>12}{'tok/char':>10}")
    for d, pen in payload["penalty"].items():
        emit(f"  {d:<14}{pen['target_tokens']:>9}{pen['ref_tokens']:>9}"
             f"{pen['ratio']:>8.3f}{pen['normalized']:>12.3f}"
             f"{pen['tok_per_char']:>10.3f}")
    emit()
    ctrl = payload["penalty_control"]
    emit(f"  control prose under the same pair: {ctrl['target_tokens']} target / "
         f"{ctrl['ref_tokens']} reference = {ctrl['ratio']:.3f}")
    emit()

    if not payload["curves"]:
        emit("--- no generation run (--penalty-only) ---")
        emit("  The penalty above is a prediction. Nothing here measures whether")
        emit("  it holds; run with a --backend to sweep the curve.")
        return

    emit("--- RECOVERY CURVES ---")
    for ln in NOTES["curve"]:
        emit(ln)
    for key, pts in payload["curves"].items():
        emit()
        emit(f"  [{key}]")
        emit(f"    {'demos':>6}{'demo tok':>9}{'gens':>6}{'correct':>8}"
             f"  {'solved>=1 of tasks':<34}{'pass@1':>22}")
        for p in pts:
            sr, lo, hi = p["solve_rate"]
            solved = (f"{p['tasks_solved']}/{p['n_tasks']} = {100 * sr:.1f}% "
                      f"[{100 * lo:.1f},{100 * hi:.1f}]")
            p1 = p["pass_at"].get("1")
            p1s = ("-" if not p1 else
                   f"{100 * p1['mean']:.2f}% [{100 * p1['lo']:.2f},"
                   f"{100 * p1['hi']:.2f}]")
            emit(f"    {p['n_demos']:>6}{p['demo_tokens_target']:>9.0f}"
                 f"{p['n_generations']:>6}{p['n_correct']:>8}  {solved:<34}"
                 f"{p1s:>22}")
        emit(f"    pass@k at the largest budget ({pts[-1]['n_demos']} demos):")
        for kk in sorted(pts[-1]["pass_at"], key=int):
            e = pts[-1]["pass_at"][kk]
            if e["degenerate"]:
                emit(f"      pass@{kk}: 0% — nothing solved; interval degenerate")
            else:
                emit(f"      pass@{kk}: {100 * e['mean']:.2f}% "
                     f"[{100 * e['lo']:.2f}, {100 * e['hi']:.2f}] 95% bootstrap")
        strat = pts[-1]["strategies"]
        emit(f"    answer extraction at that budget: "
             + ", ".join(f"{k}={v}" for k, v in sorted(strat.items())))
        if pts[-1]["failures"]:
            emit(f"    top failure reasons: "
                 + ", ".join(f"{k}={v}" for k, v in pts[-1]["failures"].items()))
    emit()

    fc = payload["analysis"].get("format_control")
    if fc is not None:
        emit("--- format control ---")
        emit(f"  {fc['summary']}")
        for ln in NOTES["format_control_ok" if fc["passed"] else
                       "format_control_fail"]:
            emit(f"  {ln}")
        emit()

    for stratum, entry in (payload["analysis"].get("by_stratum") or {}).items():
        emit(f"=== ANALYSIS: {stratum} ===")
        emit()
        _render_stratum(entry, emit)

    cost = payload["analysis"].get("cost")
    if cost:
        emit("--- what a conclusive run would cost ---")
        emit(f"  {cost['note']}")
        for ln in cost["lines"]:
            emit(f"  {ln}")
        emit()


def _render_stratum(entry, emit):
    al = entry.get("axis_alignment")
    if al:
        emit("--- which axis is the curve a function of? ---")
        for ln in NOTES["axis"]:
            emit(ln)
        emit()
        emit(f"  {'axis':<22}{'shared range':>26}{'mean spread':>14}")
        for axis, info in al.items():
            rng = ("-" if not info.get("overlap") else
                   f"[{info['overlap'][0]:.0f}, {info['overlap'][1]:.0f}]")
            sp = ("-" if info.get("mean_spread") is None
                  else f"{100 * info['mean_spread']:.2f} pp")
            emit(f"  {axis:<22}{rng:>26}{sp:>14}")
        ranked = [(i["mean_spread"], a) for a, i in al.items()
                  if i.get("mean_spread") is not None]
        if len(ranked) >= 2:
            ranked.sort()
            emit(f"  smallest spread: {ranked[0][1]}")
        emit()

    h = entry.get("penalty_vs_recovery")
    if h:
        emit("--- H1: does the encoding penalty predict recovery difficulty? ---")
        for ln in NOTES["hypothesis"]:
            emit(ln)
        if h.get("matched_budget_target_tokens"):
            emit()
            emit(f"  matched at {h['matched_budget_target_tokens']:.0f} target "
                 f"tokens of demonstrations, scored on pass@{h.get('k', 1)}")
        for key, title, pred in (
            ("H1a_tokens_per_demo", "H1a MECHANISM — target tokens per "
             "demonstration", "tok/demo"),
            ("H1b_encoding_penalty", "H1b PROXY — tokenizer-derived normalised "
             "penalty", "penalty"),
        ):
            r = h.get(key) or {}
            emit()
            emit(f"  {title}")
            if not r.get("testable"):
                emit(f"    NOT TESTABLE: {r.get('reason', 'no result')}")
                continue
            emit(f"    {'dialect':<14}{pred:>10}{'pass@1':>12}")
            for d, x, y in zip(r["dialects"], r["predictor"], r["recovery"]):
                emit(f"    {d:<14}{x:>10.3f}{100 * y:>11.2f}%")
            emit(f"    Spearman rho = {r['spearman_rho']:+.3f} "
                 f"(predicted {r['predicted_sign']}; direction "
                 f"{'matches' if r['direction_matches'] else 'DOES NOT match'})")
            emit(f"    permutation p = {r['permutation_p']:.4f} on n="
                 f"{r['n_dialects']} dialects")
            need = r.get("rho_needed_for_significance")
            if need is None:
                emit(f"    NOT RESOLVABLE: at n={r['n_dialects']} no rank")
                emit("    correlation whatsoever can reach p<=0.05. This line is")
                emit("    a direction, and cannot become a finding at this n.")
            elif r["underpowered"]:
                emit(f"    UNDERPOWERED: p<=0.05 needs |rho| >= {need:.3f} at "
                     f"n={r['n_dialects']};")
                emit(f"    observed |rho| = {abs(r['spearman_rho']):.3f}. Not a "
                     "null result — an unresolved one.")
            else:
                emit(f"    resolved: |rho| >= {need:.3f} was required at "
                     f"n={r['n_dialects']}")
        lev = entry.get("leverage")
        if lev:
            emit()
            emit(f"  LEVERAGE: content-preserving dialects span "
                 f"{lev['min']:.1f}-{lev['max']:.1f} tokens/demo = "
                 f"{lev['ratio']:.2f}x.")
            for ln in NOTES["leverage"]:
                emit(f"  {ln}")
        emit()


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def reference_point():
    """The primary-source anchor, read from the recorded figure data."""
    import json

    path = Path(__file__).resolve().parent / "reference" / "talkie-figure3.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    h = d["headline"]
    return {"lines": [
        f"source: {d['provenance']['primary_source_url']} "
        f"(Figure 3, {d['provenance']['source_kind']})",
        "The report states NO numeric pass@k in prose. The numbers below are the",
        "series its own chart plots, read from the page's chart source.",
        f"largest vintage model ({h['largest_vintage_model']['training_flops']:.3g}"
        f" FLOPs, inferred to be {h['largest_vintage_model']['identified_as']}):",
        f"    HumanEval pass@100 = {h['largest_vintage_model']['pass_at_100']:.5f}"
        f" ({h['largest_vintage_model']['pass_at_100_pct']:.3f}%)",
        f"matched modern control ({h['matched_modern_control']['identified_as']}):",
        f"    HumanEval pass@100 = {h['matched_modern_control']['pass_at_100']:.5f}"
        f" ({h['matched_modern_control']['pass_at_100_pct']:.3f}%)"
        f"  -> {h['modern_over_vintage']:.2f}x the vintage rate",
        "no error bars, no n, no demonstration count published; the vintage",
        "series is not monotone (see reference/talkie-figure3.json).",
    ]}


def _tokens_per_demo(points):
    """Mean target tokens one demonstration costs, over the non-zero budgets.
    The direct mechanism predictor for H1a."""
    vals = [p.demo_tokens_target / p.n_demos for p in points if p.n_demos]
    return sum(vals) / len(vals) if vals else float("nan")


def cost_note(published_rate=0.03215, n_problems=164):
    lines = []
    for hw in (0.02, 0.01, 0.005):
        n = S.required_n(published_rate, hw)
        lines.append(f"to pin a {100 * published_rate:.2f}% rate to +/-"
                     f"{100 * hw:.1f} pp (95% Wilson): {n:,} problem-level trials")
    lines.append(f"at {n_problems} problems that is "
                 f"{S.required_n(published_rate, 0.01) / n_problems:.0f}"
                 " samples per problem for the +/-1.0 pp row")
    lines.append("pass@100 needs >= 100 samples per problem regardless of the")
    lines.append("  interval: the estimator does not extrapolate upward")
    return {"note": "Rates near a few percent need thousands of trials to resolve.",
            "lines": lines}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default=None,
                    help="talkie-chat | synthetic | local:<hf-repo>")
    ap.add_argument("--capability", default="python")
    ap.add_argument("--target-tokenizer", default="talkie")
    ap.add_argument("--ref-tokenizer", default="tiktoken:o200k_base")
    ap.add_argument("--control", default="period-prose",
                    choices=sorted(P.CONTROLS))
    ap.add_argument("--strata", default="format-control,oneline,humaneval-style")
    ap.add_argument("--dialects", default="py-canonical")
    ap.add_argument("--budgets", default="0,1,2,4,8")
    ap.add_argument("--samples", type=int, default=10,
                    help="draws per (task, budget); caps the largest reportable k")
    ap.add_argument("--tasks", type=int, default=None, help="limit tasks per stratum")
    ap.add_argument("--max-tokens", type=int, default=140)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-requests", type=int, default=None,
                    help="hard ceiling on generations; the endpoint is a free demo")
    ap.add_argument("--min-interval", type=float, default=None,
                    help="seconds between requests. The talkie chat endpoint is "
                         "burst-rate-limited (429, no Retry-After); the backend "
                         "default is 3.0s")
    ap.add_argument("--allow-exec", action="store_true",
                    help="permit executing generated code (see sandbox.py)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--penalty-only", action="store_true",
                    help="tokenizer measurement only: no network, no execution")
    ap.add_argument("--humaneval-jsonl", default=None,
                    help="official HumanEval JSONL, for comparable numbers")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if not a.penalty_only and not a.backend:
        ap.error("give a --backend, or use --penalty-only")

    target = P.load_tokenizer(a.target_tokenizer)
    ref = P.load_tokenizer(a.ref_tokenizer)
    control = P.CONTROLS[a.control]
    cap = C.load_capability(a.capability, a.humaneval_jsonl)
    dialects = [d.strip() for d in a.dialects.split(",") if d.strip()]
    strata = [s.strip() for s in a.strata.split(",") if s.strip()]
    budgets = [int(b) for b in a.budgets.split(",") if b.strip()]

    backend = (load_backend(a.backend, cache=not a.no_cache)
               if a.backend else None)
    if a.min_interval is not None and backend is not None:
        inner = getattr(backend, "inner", backend)
        if hasattr(inner, "min_interval"):
            inner.min_interval = a.min_interval

    penalties, pen_norm = {}, {}
    for d in dialects:
        pen = P.penalty_of(target, ref, cap.penalty_material(d), control, d)
        penalties[d] = {
            "target_tokens": pen.material.target_tokens,
            "ref_tokens": pen.material.ref_tokens,
            "chars": pen.material.chars,
            "ratio": pen.material.ratio,
            "normalized": pen.normalized,
            "tok_per_char": pen.material.tok_per_char,
            "tok_per_char_excess": pen.tok_per_char_excess,
        }
        pen_norm[d] = pen.normalized
    ctrl_m = P.measure(target, ref, control, "control")

    payload = {
        "meta": (V.meta_block(backend, cap, target, ref, control=a.control,
                              humaneval_jsonl=a.humaneval_jsonl,
                              samples_per_task=a.samples, budgets=budgets,
                              dialects=dialects, strata=strata,
                              temperature=a.temperature)
                 if backend else
                 {**V.meta_block(_NullBackend(), cap, target, ref,
                                 control=a.control,
                                 humaneval_jsonl=a.humaneval_jsonl),
                  "budgets": budgets, "dialects": dialects}),
        "reference_point": reference_point(),
        "penalty": penalties,
        "penalty_control": {"target_tokens": ctrl_m.target_tokens,
                            "ref_tokens": ctrl_m.ref_tokens,
                            "ratio": ctrl_m.ratio},
        "curves": {}, "cells": {}, "analysis": {},
    }

    if not a.penalty_only:
        code_curves = {}
        for stratum in strata:
            for d in dialects:
                if not cap.is_code(stratum) and d != dialects[0]:
                    continue  # dialects are a property of the code surface only
                label = f"{stratum}|{d}" if cap.is_code(stratum) else stratum
                print(f"sweeping {label} ...")
                cells = V.run_sweep(
                    backend, cap, stratum=stratum, dialect=d, budgets=budgets,
                    n_samples=a.samples, target_tok=target, ref_tok=ref,
                    max_tokens=a.max_tokens, temperature=a.temperature,
                    max_requests=a.max_requests, allow_exec=a.allow_exec,
                    task_limit=a.tasks)
                pts = V.build_curve(cells, seed=a.seed)
                payload["curves"][label] = [p.as_dict() for p in pts]
                payload["cells"][label] = [c.as_dict() for c in cells]
                if cap.is_code(stratum):
                    code_curves.setdefault(stratum, {})[d] = pts
                if stratum == "format-control" and pts:
                    last = pts[-1]
                    sr, lo, hi = last.solve_rate
                    payload["analysis"]["format_control"] = {
                        "passed": last.tasks_solved > 0,
                        "summary": S.fmt_rate(last.tasks_solved, last.n_tasks,
                                              label="tasks solved at the largest "
                                                    "budget"),
                        "solve_rate": [sr, lo, hi],
                    }
        # H1 is tested per code stratum, not once on whichever stratum happened
        # to be swept last. The strata differ by design in how much signal they
        # carry — `oneline` is the regime the report says works and
        # `humaneval-style` sits near the floor — so pooling them, or silently
        # analysing only the last, would test the hypothesis on whichever
        # stratum has the least to say.
        for stratum, by_dialect in code_curves.items():
            if len(by_dialect) >= 2:
                tpd = {d: _tokens_per_demo(p) for d, p in by_dialect.items()}
                pure = [tpd[d] for d in tpd if d in C.CONTENT_PRESERVING]
                entry = {
                    "axis_alignment": V.axis_alignment(by_dialect),
                    "penalty_vs_recovery": V.penalty_vs_recovery(
                        by_dialect, pen_norm, tokens_per_demo=tpd,
                        content_preserving=C.CONTENT_PRESERVING, seed=a.seed),
                    "tokens_per_demo": tpd,
                }
                if len(pure) >= 2:
                    entry["leverage"] = {
                        "min": min(pure), "max": max(pure),
                        "ratio": max(pure) / min(pure),
                        "n_content_preserving": len(pure),
                    }
            else:
                entry = {"penalty_vs_recovery": {
                    "testable": False,
                    "reason": (f"only {len(by_dialect)} dialect swept for this "
                               "stratum; pass --dialects with 3+ "
                               "content-preserving ones (py-canonical,"
                               "py-indent8,py-indent16,py-tab,py-crlf) to "
                               "test H1"),
                }}
            payload["analysis"].setdefault("by_stratum", {})[stratum] = entry
        if hasattr(backend, "hits"):
            payload["meta"]["cache"] = {"hits": backend.hits,
                                        "misses": backend.misses}
        payload["meta"]["generations"] = backend.requests_made
        inner = getattr(backend, "inner", backend)
        if getattr(inner, "rate_limit_hits", 0):
            payload["meta"]["rate_limit_hits"] = inner.rate_limit_hits

    payload["analysis"]["cost"] = cost_note()

    lines = []
    render(payload, emit_to(lines))
    stem = a.out or (
        f"results/reconstitution-{a.capability}-"
        f"{'penalty' if a.penalty_only else a.backend.replace(':', '-').replace('/', '-')}"
    )
    txt = ROOT / f"{stem}.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("\n".join(lines) + "\n")
    V.dump_json(ROOT / f"{stem}.json", payload)
    print(f"\nwrote {stem}.txt and {stem}.json")


class _NullBackend:
    name = "none (penalty-only run)"
    variant = "none"
    contamination = ("No generation performed. The encoding penalty is a "
                     "prediction about reconstitution cost, not a measurement "
                     "of it.")
    requests_made = 0


if __name__ == "__main__":
    main()
