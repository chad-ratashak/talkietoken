"""The recovery curve: capability as a function of demonstration budget.

A single pass@k number cannot answer the question this harness is for.
"Capability was removed from the weights" and "capability comes back when
you show the model four examples" are compatible; what distinguishes a
robust removal from a cosmetic one is the SHAPE — how much context has to
be spent before the capability reappears, and whether it plateaus low or
keeps climbing. So the unit of measurement here is a curve, swept over a
demonstration-budget ladder (`--budgets`, default `0,1,2,4,8`; the pool holds
16). Budget zero is load-bearing rather than a formality: it is the floor the
curve rises from, and a non-zero value there means the capability was never
absent to begin with.

Every point is reported three ways, because the x-axis is itself
a research question:

    n_demos           what the experimenter chose
    demo tokens under the TARGET tokenizer      what the model actually paid
    demo tokens under a REFERENCE tokenizer     what the material "costs"
                                                in a modern vocabulary

The encoding-penalty hypothesis is a claim about which of those axes the
curve is really a function of. If demonstrations are the unit, curves from
cheap and expensive dialects coincide on the first axis. If context tokens
are the unit, they coincide on the second. `axis_alignment` measures which,
and `penalty_vs_recovery` tests the ordering directly.

Reported for every rate, without exception: the raw numerator, the raw
denominator, and an interval. See statutils for why each rate gets the
interval it gets.
"""

import json
import math
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

import statutils as S


@dataclass
class Cell:
    """One task at one budget: n draws, c correct."""

    task_id: str
    stratum: str
    dialect: str
    n_demos: int
    n_samples: int = 0
    n_correct: int = 0
    demo_tokens_target: float = 0.0
    demo_tokens_ref: float = 0.0
    prompt_tokens_target: float = 0.0
    strategies: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)

    def as_dict(self):
        d = asdict(self)
        d["strategies"] = dict(self.strategies)
        d["failures"] = dict(self.failures)
        return d


def _sample_demos(pool, n, task, seed_parts):
    """`n` demonstrations, drawn per (task, budget, draw) so that different
    draws genuinely see different examples — the report's own protocol is
    "randomly chosen Python functions", not a fixed curated prefix.

    A demo defining the task's own entry point would be the answer written
    out, so it is removed. Nothing else is filtered: a demo that is a near
    neighbour of the task is exactly the "small modification to an
    in-context example" mechanism under study, and censoring it would
    quietly change what the experiment measures.
    """
    usable = [d for d in pool if d.id != task.entry_point]
    if n <= 0:
        return []
    rng = random.Random("|".join(str(p) for p in seed_parts))
    return rng.sample(usable, min(n, len(usable)))


def run_sweep(backend, cap, *, stratum, dialect, budgets, n_samples,
              target_tok, ref_tok, max_tokens=140, temperature=0.8,
              max_requests=None, allow_exec=True, task_limit=None,
              progress=print):
    """Sweep one (stratum, dialect) pair. Returns a list of Cells."""
    tasks = cap.tasks(stratum)
    if task_limit:
        tasks = tasks[:task_limit]
    pool = cap.demos(stratum)
    is_code = cap.is_code(stratum)
    if is_code and not allow_exec:
        raise RuntimeError(
            f"stratum {stratum!r} is graded by executing generated code; "
            "pass allow_exec=True (CLI: --allow-exec) after reading "
            "sandbox.py's limitations, or restrict to --strata format-control"
        )

    # Verification is a pure function of (task, candidate program), and
    # models repeat themselves heavily — the same one-line body comes back
    # for most draws of an easy task. Memoising turns tens of thousands of
    # interpreter spawns into a few hundred without changing any count.
    verdict_cache = {}

    cells, used = [], 0
    for n_demos in budgets:
        for task in tasks:
            cell = Cell(task.id, stratum, dialect, n_demos)
            for draw in range(n_samples):
                if max_requests is not None and used >= max_requests:
                    progress(f"  [request cap {max_requests} reached]")
                    if cell.n_samples:
                        cells.append(cell)
                    return cells
                demos = _sample_demos(pool, n_demos, task,
                                      (stratum, dialect, n_demos, task.id, draw))
                prompt = cap.render(task, demos, dialect)
                # The demonstration block as rendered in this dialect: the
                # bytes whose cost the penalty hypothesis is about. Measured
                # post-transform, or a dialect's whole effect would vanish.
                dl = getattr(cap, "dialect_of", lambda d: None)(dialect)
                demo_text = "".join(
                    (dl.demo(d.source) if dl else d.source) for d in demos)
                prompt_tokens = target_tok.count(prompt)
                cell.demo_tokens_target += target_tok.count(demo_text)
                cell.demo_tokens_ref += ref_tok.count(demo_text)
                cell.prompt_tokens_target += prompt_tokens

                completion = backend.complete(
                    prompt, max_tokens=max_tokens, temperature=temperature,
                    **({"_task": task.entry_point, "_n_demos": n_demos,
                        "_sample": draw, "_prompt_tokens": prompt_tokens}
                       if backend.variant == "synthetic" else {"_sample": draw}),
                )
                used += 1
                candidate, strategy = cap.extract(completion, task)
                ckey = (task.id, candidate)
                if ckey not in verdict_cache:
                    verdict_cache[ckey] = cap.verify(task, candidate)
                verdict = verdict_cache[ckey]
                cell.n_samples += 1
                cell.n_correct += int(verdict.ok)
                cell.strategies[strategy] += 1
                if not verdict.ok:
                    cell.failures[verdict.reason or "wrong"] += 1
            # Averages, so a Cell carries per-draw means rather than sums.
            if cell.n_samples:
                for f in ("demo_tokens_target", "demo_tokens_ref",
                          "prompt_tokens_target"):
                    setattr(cell, f, getattr(cell, f) / cell.n_samples)
            cells.append(cell)
        solved = sum(1 for c in cells if c.n_demos == n_demos and c.n_correct)
        progress(f"  {stratum}/{dialect} demos={n_demos:<3} "
                 f"tasks solved>=1: {solved}/{len(tasks)}  (requests: {used})")
    return cells


# --- turning cells into a curve ----------------------------------------

@dataclass
class Point:
    n_demos: int
    n_tasks: int
    n_generations: int
    n_correct: int
    tasks_solved: int
    demo_tokens_target: float
    demo_tokens_ref: float
    prompt_tokens_target: float
    counts: list                      # [(n, c)] per task, for pass@k
    pass_at: dict = field(default_factory=dict)
    solve_rate: tuple = (0.0, 0.0, 1.0)   # (point, lo, hi) Wilson
    strategies: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)

    def as_dict(self):
        return {**asdict(self), "counts": [list(c) for c in self.counts]}


def build_curve(cells, alpha=0.05, n_boot=10_000, seed=0):
    """Group cells by budget and attach every rate with its interval."""
    points = []
    for n_demos in sorted({c.n_demos for c in cells}):
        group = [c for c in cells if c.n_demos == n_demos and c.n_samples]
        if not group:
            continue
        counts = [(c.n_samples, c.n_correct) for c in group]
        n_gen = sum(n for n, _ in counts)
        solved = sum(1 for _, c in counts if c > 0)
        strat, fail = Counter(), Counter()
        for c in group:
            strat.update(c.strategies)
            fail.update(c.failures)
        p = Point(
            n_demos=n_demos,
            n_tasks=len(group),
            n_generations=n_gen,
            n_correct=sum(c for _, c in counts),
            tasks_solved=solved,
            demo_tokens_target=sum(c.demo_tokens_target for c in group) / len(group),
            demo_tokens_ref=sum(c.demo_tokens_ref for c in group) / len(group),
            prompt_tokens_target=sum(c.prompt_tokens_target for c in group) / len(group),
            counts=counts,
            strategies=dict(strat),
            failures=dict(fail.most_common(6)),
        )
        kmax = S.max_reliable_k(counts)
        for k in sorted({1, min(5, kmax), kmax}):
            vals = S.per_problem_pass_at_k(counts, k)
            mean, lo, hi = S.cluster_bootstrap_ci(vals, alpha, n_boot, seed)
            p.pass_at[str(k)] = {
                "k": k, "mean": mean, "lo": lo, "hi": hi,
                "degenerate": solved == 0,
            }
        p.solve_rate = S.wilson_ci(solved, len(group), alpha)
        points.append(p)
    return points


def recovery_at(points, budget, axis="demo_tokens_target", k=1):
    """Recovery interpolated to a common budget on one axis.

    The only way to compare dialects fairly: they reach different token
    totals at the same demonstration count, so "recovery at 8 demos" and
    "recovery at 300 tokens" are different questions and only the second one
    tests the penalty hypothesis. Linear interpolation, and None outside the
    measured range rather than an extrapolation nobody asked for.

    k defaults to 1 and should stay there for any comparison between
    conditions. pass@k at large k saturates: at a per-draw rate of 0.3, 25
    draws give pass@25 = 0.9998, and every condition worth telling apart
    compresses into the last two decimals. pass@1 is the per-attempt success
    probability, monotone in capability and flat nowhere. Large-k values are
    for reporting the ceiling, not for ranking.
    """
    xs, ys = [], []
    for p in points:
        key = str(k) if k is not None else max(p.pass_at, key=lambda s: int(s))
        if key not in p.pass_at:
            continue
        xs.append(p.n_demos if axis == "n_demos" else getattr(p, axis))
        ys.append(p.pass_at[key]["mean"])
    if len(xs) < 2:
        return None
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    if budget < xs[0] or budget > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if budget <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (budget - x0) / (x1 - x0)
    return ys[-1]


AXES = ("n_demos", "demo_tokens_target", "demo_tokens_ref")


def axis_alignment(curves, k=1, n_grid=9):
    """Which x-axis makes the dialect curves coincide?

    For each candidate axis, walk a grid across the range all dialects have
    in common and record the spread of recovery across dialects. The axis
    with the smallest mean spread is the one the curve is really a function
    of — i.e. the currency reconstitution is actually bought with.

    Reported as a descriptive statistic, not a test. With a handful of
    dialects the differences will usually be inside the bootstrap intervals
    of the points themselves, and the output says so rather than ranking
    axes as if the ordering were established.
    """
    out = {}
    for axis in AXES:
        lo_hi = []
        for pts in curves.values():
            xs = [p.n_demos if axis == "n_demos" else getattr(p, axis)
                  for p in pts]
            # Skip the zero-budget point. Every dialect is identically zero
            # there by construction, so including it pulls the spread toward
            # zero on all three axes equally and blunts the comparison the
            # diagnostic exists to make.
            positive = [x for x in xs if x > 0]
            if len(positive) >= 2:
                lo_hi.append((min(positive), max(positive)))
        if len(lo_hi) < 2:
            continue
        lo = max(a for a, _ in lo_hi)
        hi = min(b for _, b in lo_hi)
        if not hi > lo:
            out[axis] = {"overlap": None, "mean_spread": None,
                         "note": "dialect ranges do not overlap on this axis"}
            continue
        spreads = []
        for i in range(n_grid):
            b = lo + (hi - lo) * i / (n_grid - 1)
            vals = [recovery_at(pts, b, axis, k) for pts in curves.values()]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                spreads.append(max(vals) - min(vals))
        out[axis] = {
            "overlap": [lo, hi],
            "mean_spread": (sum(spreads) / len(spreads)) if spreads else None,
            "n_grid_points": len(spreads),
        }
    return out


def _matched_budget(curves, names, frac=0.75):
    """A target-token budget every named dialect actually reached."""
    ranges = []
    for d in names:
        xs = [p.demo_tokens_target for p in curves[d]]
        if len(xs) >= 2:
            ranges.append((min(xs), max(xs)))
    if len(ranges) < 2:
        return None
    lo, hi = max(a for a, _ in ranges), min(b for _, b in ranges)
    return lo + (hi - lo) * frac if hi > lo else None


def _rank_test(curves, predictor, names, budget, k, n_perm, seed, sign):
    xs, ys, used = [], [], []
    for d in names:
        r = recovery_at(curves[d], budget, "demo_tokens_target", k)
        if r is None or d not in predictor:
            continue
        xs.append(predictor[d])
        ys.append(r)
        used.append(d)
    if len(xs) < 3:
        return {"testable": False,
                "reason": f"needs >= 3 dialects at the shared budget, have {len(xs)}"}
    p_value, rho, n_distinct = S.permutation_p(xs, ys, S.spearman, n_perm, seed)
    if math.isnan(rho):
        return {"testable": False,
                "reason": "predictor or recovery has no variance across dialects"}
    # With n groups the smallest two-sided permutation p is 2/n!, and only a
    # near-perfect ordering attains it. Both facts are reported, because a
    # "non-significant" result here is usually the test declaring it cannot
    # resolve anything rather than evidence of no effect — at n=5 nothing
    # short of rho = +/-1.0 can reach p <= 0.05, and at n <= 4 nothing can.
    floor = 2.0 / math.factorial(len(xs)) if len(xs) <= 8 else 0.0
    need = S.min_significant_rho(len(xs))
    return {
        "testable": True, "matched_budget_target_tokens": budget,
        "k": k, "dialects": used, "predictor": xs, "recovery": ys,
        "spearman_rho": rho, "permutation_p": p_value,
        "n_distinct_permutations": n_distinct, "predicted_sign": sign,
        "n_dialects": len(xs), "min_attainable_p": floor,
        "rho_needed_for_significance": need,
        "underpowered": need is None or abs(rho) < need,
        "direction_matches": (rho < 0) if sign == "negative" else (rho > 0),
    }


def penalty_vs_recovery(curves, penalties, tokens_per_demo=None,
                        content_preserving=(), k=1, n_perm=20_000, seed=0):
    """Does a bigger encoding penalty mean harder reconstitution?

    The brief's hypothesis contains two claims that have to be tested
    separately, because a single test of "penalty vs recovery" across all
    dialects mixes them and returns noise:

      H1a  MECHANISM. At a matched context budget in target tokens,
           demonstrations that cost more tokens each deliver fewer lessons,
           so recovery is lower. Predictor: mean target tokens per
           demonstration. Every dialect is admissible — this is a claim
           about context arithmetic, not about tokenizers.

      H1b  PROXY. The tokenizer-derived normalised penalty — computable
           with no generation at all — predicts that difficulty. Only
           content-preserving dialects are admissible: py-nodoc is cheap
           because it teaches less, not because it encodes better, and
           including it would let a content effect masquerade as an
           encoding effect.

    H1b is the one worth having. If a number obtainable from vocab.txt
    alone forecasts how hard a capability is to reconstitute, the
    forecast can be made for a model nobody has generation access to.
    """
    names = [d for d in curves if d in penalties]
    if len(names) < 3:
        return {"testable": False,
                "reason": f"needs >= 3 dialects with penalties, have {len(names)}"}
    budget = _matched_budget(curves, names)
    if budget is None:
        return {"testable": False,
                "reason": "no common target-token budget across dialects"}

    tokens_per_demo = tokens_per_demo or {}
    pure = [d for d in names if d in content_preserving]
    out = {
        "matched_budget_target_tokens": budget,
        "k": k,
        "H1a_tokens_per_demo": (
            _rank_test(curves, tokens_per_demo, names, budget, k, n_perm, seed,
                       "negative")
            if tokens_per_demo else
            {"testable": False, "reason": "no tokens-per-demo predictor supplied"}),
        "H1b_encoding_penalty": (
            _rank_test(curves, penalties, pure, budget, k, n_perm, seed, "negative")
            if len(pure) >= 3 else
            {"testable": False,
             "reason": (f"only {len(pure)} content-preserving dialect(s) swept "
                        f"({', '.join(pure) or 'none'}); H1b needs >= 3. Add "
                        "--dialects py-canonical,py-indent8,py-tab")}),
    }
    out["testable"] = any(out[key].get("testable") for key in
                          ("H1a_tokens_per_demo", "H1b_encoding_penalty"))
    return out


# --- provenance ---------------------------------------------------------

def git_stamp():
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"commit": run("git", "rev-parse", "--short", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def meta_block(backend, cap, target_tok, ref_tok, **extra):
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git": git_stamp(),
        "backend": backend.name,
        "backend_variant": backend.variant,
        "CONTAMINATION": backend.contamination,
        "capability": cap.name,
        "target_tokenizer": {"spec": target_tok.spec, "label": target_tok.label,
                             "vocab_size": target_tok.vocab_size},
        "reference_tokenizer": {"spec": ref_tok.spec, "label": ref_tok.label,
                                "vocab_size": ref_tok.vocab_size},
        **extra,
    }


def dump_json(path, payload):
    path.write_text(json.dumps(payload, indent=1, default=str))
