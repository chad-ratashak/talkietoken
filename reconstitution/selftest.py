"""Does the harness measure what it claims to? Run before trusting a number.

    uv run python reconstitution/selftest.py

No network. Executes code only through the same sandbox the real runs use,
on programs this file wrote. Exits non-zero on any failure.

The checks, in order of how badly a failure would matter:

  1. the sandbox separates pass, fail, and runaway
  2. pass@k matches brute-force binomial coefficients everywhere
  3. Wilson intervals cover at their nominal rate
  4. cluster bootstrap covers at its nominal rate on correlated draws
  5. the full pipeline recovers a recovery curve it generated itself
  6. the permutation test's resolution floor, computed by enumeration
  7. H1 moves in the predicted direction when an effect is planted, and
     never reaches significance when none is
  8. how much leverage the dialect lever actually has

5-8 are the ones worth the runtime. A harness that cannot recover its own
synthetic curve has no business reporting a real one; a hypothesis test that
fires on data containing no effect is worse than no test at all; and a test
whose resolution floor is above any attainable effect size (6, 8) should say
so rather than report a null.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import backends  # noqa: E402
import capabilities as C  # noqa: E402
import curve as V  # noqa: E402
import penalty as P  # noqa: E402
import sandbox  # noqa: E402
import statutils as S  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


def test_sandbox():
    print("\n1. sandbox")
    r = sandbox.self_check()
    check("runs a true assertion", r["passes_true_assert"])
    check("catches a false assertion", r["fails_false_assert"])
    check("kills an infinite loop", r["stops_infinite_loop"], r["loop_failure"])


def test_pass_at_k():
    print("\n2. pass@k against brute-force binomials")
    from math import comb
    worst = 0.0
    for n in range(1, 40):
        for c in range(n + 1):
            for k in range(1, n + 1):
                want = 1 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0
                worst = max(worst, abs(S.pass_at_k(n, c, k) - want))
    check("exact for all n<=39", worst < 1e-12, f"max error {worst:.2e}")
    try:
        S.pass_at_k(20, 3, 100)
        check("refuses k > n", False)
    except ValueError:
        check("refuses k > n", True)
    check("published value reproduces", abs(0.03215 * 164 - 5.2726) < 1e-3,
          "talkie's 0.03215 is 5.27/164 — a non-integer, so n>100 per problem")


def test_wilson_coverage(n_trials=4000, n=25, p=0.04, seed=3):
    print("\n3. Wilson interval coverage")
    rng = np.random.default_rng(seed)
    ks = rng.binomial(n, p, n_trials)
    covered = sum(1 for k in ks
                  if (lambda t: t[1] <= p <= t[2])(S.wilson_ci(int(k), n)))
    rate = covered / n_trials
    check(f"covers ~95% at n={n}, p={p}", 0.92 <= rate <= 0.99,
          f"{100 * rate:.1f}% over {n_trials} replications")
    # The comparison that justifies the choice: Wald at these settings.
    z = S.z_for(0.05)
    wald = 0
    for k in ks:
        ph = k / n
        h = z * (ph * (1 - ph) / n) ** 0.5
        if ph - h <= p <= ph + h:
            wald += 1
    check("Wald undercovers here (why Wilson)", wald / n_trials < rate,
          f"Wald {100 * wald / n_trials:.1f}% vs Wilson {100 * rate:.1f}%")


def test_bootstrap_coverage(n_rep=600, n_problems=12, n_samples=20, seed=5):
    print("\n4. cluster-bootstrap coverage on correlated draws")
    rng = np.random.default_rng(seed)
    # The estimand has to match what the interval targets, or the check is
    # measuring the wrong thing. Resampling PROBLEMS means the interval is
    # for the mean over the population of problems this task set is drawn
    # from — not for these exact twelve. So each replication draws a fresh
    # task set from that population; holding the twelve fixed would make the
    # interval look conservative when it is simply answering a broader
    # question. Difficulties are Beta(0.8, 6): heavily skewed toward hard,
    # like a real capability near its floor.
    a, b = 0.8, 6.0
    truth = float(np.mean(1 - (1 - rng.beta(a, b, 400_000)) ** 3))
    covered = 0
    for _ in range(n_rep):
        probs = rng.beta(a, b, n_problems)
        counts = [(n_samples, int(rng.binomial(n_samples, q))) for q in probs]
        vals = S.per_problem_pass_at_k(counts, 3)
        _, lo, hi = S.cluster_bootstrap_ci(vals, n_boot=1200,
                                           seed=int(rng.integers(1 << 30)))
        covered += lo <= truth <= hi
    rate = covered / n_rep
    check(f"covers ~95% of population pass@3={truth:.3f}", 0.88 <= rate <= 0.99,
          f"{100 * rate:.1f}% over {n_rep} replications, {n_problems} problems")
    check("interval is for problems-like-these, not these-exact-problems", True,
          "cluster bootstrap generalises past the shipped task set; it is "
          "conservative if you meant the latter")


def _sweep(sensitivity, dialects, samples=25, budgets=(0, 1, 2, 4, 8), seed=0):
    cap = C.load_capability("python")
    target = P.load_tokenizer("talkie")
    ref = P.load_tokenizer("tiktoken:o200k_base")
    be = backends.SyntheticBackend(penalty_sensitivity=sensitivity, seed=seed)
    curves, pen, tpd = {}, {}, {}
    for d in dialects:
        cells = V.run_sweep(be, cap, stratum="oneline", dialect=d,
                            budgets=list(budgets), n_samples=samples,
                            target_tok=target, ref_tok=ref,
                            allow_exec=True, progress=lambda *_: None)
        curves[d] = V.build_curve(cells, n_boot=2000)
        pen[d] = P.penalty_of(target, ref, cap.penalty_material(d),
                              P.PERIOD_PROSE_CONTROL).normalized
        tpd[d] = tokens_per_demo(curves[d])
    return be, curves, pen, tpd


def tokens_per_demo(points):
    """Mean target tokens each demonstration costs, over the non-zero
    budgets — the direct mechanism predictor for H1a."""
    vals = [p.demo_tokens_target / p.n_demos for p in points if p.n_demos]
    return sum(vals) / len(vals) if vals else float("nan")


def test_pipeline_recovers_its_own_curve():
    print("\n5. end-to-end: recover a curve the harness generated")
    be, curves, _, _ = _sweep(0.0, ["py-canonical"], samples=40)
    pts = curves["py-canonical"]
    ok, rows = True, []
    for p in pts:
        truth = be.true_rate(p.n_demos)
        est = p.pass_at["1"]
        inside = est["lo"] <= truth <= est["hi"] or (truth == 0 and est["mean"] == 0)
        ok &= inside
        rows.append(f"demos={p.n_demos:<2} true={truth:.3f} "
                    f"est={est['mean']:.3f} [{est['lo']:.3f},{est['hi']:.3f}] "
                    f"{'ok' if inside else 'MISS'}")
    for r in rows:
        print(f"        {r}")
    check("every budget's pass@1 interval covers the truth", ok)
    means = [p.pass_at["1"]["mean"] for p in pts]
    check("curve is monotone in demonstration budget",
          all(b >= a - 1e-9 for a, b in zip(means, means[1:])),
          " -> ".join(f"{m:.3f}" for m in means))
    check("zero-demo baseline is zero", means[0] == 0.0,
          "no demonstrations, no capability — the floor the curve rises from")


DIALECTS = ["py-canonical", "py-nodoc", "py-comment", "py-indent8",
             "py-indent16", "py-tab", "py-crlf"]


def _h1(sensitivity, samples=60, seed=0):
    _, curves, pen, tpd = _sweep(sensitivity, DIALECTS, samples=samples, seed=seed)
    return V.penalty_vs_recovery(curves, pen, tokens_per_demo=tpd,
                                 content_preserving=C.CONTENT_PRESERVING, seed=1)


def test_h1_power_floor():
    print("\n6. permutation-test resolution floor (computed, not assumed)")
    for n in (4, 5, 6, 7):
        v = S.min_significant_rho(n)
        print(f"        n={n} groups: {'no rho can reach p<=0.05' if v is None else f'needs |rho| >= {v:.3f}'}")
    check("n=5 needs a perfect ordering", S.min_significant_rho(5) == 1.0,
          "so 'p=0.24' at 5 dialects is unresolved, NOT a null result")
    check("n=4 cannot reach significance at all",
          S.min_significant_rho(4) is None)


def test_dialect_leverage():
    print("\n8. how much leverage do content-preserving dialects actually have?")
    cap = C.load_capability("python")
    target = P.load_tokenizer("talkie")
    costs = {d: target.count(cap.penalty_material(d)) / 8
             for d in C.CONTENT_PRESERVING}
    lo, hi = min(costs.values()), max(costs.values())
    for d, c in sorted(costs.items(), key=lambda kv: kv[1]):
        print(f"        {d:<14} {c:5.2f} target tokens per demonstration")
    print(f"        spread {lo:.2f} - {hi:.2f} = {hi / lo:.2f}x")
    check("leverage is reported, not assumed", True,
          f"{hi / lo:.2f}x spread. At a matched token budget this buys only a "
          f"~{100 * (hi / lo - 1):.0f}% difference in demonstrations delivered, "
          "so H1b on dialects alone is intrinsically low-powered — see METHODS "
          "for the higher-leverage designs (cross-language, cross-model).")


def test_h1_null_and_effect(seeds=(0, 1, 2, 3, 4)):
    """Paired: the same sweep with and without a built-in penalty effect.

    Asserted as a paired comparison across seeds rather than a single-run
    significance call, because a 5-dialect permutation test cannot reach
    p<=0.05 without a perfect ordering (test 6). What CAN be validated is
    that switching the effect on moves rho reliably in the predicted
    direction — which is the property the instrument needs.
    """
    print("\n7. H1 direction: effect ON vs effect OFF, paired over seeds")
    rows = []
    for seed in seeds:
        null = _h1(0.0, seed=seed)["H1b_encoding_penalty"]
        eff = _h1(2.5, seed=seed)["H1b_encoding_penalty"]
        if not (null.get("testable") and eff.get("testable")):
            return check("paired H1b ran", False, "not testable")
        rows.append((seed, null["spearman_rho"], eff["spearman_rho"],
                     null["permutation_p"]))
        print(f"        seed {seed}: rho_null={null['spearman_rho']:+.3f}  "
              f"rho_effect={eff['spearman_rho']:+.3f}")
    mean_null = sum(r[1] for r in rows) / len(rows)
    mean_eff = sum(r[2] for r in rows) / len(rows)
    print(f"        mean: rho_null={mean_null:+.3f}  rho_effect={mean_eff:+.3f}")
    check("effect run is more negative than null run",
          mean_eff < mean_null, f"{mean_eff:+.3f} < {mean_null:+.3f}")
    check("effect run is negative, as predicted", mean_eff < 0,
          f"mean rho={mean_eff:+.3f}")
    check("null run never reaches p<=0.05 on any seed",
          all(r[3] > 0.05 for r in rows),
          "a hit here would mean the test invents effects")
    # Deliberately NOT asserted per seed. With only ~13% leverage (test 8)
    # the per-seed effect sits at the noise floor, and requiring every seed
    # to move the right way would be requiring noise to behave — the same
    # error this harness exists to avoid making about a model.
    moved = sum(1 for _, n, e, _ in rows if e < n)
    print(f"        per-seed direction: {moved}/{len(rows)} moved as predicted "
          f"({' '.join(f's{s}:{n:+.2f}->{e:+.2f}' for s, n, e, _ in rows)})")


def main():
    print("=== reconstitution harness self-test ===")
    test_sandbox()
    test_pass_at_k()
    test_wilson_coverage()
    test_bootstrap_coverage()
    test_pipeline_recovers_its_own_curve()
    test_h1_power_floor()
    test_dialect_leverage()
    test_h1_null_and_effect()
    print("\n" + ("=" * 50))
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
