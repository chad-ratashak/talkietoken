"""Rate estimation with uncertainty. numpy only — no scipy.

Every rate this harness reports goes through here, because the regime we
measure in is the one where casual statistics fail hardest: proportions
near zero (talkie's own published pass@100 is 0.032) at sample sizes small
enough that the normal approximation produces negative lower bounds.

Three estimators, each for a different shape of question:

    wilson_ci          a plain proportion — k successes out of n
                       independent units. Score interval, not Wald: at
                       p=0.03, n=20 the Wald interval is [-0.045, 0.145],
                       which is not a probability. Wilson stays inside
                       [0, 1] and keeps roughly nominal coverage down to
                       single-digit counts.

    per_problem_pass_at_k  the HumanEval estimator: pass@k per problem is a
                       nonlinear function of that problem's success count,
                       so the mean over problems is not a proportion and
                       Wilson does NOT apply to it.

    cluster_bootstrap  the CI that does apply to its mean. Resamples
                       PROBLEMS (not generations) with replacement, because
                       the k samples drawn for one problem are not
                       independent of each other — they share a prompt.
                       Bootstrapping generations would understate the
                       interval by pretending n_problems x n_samples
                       independent trials.

And two inference helpers used for the encoding-penalty hypothesis:
spearman + permutation_p (rank correlation with an exact-ish null, no
distributional assumption, valid at the tiny group counts a dialect sweep
produces), and required_n, which answers the only question worth asking
before spending money on an underpowered run.

Deliberately absent: any function that extrapolates pass@k above the k you
actually sampled. The unbiased estimator needs n >= k. You cannot buy
pass@100 with 20 samples; see max_reliable_k.
"""

import math

import numpy as np

# --- normal quantile, since scipy.stats.norm.ppf is off the table --------

# Acklam's rational approximation, then one Halley step against math.erfc.
# Acklam alone is ~1.15e-9 relative; the refinement takes it to full double
# precision, which matters only for reproducibility across machines but is
# two lines.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)
_P_LOW = 0.02425


def norm_ppf(p):
    """Inverse standard-normal CDF. norm_ppf(0.975) == 1.959963984540054."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q
             + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r
             + _A[5]) * q / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r
                              + _B[4]) * r + 1)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q
              + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    # Halley refinement: e is the CDF error at x, u the Newton step.
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def z_for(alpha=0.05):
    """Two-sided critical value. z_for(0.05) == 1.96."""
    return norm_ppf(1.0 - alpha / 2.0)


# --- proportions --------------------------------------------------------

def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for k successes in n independent trials.

    Returns (point, lo, hi) where point is the raw k/n, not the shrunk
    Wilson centre — the reported rate should be the thing that was
    measured, with the interval carrying the uncertainty. n == 0 gives
    (nan, 0, 1): no information, and saying so beats dividing by zero.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"wilson_ci needs 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return float("nan"), 0.0, 1.0
    z = z_for(alpha)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def required_n(p, halfwidth, alpha=0.05, cap=10_000_000):
    """Smallest n whose Wilson halfwidth at rate p is <= halfwidth.

    Solved by search on the exact Wilson expression rather than the
    normal-approximation formula n = z^2 p(1-p)/h^2, which is badly wrong
    in the small-p regime this harness lives in. Answers "what would this
    experiment have to cost to say anything?" — the question to settle
    before a run, not after.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"required_n needs 0 <= p <= 1, got {p}")
    if halfwidth <= 0:
        raise ValueError("halfwidth must be positive")
    lo, hi = 1, 1024
    while hi < cap:
        _, a, b = wilson_ci(round(p * hi), hi, alpha)
        if (b - a) / 2.0 <= halfwidth:
            break
        lo, hi = hi, hi * 2
    else:
        return None  # would need more than `cap` trials; report as such
    while lo < hi:  # binary search the boundary
        mid = (lo + hi) // 2
        _, a, b = wilson_ci(round(p * mid), mid, alpha)
        if (b - a) / 2.0 <= halfwidth:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --- pass@k -------------------------------------------------------------

def pass_at_k(n, c, k):
    """Unbiased pass@k for one problem: n samples drawn, c of them correct.

    1 - C(n-c, k) / C(n, k), evaluated as a product so it stays stable at
    the n where the binomials overflow. This is the Chen et al. (2021)
    HumanEval estimator; the numbers it produces are NOT integer multiples
    of 1/n_problems, which is why talkie's published 0.03215 x 164 = 5.27.
    """
    if k < 1:
        raise ValueError(f"pass_at_k needs k >= 1, got {k}")
    if not 0 <= c <= n:
        raise ValueError(f"pass_at_k needs 0 <= c <= n, got c={c}, n={n}")
    if k > n:
        raise ValueError(
            f"pass@{k} needs at least {k} samples per problem, got n={n}. "
            "The estimator does not extrapolate upward; sample more."
        )
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def per_problem_pass_at_k(counts, k):
    """pass@k for each problem. counts is an iterable of (n, c) pairs."""
    return np.array([pass_at_k(n, c, k) for n, c in counts], dtype=float)


def max_reliable_k(counts):
    """Largest k the drawn samples support — min n over problems.

    Guards the one mistake that would make this harness's numbers look
    comparable to the report's when they are not: pass@100 requires 100
    samples per problem, full stop.
    """
    ns = [n for n, _ in counts]
    return min(ns) if ns else 0


# --- bootstrap ----------------------------------------------------------

def cluster_bootstrap_ci(values, alpha=0.05, n_boot=10_000, seed=0):
    """Percentile CI for the mean of `values`, resampling the units.

    `values` is one number per independent unit — per-problem pass@k, say.
    Resampling problems (clusters) rather than generations is the whole
    point: samples within a problem share a prompt and are correlated, so
    treating them as independent would shrink the interval by roughly
    sqrt(n_samples) for no reason.

    Returns (mean, lo, hi). A degenerate all-equal input (every problem
    zero, the common case near the floor) returns a zero-width interval,
    which is honest about the bootstrap and NOT a claim that the rate is
    known exactly — see zero_ceiling for the number to quote there.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(v.mean()), float(lo), float(hi)


def zero_ceiling(n, alpha=0.05):
    """Upper bound on a rate after n trials with zero successes.

    The rule-of-three generalisation: 1 - alpha^(1/n). At n=20, alpha=0.05
    this is 0.139 — so "0 of 20" is consistent with a true rate up to 14%,
    which is four times talkie's reported pass@100. Quote this instead of
    "0%" whenever a cell is empty, or the run will read as a null result
    when it is actually a non-result.
    """
    if n <= 0:
        return 1.0
    return 1.0 - alpha ** (1.0 / n)


# --- rank correlation, for the penalty hypothesis -----------------------

def _ranks(a):
    """Average ranks, ties shared. Hand-rolled: scipy.stats.rankdata is out."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # average tied groups
    uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    for gi in np.nonzero(counts > 1)[0]:
        mask = inv == gi
        ranks[mask] = ranks[mask].mean()
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def spearman(x, y):
    """Rank correlation. Chosen over Pearson because the penalty-recovery
    relation is predicted to be monotone, not linear, and because with a
    handful of dialects one outlier would own a Pearson coefficient."""
    return pearson(_ranks(x), _ranks(y))


def permutation_p(x, y, stat=spearman, n_perm=20_000, seed=0):
    """Two-sided permutation p-value for an association statistic.

    Shuffles y against x, so the null is exactly "no association" with no
    appeal to a t-distribution that does not hold at n=5. Uses the
    (1 + hits) / (1 + n_perm) form, which cannot return p = 0 — a p of
    exactly zero is always a lie about resolution.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = stat(x, y)
    if math.isnan(obs):
        return float("nan"), obs, 0
    rng = np.random.default_rng(seed)
    yv = y.copy()
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(yv)
        s = stat(x, yv)
        if not math.isnan(s) and abs(s) >= abs(obs) - 1e-12:
            hits += 1
    n_distinct = math.factorial(x.size) if x.size <= 8 else None
    return (1 + hits) / (1 + n_perm), obs, n_distinct


def min_significant_rho(n, alpha=0.05):
    """Smallest |Spearman rho| that could reach two-sided p <= alpha at n.

    A power statement you can compute before running anything, by exact
    enumeration of the permutation null (n! orderings, so n <= 8). It exists
    because the dialect sweep produces very few groups, and at small n the
    permutation test has a hard resolution floor that is easy to mistake for
    a null result:

        n = 4   no rho whatsoever can reach p <= 0.05
        n = 5   only a perfect +/-1.0 ordering can (p = 2/120 = 0.017)
        n = 6   |rho| >= 0.886
        n = 7   |rho| >= 0.786

    So at five dialects, "p = 0.24" is not evidence of no effect; it is the
    test saying it cannot resolve one. Returns None when no value qualifies.
    """
    import itertools

    if n < 2:
        return None
    if n > 8:
        return None  # enumeration too large; the floor is negligible by then
    base = np.arange(1.0, n + 1)
    rhos = np.array([pearson(base, np.array(p, dtype=float))
                     for p in itertools.permutations(range(1, n + 1))])
    total = rhos.size
    best = None
    for r in np.unique(np.abs(rhos)):
        p_two = float((np.abs(rhos) >= r - 1e-12).sum()) / total
        if p_two <= alpha:
            best = float(r)
            break
    return best


# --- formatting: the guard against bare percentages ---------------------

def fmt_rate(k, n, alpha=0.05, label=""):
    """'12/240 = 5.00% [2.9, 8.5] 95% Wilson' — never a bare percentage.

    n is printed first and always, because the number that decides whether
    a rate means anything is the denominator, and a formatter that lets you
    omit it will eventually be used to omit it.
    """
    p, lo, hi = wilson_ci(k, n, alpha)
    conf = int(round((1 - alpha) * 100))
    head = f"{label + ': ' if label else ''}{k}/{n}"
    if n == 0:
        return f"{head} = n/a (no trials)"
    if k == 0:
        ceil = zero_ceiling(n, alpha)
        return (f"{head} = 0% — consistent with any rate up to "
                f"{100 * ceil:.1f}% ({conf}% one-sided)")
    return (f"{head} = {100 * p:.2f}% "
            f"[{100 * lo:.2f}, {100 * hi:.2f}] {conf}% Wilson")


def fmt_pass_at_k(counts, k, alpha=0.05, n_boot=10_000, seed=0):
    """pass@k with the cluster-bootstrap interval and every raw n shown."""
    vals = per_problem_pass_at_k(counts, k)
    mean, lo, hi = cluster_bootstrap_ci(vals, alpha, n_boot, seed)
    n_problems = len(vals)
    n_gen = sum(n for n, _ in counts)
    n_solved = sum(1 for _, c in counts if c > 0)
    conf = int(round((1 - alpha) * 100))
    raw = (f"problems={n_problems}, samples/problem="
           f"{min(n for n, _ in counts)}-{max(n for n, _ in counts)}, "
           f"generations={n_gen}, problems solved>=1: {n_solved}")
    if n_solved == 0:
        # A bootstrap of all-zero clusters returns [0, 0]. Printing that
        # would read as "the rate is exactly zero, precisely known", which
        # is the opposite of what n generations of nothing tell you.
        ceil = zero_ceiling(n_problems, alpha)
        return (f"pass@{k} = 0% — no problem solved; bootstrap is degenerate. "
                f"Consistent with a per-problem solve rate up to "
                f"{100 * ceil:.1f}% ({conf}% one-sided on {n_problems} "
                f"problems) ({raw})")
    return (f"pass@{k} = {100 * mean:.2f}% "
            f"[{100 * lo:.2f}, {100 * hi:.2f}] {conf}% cluster-bootstrap "
            f"({raw})")
