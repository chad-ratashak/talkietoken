"""Atomic-age probe: what talkie and the modern baselines each *know*
about the Manhattan Project — its places, hardware, and people.

Two weight-free measurements:

1. **Shatter count** — how many tokens a name costs. A string the corpus
   saw often earns merges and collapses to one token; a string it never
   saw shatters. Everything is encoded with a LEADING SPACE (" Bohr", not
   "Bohr"), because that is how a name actually appears in running text
   and it is how word-initial tokens are stored. Mixing the two forms
   silently compares different strings.

2. **Percentile merge rank** — for a token whole in BOTH vocabularies,
   rank/vocab_size is an ordinal frequency proxy (0 = earliest merge =
   most useful substring). Δ between talkie and o200k is a fame-shift
   signal. Method and caveats as in talkietoken-rankshift.py.

THE CONFOUND, which is also the headline finding
------------------------------------------------
A cheap token means the *string* was common, not that the *referent* was
known. "Teller", "Jeans", "Trinity", "Gay", "Little", "Boy", "Fat" are
ordinary English words; a 1930 corpus merges them for reasons that have
nothing to do with Edward Teller or a plutonium bomb. So every term is
flagged against /usr/share/dict/words (macOS ships Webster's Second,
1934 — period-appropriate). A cheap token on a dictionary word is
evidence of NOTHING about the referent, and is reported as such.

That flag is what makes the codename hypothesis testable: if Fat Man,
Little Boy, Enola Gay and Trinity are cheap in talkie *and* every piece
is a dictionary word, the tokenizer is reading them as plain nouns.

The scientist set is three cohorts so the measure can be falsified:

    KEPT   famous before 1931 AND after   -> present both sides
    ROSE   famous only after the war      -> expect costly/late in talkie
    FADED  famous before, forgotten since -> expect cheap/early in talkie
                                             but costly/late in o200k

FADED is load-bearing. Without it, "these names are expensive in talkie"
is equally explained by "old tokenizers are just bad at proper nouns."

Writes results/atomic.txt and results/atomic.json.
"""

import json
from pathlib import Path

import baselines
import talkie_tokenizer

ROOT = Path(__file__).resolve().parent.parent

t = talkie_tokenizer.from_pretrained()
talkie_vocab = t.get_vocab()
talkie_max = max(talkie_vocab.values())

encs = baselines.load_baselines()
o200k = encs["o200k"]
o200k_vocab = {
    b.decode("utf-8", errors="replace"): r
    for b, r in o200k._mergeable_ranks.items()  # noqa: SLF001
}
o200k_max = max(o200k_vocab.values())

# Webster's Second (1934) as shipped by macOS — used only to flag terms
# whose cheapness is explained by being an ordinary word.
DICT = set()
try:
    with open("/usr/share/dict/words") as f:
        DICT = {w.strip().lower() for w in f if w.strip().isalpha()}
except FileNotFoundError:
    pass


def is_common_word(s):
    """True if s (or its singular) is an ordinary dictionary word."""
    w = s.strip().lower()
    if not w.isalpha():
        return False
    if w in DICT:
        return True
    return w.endswith("s") and w[:-1] in DICT


# --- term sets --------------------------------------------------------
PLACES = [
    ("Hiroshima", "real city pre-1931 — Meiji garrison town and port"),
    ("Nagasaki", "real city pre-1931 — treaty port, Madame Butterfly"),
    ("Los Alamos", "existed, but an obscure New Mexico ranch school"),
    ("Alamogordo", "existed as a New Mexico railroad town"),
    ("Oak Ridge", "the site did not exist until 1942"),
    ("Hanford", "tiny farm town; the site did not exist until 1943"),
    ("the Alamo", "control — famous in 1930 and famous now"),
]

HARDWARE = [
    ("Enola Gay", "rare given name + a very common adjective"),
    ("Bockscar", "pure post-war coinage"),
    ("Fat Man", "two ordinary words"),
    ("Little Boy", "two ordinary words"),
    ("Manhattan Project", "place name + ordinary noun"),
    ("USS Indianapolis", "ship launched Nov 1931 — just past the cutoff"),
    ("Trinity", "ordinary theological noun pre-1931"),
    ("atomic bomb", "both words existed; the compound did not"),
    ("uranium", "known element, mundane pre-1931"),
    ("plutonium", "not synthesized until 1940"),
    ("fission", "biological sense only pre-1931"),
    ("chain reaction", "chemistry sense existed; nuclear sense did not"),
]

SCIENTISTS = {
    "kept": [
        ("Einstein", "world-famous from the 1919 eclipse onward"),
        ("Curie", "Nobel 1903 and 1911; died 1934"),
        ("Planck", "quantum theory, Nobel 1918"),
        ("Bohr", "Nobel 1922"),
        ("Rutherford", "Nobel 1908, the nuclear atom"),
        ("Edison", "household name by 1900"),
        ("Marconi", "household name — wireless"),
        ("Freud", "famous well before 1931"),
    ],
    "rose": [
        ("Oppenheimer", "obscure young theorist in 1930"),
        ("Feynman", "born 1918 — a schoolboy at the cutoff"),
        ("Fermi", "rising, but Nobel 1938 and Chicago Pile 1942"),
        ("Teller", "Hungarian, obscure in 1930"),
        ("Szilard", "Hungarian, obscure in 1930"),
        ("Wigner", "Hungarian, obscure in 1930"),
        ("Bethe", "obscure in 1930"),
        ("Ulam", "born 1909, obscure in 1930"),
        ("Groves", "career army engineer, unknown"),
        ("Meitner", "known in physics circles, not to the public"),
        ("Turing", "born 1912 — a schoolboy at the cutoff"),
    ],
    "faded": [
        ("Millikan", "Nobel 1923; a household science name then"),
        ("Eddington", "the man who 'proved' Einstein — a celebrity"),
        ("Jeans", "best-selling science populariser"),
        ("Michelson", "first US science Nobel, 1907"),
        ("Steinmetz", "the 'wizard of GE', famous in the 1920s"),
        ("Haeckel", "hugely famous Victorian biologist"),
        ("Tyndall", "Victorian science celebrity"),
        ("Osler", "the most famous physician of the era"),
        ("Burbank", "Luther Burbank — famous plant breeder"),
        ("Bergson", "the most famous philosopher alive in 1920"),
    ],
}


# --- helpers ----------------------------------------------------------
def split(enc, term):
    """Decoded pieces of ' term' (leading space = running-text form)."""
    ids = enc.encode(" " + term)
    return [enc.decode([i]) for i in ids]


def talkie_split(term):
    ids = t.encode(" " + term)
    return [t.decode([i]) for i in ids]


def pct_of(vocab, vmax, term):
    """Percentile rank of ' term' if it is a whole token, else None."""
    s = " " + term
    return vocab[s] / vmax if s in vocab else None


def measure(term, note):
    ts = talkie_split(term)
    os_ = split(o200k, term)
    tp = pct_of(talkie_vocab, talkie_max, term)
    op = pct_of(o200k_vocab, o200k_max, term)
    # A term is "explained by ordinary words" when every alphabetic piece
    # of it is a dictionary word.
    parts = [p for p in term.split() if p.isalpha()]
    common = bool(parts) and all(is_common_word(p) for p in parts)
    return {
        "term": term, "note": note,
        "talkie_n": len(ts), "o200k_n": len(os_),
        "gpt2_n": len(encs["gpt2"].encode(" " + term)),
        "cl100k_n": len(encs["cl100k"].encode(" " + term)),
        "talkie_pieces": ts, "o200k_pieces": os_,
        "talkie_pct": tp, "o200k_pct": op,
        "delta": (tp - op) if (tp is not None and op is not None) else None,
        "common_word": common,
    }


out = []


def emit(s=""):
    print(s)
    out.append(s)


emit("=== ATOMIC-AGE PROBE — talkie (pre-1931) vs o200k (2024) ===")
emit(f"talkie vocab {len(talkie_vocab)} · o200k vocab {len(o200k_vocab)}")
emit("All terms encoded with a leading space (running-text form).")
emit("shatter count = tokens the name costs. 1 = the corpus knew the string.")
emit("[word] = every piece is an ordinary dictionary word — a cheap token")
emit("        here says nothing about the referent.")
emit()

data = {"places": [], "hardware": [], "scientists": {}}

for label, terms, key in (("PLACES", PLACES, "places"),
                          ("HARDWARE / CODENAMES", HARDWARE, "hardware")):
    emit(f"--- {label} ---")
    emit(f"{'term':<20} {'talkie':>6} {'o200k':>6}  {'talkie split':<28} note")
    for term, note in terms:
        r = measure(term, note)
        data[key].append(r)
        sp = "|".join(p.strip() or "_" for p in r["talkie_pieces"])
        tag = " [word]" if r["common_word"] else ""
        emit(f"{term:<20} {r['talkie_n']:>6} {r['o200k_n']:>6}  "
             f"{sp:<28}{tag} {note}")
    emit()

emit("--- SCIENTISTS ---")
emit("Δpct < 0 → relatively MORE prominent in talkie (1900s-leaning)")
emit("Δpct > 0 → relatively MORE prominent in o200k (modern-leaning)")
emit()
COHORT = {
    "kept": "famous before 1931 AND after",
    "rose": "famous only after the war      — expect costly in talkie",
    "faded": "famous before, forgotten since — CONTROL, expect the reverse",
}
for cohort, people in SCIENTISTS.items():
    emit(f"  [{cohort.upper()}] {COHORT[cohort]}")
    emit(f"    {'name':<13} {'tk':>3} {'o2':>3} {'tk pct':>8} {'o2 pct':>8} "
         f"{'Δpct':>7}  note")
    rows = []
    for name, note in people:
        r = measure(name, note)
        rows.append(r)
        f_t = f"{r['talkie_pct']:.3f}" if r["talkie_pct"] is not None else "—"
        f_o = f"{r['o200k_pct']:.3f}" if r["o200k_pct"] is not None else "—"
        f_d = f"{r['delta']:+.3f}" if r["delta"] is not None else "—"
        tag = " [word]" if r["common_word"] else ""
        emit(f"    {name:<13} {r['talkie_n']:>3} {r['o200k_n']:>3} "
             f"{f_t:>8} {f_o:>8} {f_d:>7}{tag}  {note}")
    data["scientists"][cohort] = rows
    emit()

# --- summaries --------------------------------------------------------
emit("--- COHORT SUMMARY ---")
emit(f"  {'cohort':<7} {'mean talkie tokens':>19} {'mean o200k tokens':>19} "
     f"{'whole-token in talkie':>23}")
for cohort in ("kept", "rose", "faded"):
    rows = data["scientists"][cohort]
    mt = sum(r["talkie_n"] for r in rows) / len(rows)
    mo = sum(r["o200k_n"] for r in rows) / len(rows)
    whole = [r["term"] for r in rows if r["talkie_n"] == 1]
    emit(f"  {cohort:<7} {mt:>19.2f} {mo:>19.2f} {len(whole):>13}/{len(rows):<9}")
emit()
for cohort in ("kept", "rose", "faded"):
    rows = data["scientists"][cohort]
    whole = [r["term"] + (" [word]" if r["common_word"] else "")
             for r in rows if r["talkie_n"] == 1]
    emit(f"  {cohort:<6} single-token in talkie: {whole or 'none'}")
emit()
emit("  Cheapness on a [word] name (Teller, Jeans, Groves, Burbank) is the")
emit("  ordinary noun, not the person. Discount those before reading the")
emit("  cohort means.")

(ROOT / "results" / "atomic.txt").write_text("\n".join(out) + "\n")
(ROOT / "results" / "atomic.json").write_text(json.dumps(data, indent=1))
emit()
emit("wrote results/atomic.txt and results/atomic.json")
