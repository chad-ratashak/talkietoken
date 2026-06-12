"""Rank-shift analysis: talkie (pre-1931 English) vs o200k (modern web).

BPE merge rank is a rough frequency proxy — a token merged earlier was a
more useful/frequent substring during BPE training. For a token string
present in BOTH vocabularies we can compare its *percentile* rank
(rank / vocab_size, so 0 = earliest/most frequent, 1 = latest) across the
two corpora. The delta is an ordinal signal of how a word's relative
frequency shifted between 1880–1930 English print and the modern web.

Caveats (deliberately not over-claimed):
  * rank ≈ frequency is approximate, not a count;
  * the two tokenizers differ in size (262k vs 198k) and split regex, so
    only PERCENTILE ranks are comparable, not raw ranks;
  * a single word's delta is noisy — trust thematic CLUSTERS that move
    together, not individual extremes.

Also answers the "was it really 100% English?" question by breaking the
talkie vocab down by script/character class.
"""

import csv
import unicodedata
from pathlib import Path

import talkie_tokenizer

ROOT = Path(__file__).resolve().parent.parent

# --- load both vocabularies as {string: rank} -------------------------
t = talkie_tokenizer.from_pretrained()
talkie = t.get_vocab()
talkie_max = max(talkie.values())

o200k = {}
with open(ROOT / "baseline" / "o200k_tokens.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        o200k[row["token"]] = int(row["token_id"])
o200k_max = max(o200k.values())


# ====================================================================
# PART 1 — script / character-class composition of the talkie vocab
# (Is the corpus really "English-only"?)
# ====================================================================
def classify(s):
    core = s.strip()
    if not core:
        return "whitespace"
    if all(ord(c) < 128 for c in core):
        if any(c.isalpha() for c in core):
            return "ascii-latin"
        return "ascii-symbol/digit"
    scripts = set()
    for c in core:
        if ord(c) < 128:
            continue
        try:
            name = unicodedata.name(c)
        except ValueError:
            scripts.add("other")
            continue
        if "CJK" in name:
            scripts.add("CJK")
        elif "CYRILLIC" in name:
            scripts.add("Cyrillic")
        elif "GREEK" in name:
            scripts.add("Greek")
        elif "ARABIC" in name:
            scripts.add("Arabic")
        elif "HEBREW" in name:
            scripts.add("Hebrew")
        elif "LATIN" in name:
            scripts.add("Latin-extended")  # accented: café, naïve, ü…
        elif "HIRAGANA" in name or "KATAKANA" in name:
            scripts.add("Japanese-kana")
        elif "DEVANAGARI" in name:
            scripts.add("Devanagari")
        else:
            scripts.add("other")
    if len(scripts) == 1:
        return next(iter(scripts))
    if scripts == {"Latin-extended"}:
        return "Latin-extended"
    return "mixed/" + "+".join(sorted(scripts))


from collections import Counter
comp = Counter(classify(s) for s in talkie)
total = sum(comp.values())
print("=== talkie vocab composition by script (262k tokens) ===")
for cls, n in comp.most_common():
    print(f"  {cls:<22} {n:>7}  {100*n/total:5.2f}%")
nonlatin = total - comp["ascii-latin"] - comp["ascii-symbol/digit"] - comp["whitespace"]
print(f"  → non-ASCII (foreign-script) tokens: {nonlatin} "
      f"({100*nonlatin/total:.2f}%)")


# ====================================================================
# PART 2 — rank shift on shared, clean English words
# ====================================================================
# English dictionary filter (removes proper nouns? no — but removes
# non-words, foreign tokens, code fragments). Lowercased match.
words = set()
for p in ("/usr/share/dict/words",):
    try:
        with open(p) as f:
            words = {w.strip().lower() for w in f if w.strip().isalpha()}
        break
    except FileNotFoundError:
        pass

def english_word(s):
    """Token that is a leading-space English word, e.g. ' house'."""
    if not s.startswith(" "):
        return None
    core = s[1:]
    if not core.isalpha() or not core.isascii():
        return None
    if words and core.lower() not in words:
        return None
    return core.lower()

shared = []
seen = set()
for s, r in talkie.items():
    w = english_word(s)
    if w is None or s not in o200k or w in seen:
        continue
    seen.add(w)
    pt = r / talkie_max
    po = o200k[s] / o200k_max
    shared.append((w, pt, po, pt - po, r, o200k[s]))

print(f"\n=== rank shift on {len(shared)} shared clean English words ===")
print("Δpct < 0  → relatively MORE frequent in talkie (1900s-leaning)")
print("Δpct > 0  → relatively MORE frequent in o200k (modern-leaning)\n")

shared.sort(key=lambda x: x[3])
print("--- most 1900s-leaning (earlier in talkie) ---")
for w, pt, po, d, rt, ro in shared[:25]:
    print(f"  {w:<16} Δ={d:+.3f}  talkie {pt:.3f} (#{rt})  o200k {po:.3f} (#{ro})")
print("\n--- most modern-leaning (earlier in o200k) ---")
for w, pt, po, d, rt, ro in shared[-25:]:
    print(f"  {w:<16} Δ={d:+.3f}  talkie {pt:.3f} (#{rt})  o200k {po:.3f} (#{ro})")

# Curated thematic clusters — do related words move together?
CLUSTERS = {
    "horse-era transport": ["horse", "carriage", "coach", "saddle", "stable",
                            "railway", "steamer", "telegraph"],
    "domestic service": ["servant", "governess", "maid", "footman",
                          "scullery", "valet"],
    "empire & religion": ["empire", "colony", "providence", "heathen",
                          "sermon", "almighty", "providence"],
    "modern-ish": ["car", "phone", "movie", "video", "online", "data",
                   "user", "app", "internet"],
}
print("\n=== thematic clusters (mean Δpct; negative = 1900s-leaning) ===")
lut = {w: (pt, po, d) for w, pt, po, d, *_ in shared}
for name, ws in CLUSTERS.items():
    vals = [lut[w][2] for w in dict.fromkeys(ws) if w in lut]
    present = [w for w in dict.fromkeys(ws) if w in lut]
    if vals:
        print(f"  {name:<22} meanΔ={sum(vals)/len(vals):+.3f}  "
              f"n={len(vals)}  {present}")

# Write the full shared table for the visualization step.
with open(ROOT / "results" / "rankshift.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["word", "talkie_pct", "o200k_pct", "delta",
                 "talkie_rank", "o200k_rank"])
    for row in shared:
        wr.writerow([row[0], f"{row[1]:.5f}", f"{row[2]:.5f}",
                     f"{row[3]:.5f}", row[4], row[5]])
print(f"\nwrote results/rankshift.csv ({len(shared)} rows)")
