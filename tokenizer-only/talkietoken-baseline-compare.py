"""Compare the talkie vocab against the o200k (GPT-5) baseline vocab
borrowed from the tokenary project (baseline/o200k_tokens.csv), and
sweep the talkie vocab for post-1931 coinages.

A word that did not exist before 1931 cannot earn a BPE merge from a
pre-1931 corpus. Membership of such a word (as a whole token) is the
strongest single leakage signal available without model weights.
"""

import csv
from pathlib import Path

import talkie_tokenizer

ROOT = Path(__file__).resolve().parent.parent

t = talkie_tokenizer.from_pretrained()
talkie_vocab = set(t.get_vocab().keys())

o200k_vocab = set()
with open(ROOT / "baseline" / "o200k_tokens.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        o200k_vocab.add(row["token"])

print(f"talkie vocab entries: {len(talkie_vocab)}")
print(f"o200k  vocab entries: {len(o200k_vocab)}")
overlap = talkie_vocab & o200k_vocab
print(f"shared token strings: {len(overlap)} "
      f"({100 * len(overlap) / len(talkie_vocab):.1f}% of talkie)")

# --- Post-1931 coinage sweep -------------------------------------------
# Words coined AFTER 1930 (year of coinage noted). Pre-1931 text cannot
# contain them, so a whole-token hit is hard evidence of corpus leakage.
# Deliberately excludes words that existed pre-1931 in another sense
# (computer, television, hardware, Nazi, penicillin, Google/Barney Google).
COINAGES = {
    "radar": 1940, "laser": 1957, "nylon": 1938, "astronaut": 1955,
    "teenager": 1941, "genocide": 1944, "software": 1958, "bikini": 1946,
    "jeep": 1941, "quark": 1963, "hippie": 1965, "sitcom": 1964,
    "internet": 1974, "online": 1950, "website": 1993, "email": 1979,
    "smartphone": 1995, "laptop": 1984, "podcast": 2004, "blog": 1999,
    "selfie": 2002, "emoji": 1997, "hashtag": 2007, "bitcoin": 2008,
    "transistor": 1947, "antibiotic": 1941, "supersonic_jet": 9999,
    "megabyte": 1965, "microchip": 1959, "nanotech": 1987,
    "Blitzkrieg": 1939, "Pentagon": 1943, "UNESCO": 1945, "NATO": 1949,
    "Watergate": 1972, "Chatbot": 1994, "chatbot": 1994,
}

def variants(w):
    yield w
    yield " " + w
    yield w.capitalize()
    yield " " + w.capitalize()
    yield w.upper()
    yield " " + w.upper()

print("\n--- post-1931 coinage sweep (whole-token membership) ---")
hits = []
for word, year in sorted(COINAGES.items(), key=lambda kv: kv[1]):
    found = [v for v in set(variants(word)) if v in talkie_vocab]
    if found:
        hits.append((word, year, found))
        print(f"🚨 {word} (coined {year}): {found}")
    else:
        print(f"  ✓ {word} (coined {year}) — absent")

print(f"\ncoinage hits: {len(hits)}/{len(COINAGES)}")

# --- Encoding-efficiency comparison vs o200k ---------------------------
import tiktoken
o200k = tiktoken.get_encoding("o200k_base")

TERMS = [
    "computer", "television", "radar", "helicopter", "atomic bomb",
    "transistor", "neural network", "internet", "website", "email",
    "smartphone", "New Deal", "World War II", "United Nations", "NATO",
    "Microsoft", "Google", "Netflix", "telephone", "telegraph",
    "phonograph", "aeroplane", "wireless", "motor car", "typewriter",
]

print("\n--- token counts: talkie vs o200k (modern baseline) ---")
print(f"{'term':<20} {'talkie':>7} {'o200k':>7}")
flagged = []
for term in TERMS:
    a, b = len(t.encode(term)), len(o200k.encode(term))
    marker = ""
    if a <= b and term not in ("telephone", "telegraph", "phonograph",
                               "aeroplane", "wireless", "motor car",
                               "typewriter"):
        marker = "  ⚠️ as efficient as a modern-corpus tokenizer"
        flagged.append(term)
    print(f"{term:<20} {a:>7} {b:>7}{marker}")

print(f"\nmodern terms tokenized at least as efficiently as o200k: {flagged}")
