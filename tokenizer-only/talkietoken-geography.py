"""Geography / polity / script-reform probes at the tokenizer level.

The 1931 cutoff sits right on top of a cluster of renamings, alphabet
reforms, and short-lived polities. A pre-1931 English corpus should merge
the OLD name (it was the standard usage) and lack the MODERN one. Whole-
token membership of an old name + absence of its modern replacement is a
positive signal for the claim; the reverse would be leakage.

Tokenizer-level only (no model weights, no API). Companion generation
probes live in ../talkietoken-genprobe2-api.py.
"""

import talkie_tokenizer
import tiktoken
from pathlib import Path
import csv

t = talkie_tokenizer.from_pretrained()
v = set(t.get_vocab().keys())

ROOT = Path(__file__).resolve().parent.parent
o200k_strings = set()
with open(ROOT / "baseline" / "o200k_tokens.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        o200k_strings.add(row["token"])


def variants(w):
    return {w, " " + w, w.lower(), " " + w.lower()}


def merged(w):
    """Is any cased/spaced variant a single whole token?"""
    hits = [x for x in variants(w) if x in v]
    return hits


def report(title, pairs):
    print(f"\n=== {title} ===")
    print(f"{'OLD (pre-1931)':<22}{'merged?':<10}{'MODERN':<20}{'merged?':<10}")
    for old, new, note in pairs:
        oh, nh = merged(old), merged(new)
        om = "yes " + str(oh) if oh else "no"
        nm = "yes " + str(nh) if nh else "no"
        flag = ""
        if nh and not oh:
            flag = "  🚨 modern merged, old absent — LEAK signal"
        print(f"{old:<22}{om:<10}{new:<20}{nm:<10}{flag}")
        if note:
            print(f"    └ {note}")


# (old name, modern name, note). Old should merge; modern should be absent.
PRUSSIA = [
    ("Königsberg", "Kaliningrad", "renamed 1946"),
    ("Breslau", "Wrocław", "became Polish 1945"),
    ("Danzig", "Gdańsk", "Free City 1920–39; Polish 1945"),
    ("Stettin", "Szczecin", "became Polish 1945"),
    ("Memel", "Klaipėda", "Memelland; Lithuanian"),
    ("Posen", "Poznań", "Prussian to 1919"),
]

COLONIAL_RENAMES = [
    ("Constantinople", "Istanbul", "name formalized 1930 — on the cutoff"),
    ("Angora", "Ankara", "western usage shifted ~1930"),
    ("Siam", "Thailand", "renamed 1939"),
    ("Persia", "Iran", "renamed 1935"),
    ("Abyssinia", "Ethiopia", "common western name pre-WWII"),
    ("Ceylon", "Sri Lanka", "renamed 1972"),
    ("Bombay", "Mumbai", "renamed 1995"),
    ("Peking", "Beijing", "Peking was standard pre-1931"),
    ("Christiania", "Oslo", "renamed Jan 1925 — inside the window"),
]

# Short-lived / window-specific polities.
# KNOWN side existed pre-1931; UNKNOWN side is post-1930 (negative controls).
POLITIES_KNOWN = [
    "Tannu Tuva", "Yugoslavia", "Transjordan", "Mesopotamia",
    "Manchuria", "Bessarabia", "Ruthenia", "Tanganyika",
    "Somaliland",  # British/Italian/French — no unified "Somalia" until 1960
]
POLITIES_POST1930 = [
    ("Manchukuo", 1932), ("Pakistan", 1947), ("Israel", 1948),
    ("Bangladesh", 1971), ("Zimbabwe", 1980), ("Eritrea", 1993),
    ("Somalia", 1960),
]

report("Former Prussia / German East", PRUSSIA)
report("Colonial-era renames", COLONIAL_RENAMES)

print("\n=== Window polities that SHOULD be known (existed pre-1931) ===")
for p in POLITIES_KNOWN:
    print(f"  {p:<14} merged={merged(p)}  tokens={len(t.encode(p))}")

print("\n=== Polities founded AFTER 1930 (whole-token = leak signal) ===")
for p, year in POLITIES_POST1930:
    h = merged(p)
    flag = "  🚨 merged — founded after cutoff!" if h else ""
    print(f"  {p:<12}(founded {year})  merged={h}  tokens={len(t.encode(p))}{flag}")

# Bird names — common-name standardizations (weaker signal; mostly post-1980).
print("\n=== Bird common names (old vs modern AOU/IOC name) ===")
BIRDS = [
    ("Oldsquaw", "Long-tailed Duck", "renamed 2000"),
    ("Marsh Hawk", "Northern Harrier", "renamed ~1983"),
    ("Sparrow Hawk", "American Kestrel", "old US usage"),
    ("Baltimore Oriole", "Northern Oriole", "lumped 1973, re-split 1995"),
]
report("Bird names", BIRDS)

# Emoji confirmation (expected: pure byte fallback, no merges).
print("\n=== Emoji (expect byte-fallback, no whole-token merges) ===")
for e in ["🤖", "🔥", "🌍", "👋"]:
    print(f"  {e}  in-vocab={e in v}  tokens={len(t.encode(e))} (=raw UTF-8 bytes)")
