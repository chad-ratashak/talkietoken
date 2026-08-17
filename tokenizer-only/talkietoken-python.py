"""Did Python leak into the training corpus, or was it learned in-context?

Motivated by didof.dev/blog/talkie-1930-llm-reasoning, which reports
HumanEval pass@100 after pasting working Python examples into the prompt,
and asserts the model "has not seen a `def` or a `for` loop in 260
billion tokens".

Why the tokenizer is the right instrument
-----------------------------------------
In-context learning leaves NO trace in the vocabulary: code pasted into a
prompt cannot retroactively create a BPE merge. Corpus leakage does leave
a trace — BPE merges are fossilised corpus statistics. So:

    Python-specific merges present  -> strong evidence of corpus leakage
    Python-specific merges absent   -> consistent with in-context learning

The test is ONE-SIDED. A clean vocabulary does not prove a clean model:
the tokenizer may be trained on a different (or a filtered) corpus than
the weights. It bounds the tokenizer's corpus, nothing more.

The confound that eats naive versions of this test
--------------------------------------------------
Almost every Python keyword is an ordinary English word: if, else, for,
while, class, return, import, from, pass, break, continue, try, raise,
with, as, in, is, not, and, or, del, global, yield. A 1930 corpus merges
all of them for reasons that have nothing to do with code. Worse, BPE
merges sub-word FRAGMENTS: " def" is a word-initial piece of " define",
" definite", " defend", " default" — so finding " def" says nothing
either. Same trap as the codename module: cheapness is the word, not the
referent.

So evidence is tiered by how much prose could explain it:

    TIER 0  English words that happen to be Python keywords  (no signal)
    TIER 1  strings prose cannot produce                     (real signal)
    TIER 2  ecosystem names, minus the English-word decoys    (real signal)
    TIER 3  code structure: indentation, fences, call syntax  (real signal)
    TIER 4  encoding efficiency on real Python source         (magnitude)

Writes results/python-leak.txt.
"""

from pathlib import Path

import baselines
import talkie_tokenizer

ROOT = Path(__file__).resolve().parent.parent

t = talkie_tokenizer.from_pretrained()
talkie = t.get_vocab()
talkie_max = max(talkie.values())

encs = baselines.load_baselines()
o200k = encs["o200k"]
o200k_vocab = {
    b.decode("utf-8", errors="replace"): r
    for b, r in o200k._mergeable_ranks.items()  # noqa: SLF001
}
o200k_max = max(o200k_vocab.values())

DICT = set()
try:
    with open("/usr/share/dict/words") as f:
        DICT = {w.strip().lower() for w in f if w.strip().isalpha()}
except FileNotFoundError:
    pass

out = []


def emit(s=""):
    print(s)
    out.append(s)


def status(s, vocab, vmax):
    if s in vocab:
        return f"yes ({vocab[s] / vmax:.3f})"
    return "no"


def table(rows, note_width=44):
    emit(f"  {'token':<16} {'talkie':>14} {'o200k':>14}   why it may not count")
    for s, why in rows:
        emit(f"  {s!r:<16} {status(s, talkie, talkie_max):>14} "
             f"{status(s, o200k_vocab, o200k_max):>14}   {why[:note_width]}")


emit("=== DID PYTHON LEAK INTO THE CORPUS, OR WAS IT LEARNED IN-CONTEXT? ===")
emit("'yes (p)' = whole token, p = percentile merge rank (0 = earliest).")
emit("In-context learning cannot create a merge; corpus leakage must.")
emit()

# --- TIER 0 ----------------------------------------------------------
emit("--- TIER 0: Python keywords that are ordinary English words ---")
emit("Expected present in BOTH. Carries NO information about code.")
kw = ["if", "else", "for", "while", "class", "return", "import", "from",
      "pass", "break", "continue", "try", "raise", "with", "as", "in",
      "is", "not", "and", "or", "del", "global", "yield", "assert"]
present = [w for w in kw if (" " + w) in talkie]
emit(f"  {len(present)}/{len(kw)} present in talkie as ' word': {present}")
emit(f"  all are dictionary words: "
     f"{all(w in DICT for w in kw if w not in ('elif',))}")
emit("  → as expected, and worthless as evidence either way.")
emit()

# --- TIER 1 ----------------------------------------------------------
emit("--- TIER 1: strings prose cannot plausibly produce ---")
table([
    ("elif", "1991 coinage, not an English word — the key probe"),
    (" elif", "same, word-initial form"),
    (" Elif", "DECOY: a common Turkish given name"),
    ("__init__", "dunder; prose has no double underscores"),
    ("__name__", "dunder"),
    ("__main__", "dunder"),
    ("self.", "attribute access"),
    (".append", "method call"),
    (".join", "method call"),
    ("()", "empty call — see tier 3"),
    ("():", "def/class line ending"),
    (" lambda", "Greek letter in prose, keyword in code"),
    (" nonlocal", "2008 coinage"),
    (" def", "DECOY: prefix of define/defend/default/deficient"),
])
emit()

# --- TIER 2 ----------------------------------------------------------
emit("--- TIER 2: ecosystem names (English-word decoys marked) ---")
libs = [(" numpy", ""), (" pandas", "DECOY: the animals"),
        (" matplotlib", ""), (" django", ""),
        (" flask", "DECOY: a bottle"), (" pytest", ""),
        (" sklearn", ""), (" jupyter", "DECOY: Jupiter/Jupyter"),
        (" tensor", "DECOY: 1846 maths term"), (" pip", "DECOY: a seed"),
        (" json", ""), (" stdout", "")]
table([(s, why or "no prose reading") for s, why in libs])
real = [s for s, why in libs if s in talkie and not why]
emit(f"  → non-decoy library hits in talkie: {real or 'NONE'}")
emit()

# --- TIER 3 ----------------------------------------------------------
emit("--- TIER 3: code structure ---")
table([
    ("```", "markdown code fence"),
    ("):", "end of a def/if line"),
    ("->", "type-hint arrow / return annotation"),
    ("==", "equality operator"),
    ("!=", "inequality operator"),
    ("+=", "augmented assignment"),
    ("):\n", "line ending inside a block"),
])
emit()
emit("  indentation runs (code merges these; prose only for verse/tables):")
emit(f"  {'run':<16} {'talkie':>14} {'o200k':>14}")
for n in (2, 4, 8, 12, 16):
    s = " " * n
    emit(f"  {str(n) + ' spaces':<16} {status(s, talkie, talkie_max):>14} "
         f"{status(s, o200k_vocab, o200k_max):>14}")
for lab, s in [("\\n + 4sp", "\n    "), ("\\n + 8sp", "\n        "),
               ("\\n + tab", "\n\t"), ("tab", "\t")]:
    emit(f"  {lab:<16} {status(s, talkie, talkie_max):>14} "
         f"{status(s, o200k_vocab, o200k_max):>14}")
emit()
emit("  underscore-bearing tokens (identifiers vs. Gutenberg _italics_):")
tu = [s for s in talkie if "_" in s]
ou = [s for s in o200k_vocab if "_" in s]
emit(f"    talkie {len(tu):>6} / {len(talkie)} = {100*len(tu)/len(talkie):.3f}%")
emit(f"    o200k  {len(ou):>6} / {len(o200k_vocab)} = "
     f"{100*len(ou)/len(o200k_vocab):.3f}%")
alpha_t = [s for s in tu if any(c.isalpha() for c in s) and "__" not in s]
emit(f"    talkie underscore tokens containing letters: {len(alpha_t)}")
emit(f"      sample: {sorted(alpha_t)[:12]}")
emit()

# --- TIER 3b: run down the three hits that look like code ------------
emit("--- TIER 3b: the three talkie hits that look like code, checked ---")
emit("Each has a prose explanation. A story is not evidence, so each is")
emit("tested against the siblings it would have if the story were true.")
emit()
emit("  '==' as a section divider, not an operator — the ASCII-rule family:")
emit("    " + "  ".join(
    f"{s!r}={status(s, talkie, talkie_max).split()[0]}"
    f"{'/' + f'{talkie[s]/talkie_max:.3f}' if s in talkie else ''}"
    for s in ["==", "===", "====", "--", "---", "****", "____", "...."]))
emit("    Leader dots and rules are early merges throughout; '==' sits")
emit("    inside that family, where a Gutenberg text puts its separators.")
emit()
emit("  '->' as a typographic arrow, not a return annotation:")
emit("    " + "  ".join(
    f"{s!r}={talkie[s]/talkie_max:.3f}" if s in talkie else f"{s!r}=absent"
    for s in ["->", "<-", "=>", "-->", "<--"]))
emit("    '<-' is as early as '->'. In Python '->' is syntax and '<-' is")
emit("    not, so a code corpus separates them; a corpus of chemistry")
emit("    arrows and cross-references does not. They are not separated.")
emit()
emit("  ' def' as a word-initial prefix fragment, not the keyword:")
emit("    peers: " + "  ".join(
    f"{s.strip()}={talkie[s]/talkie_max:.3f}"
    for s in [" def", " con", " pre", " dis", " int", " imp", " ret"]
    if s in talkie))
defs = sorted([s for s in talkie if s.startswith(" def")],
              key=lambda s: talkie[s])[:12]
emit(f"    earliest ' def…' merges: {defs}")
emit("    Every continuation is prose. No ' def ' followed by an identifier,")
emit("    no '():'. The fragment exists because 1930 English says 'define',")
emit("    'defendant' and 'defence' constantly.")
emit()

# --- TIER 4 ----------------------------------------------------------
emit("--- TIER 4: encoding efficiency on real Python ---")
emit("If the corpus never held code, the tokenizer must spell it out.")
SAMPLES = {
    "HumanEval-style function": '''def has_close_elements(numbers, threshold):
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
''',
    "class with dunder": '''class Counter:
    def __init__(self, start=0):
        self._count = start

    def increment(self, by=1):
        self._count += by
        return self._count
''',
    "period English prose (control)": (
        "The wireless operator, having received the message, "
        "carried it directly to the captain upon the bridge, "
        "who read it twice before making any reply.\n"),
}
emit()
emit(f"  {'sample':<32} {'talkie':>8} {'o200k':>8} {'gpt2':>8} {'ratio':>8}")
for name, src in SAMPLES.items():
    a = len(t.encode(src))
    b = len(o200k.encode(src))
    g = len(encs["gpt2"].encode(src))
    emit(f"  {name:<32} {a:>8} {b:>8} {g:>8} {a/b:>7.2f}x")
emit()
emit("  ratio = talkie tokens per o200k token. ~1.0 on prose means the two")
emit("  are equally at home in English; a large ratio on code means talkie")
emit("  is spelling out something it has no merges for.")
emit()

# Show the actual shattering of one line.
line = "    def __init__(self, start=0):"
emit(f"  line: {line!r}")
emit(f"    talkie ({len(t.encode(line))} tokens): "
     f"{'|'.join(t.decode([i]) for i in t.encode(line))}")
emit(f"    o200k  ({len(o200k.encode(line))} tokens): "
     f"{'|'.join(o200k.decode([i]) for i in o200k.encode(line))}")
emit()


# --- TIER 5: the counterfactual language spectrum ---------------------
emit("--- TIER 5: what if the language were not made of English? ---")
emit("Python is close to a BEST case for a 1930 English model: its")
emit("keywords are English words whose ordinary meanings match their")
emit("semantics (return, if, while, for..in, sum, len, sort, open, class).")
emit("Same task in every notation; only the surface language changes.")
emit()

SPECTRUM = {
    "SQL": ("total", "SELECT name, MAX(score) FROM players "
            "WHERE active = 1 GROUP BY name;\n"),
    "COBOL": ("total", "       PROCEDURE DIVISION.\n"
              "           MOVE ZERO TO WS-MAX.\n"
              "           PERFORM VARYING I FROM 1 BY 1 UNTIL I > N\n"
              "               IF ITEM(I) > WS-MAX THEN "
              "MOVE ITEM(I) TO WS-MAX\n"
              "               END-IF\n           END-PERFORM.\n"),
    "Python": ("high", "def largest(items):\n    best = items[0]\n"
               "    for item in items:\n        if item > best:\n"
               "            best = item\n    return best\n"),
    "Rust": ("some", "fn largest<T: PartialOrd>(items: &[T]) -> &T {\n"
             "    let mut best = &items[0];\n"
             "    for item in items { if item > best { best = item; } }\n"
             "    best\n}\n"),
    "Haskell": ("little", "largest :: Ord a => [a] -> a\n"
                "largest = foldr1 (\\x y -> if x > y then x else y)\n"),
    "Perl": ("little", "my $best = $items[0];\n"
             "foreach my $x (@$items) { $best = $x if $x > $best; }\n"),
    "J": ("NONE", "largest =: >./\nmean =: +/ % #\nsort =: /:~\n"),
    "APL": ("NONE", "largest←{⌈/⍵}\nmean←{(+/⍵)÷≢⍵}\n"),
}
emit(f"  {'language':<10}{'English':>8}{'chars':>7}{'talkie':>8}{'o200k':>7}"
     f"{'ratio':>8}{'tok/char':>10}")
for name, (eng, src) in SPECTRUM.items():
    a, b = len(t.encode(src)), len(o200k.encode(src))
    emit(f"  {name:<10}{eng:>8}{len(src):>7}{a:>8}{b:>7}{a/b:>7.2f}x"
         f"{a/len(src):>10.2f}")
emit()
emit("  ratio vs o200k is nearly flat — it hides the effect, because o200k")
emit("  has no merges for J or APL either. tok/char is the discriminating")
emit("  measure, and it orders the languages by how much English they are")
emit("  made of. talkie BEATS o200k on COBOL and SQL: those are English.")
emit()

# --- worked splits ---------------------------------------------------
emit("--- worked splits: Python-specific syntax vs a non-English language ---")
emit("'|' marks a token boundary. Note WHAT the pieces are: in Python the")
emit("content tokens are English morphemes the model already understands.")
emit()
SNIPPETS = [
    ("Python", "def largest(a, b):"),
    ("Python", "    return a if a > b else b"),
    ("Python", "[x**2 for x in range(10)]"),
    ("Python", "with open('data.txt') as fh:"),
    ("Python", "except ValueError as err:"),
    ("Python", "@property"),
    ("Python", "self.items.append(value)"),
    ("Python", "lambda x: x + 1"),
    ("J", "largest =: >./"),
    ("J", "mean =: +/ % #"),
    ("APL", "largest←{⌈/⍵}"),
    ("APL", "mean←{(+/⍵)÷≢⍵}"),
]
def show(enc, i):
    """Render one token. A token that is not valid UTF-8 on its own is a
    raw byte from the fallback path — show it as <hex> rather than as a
    replacement character, so byte fallback is visible instead of looking
    like a mojibake bug."""
    b = enc.decode_single_token_bytes(i)
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"<{c:02x}>" for c in b)


for lang, s in SNIPPETS:
    ta, oa = t.encode(s), o200k.encode(s)
    emit(f"  [{lang}] {s!r}")
    emit(f"      talkie {len(ta):>3}: "
         + "|".join(show(t.enc, i) for i in ta))
    emit(f"      o200k  {len(oa):>3}: "
         + "|".join(show(o200k, i) for i in oa))
emit()
emit("  The Python rows break into def / return / if / else / for / in /")
emit("  open / with / range / value — words a 1930 corpus knows, carrying")
emit("  roughly the right meaning. The J and APL rows break into single")
emit("  characters and raw bytes carrying none. That, rather than any")
emit("  leaked code, is the likeliest reason in-context Python works.")

# --- structured dump for the HTML module -----------------------------
import json  # noqa: E402 — kept next to the dump it serves

LEAK_TIERS = [
    ("elif", "1991 coinage; no English reading at all"),
    ("():", "def/class line ending"),
    (".append", "method call"),
    (".join", "method call"),
    ("__init__", "dunder"),
    (" numpy", "library"),
    (" matplotlib", "library"),
    (" django", "library"),
    (" json", "library"),
    (" stdout", "library"),
    (" lambda", "keyword sense"),
    (" nonlocal", "2008 coinage"),
    ("```", "markdown code fence"),
]
DECOYS = [
    (" def", "prefix of define / defend / default / deficient"),
    (" Elif", "a common Turkish given name"),
    (" flask", "a bottle"),
    (" pip", "a seed"),
    (" tensor", "an 1846 mathematics term"),
    ("==", "Gutenberg section rule, beside ==== and ...."),
    ("->", "typographic arrow; '<-' is just as early"),
]

dump = {
    "spectrum": [
        {"lang": name, "english": eng, "chars": len(src),
         "talkie": len(t.encode(src)), "o200k": len(o200k.encode(src)),
         "tok_per_char": round(len(t.encode(src)) / len(src), 4)}
        for name, (eng, src) in SPECTRUM.items()
    ],
    "splits": [
        {"lang": lang, "src": s,
         "talkie": [show(t.enc, i) for i in t.encode(s)],
         "o200k": [show(o200k, i) for i in o200k.encode(s)]}
        for lang, s in SNIPPETS
    ],
    "leak": [
        {"token": s, "note": why,
         "talkie": s in talkie, "o200k": s in o200k_vocab,
         "o200k_pct": round(o200k_vocab[s] / o200k_max, 3)
                      if s in o200k_vocab else None}
        for s, why in LEAK_TIERS
    ],
    "decoys": [
        {"token": s, "note": why,
         "talkie_pct": round(talkie[s] / talkie_max, 3) if s in talkie else None}
        for s, why in DECOYS
    ],
    "keywords_english": {
        "n_present": len(present), "n_total": len(kw), "words": present,
    },
    "underscore": {
        "talkie_pct": round(100 * len(tu) / len(talkie), 3),
        "o200k_pct": round(100 * len(ou) / len(o200k_vocab), 3),
        "talkie_sample": sorted(alpha_t)[:10],
    },
}
(ROOT / "results" / "python-leak.json").write_text(json.dumps(dump, indent=1))
(ROOT / "results" / "python-leak.txt").write_text("\n".join(out) + "\n")
emit()
emit("wrote results/python-leak.txt and results/python-leak.json")
