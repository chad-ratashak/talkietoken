"""Inject the atomic-age measurements into results/summary.html.

The HTML module's numbers are machine-written from results/atomic.json so
they cannot drift from the probe that produced them. Only the block
between the ATOMIC-DATA markers is rewritten; markup, CSS and render code
in summary.html are left alone.

Run after talkietoken-atomic.py:
    ../.venv/bin/python render-atomic-module.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "results" / "summary.html"
START, END = "/* ATOMIC-DATA:START */", "/* ATOMIC-DATA:END */"

atomic = json.loads((ROOT / "results" / "atomic.json").read_text())

# Behavioral probes are optional — the tokenizer module still renders
# without them (they need the hosted chat API to be reachable).
probes_path = ROOT / "results" / "atomic-probes.json"
probes = {p["id"]: p for p in json.loads(probes_path.read_text())} \
    if probes_path.exists() else {}

# Which probe to show, and which saved sample to quote. The quote text is
# pulled from the saved completion rather than retyped, so it stays
# traceable to results/atomic-probes.json.
SHOWCASE = [
    ("enola_gay", 0, "aircraft", "identified as an aircraft"),
    ("bockscar", 2, "aircraft", "identified as an aircraft"),
    ("fat_man", 0, "bomb", "read as a weapon"),
    ("manhattan_project", 0, "bomb", "read as a weapon"),
    ("oppenheimer", 1, "known", "the physicist"),
    ("szilard", 0, "known", "the physicist"),
    ("nagasaki_christians", 0, "christian", "the martyrdom history"),
    ("los_alamos", 2, "alamo_confusion", "confused with the Texas Alamo"),
]


def clip(s, n=190):
    s = " ".join(s.split())
    return s[:n].rstrip() + "…" if len(s) > n else s


behavioral = []
for pid, si, theme, theme_label in SHOWCASE:
    p = probes.get(pid)
    if not p:
        continue
    counts = p.get("theme_counts_rescored", p["theme_counts"])
    behavioral.append({
        "prompt": p["prompt"],
        "quote": clip(p["samples"][si]),
        "k": counts.get(theme, 0), "n": p["n"],
        "themeLabel": theme_label,
    })

COHORT_LABEL = {
    "kept": "famous before 1931 and still famous",
    "rose": "made famous by the war and after",
    "faded": "famous then, largely forgotten now",
}

# Cohort means, computed here rather than typed by hand.
means = {}
for cohort, rows in atomic["scientists"].items():
    means[cohort] = {
        "t": round(sum(r["talkie_n"] for r in rows) / len(rows), 4),
        "o": round(sum(r["o200k_n"] for r in rows) / len(rows), 4),
        "n": len(rows),
        "whole": sum(1 for r in rows if r["talkie_n"] == 1),
        "label": COHORT_LABEL[cohort],
    }


def slim(r):
    """Only the fields the page renders."""
    return {
        "term": r["term"], "note": r["note"],
        "talkie_n": r["talkie_n"], "o200k_n": r["o200k_n"],
        "talkie_pieces": r["talkie_pieces"], "o200k_pieces": r["o200k_pieces"],
        "common_word": r["common_word"],
    }


def slim_person(r):
    return {
        "term": r["term"], "talkie_n": r["talkie_n"],
        "o200k_n": r["o200k_n"], "common_word": r["common_word"],
    }


# Hypothesis verdicts. Prose lives here (data), not in the render code.
# `verdict` drives the badge colour: yes = confirmed, no = refuted,
# partial = mixed or not settleable by the vocabulary alone.
HYPOTHESES = [
    {
        "claim": "A 1930s model associates Nagasaki with the persecution of "
                 "Japanese Christians.",
        "verdict": "partial", "verdictLabel": "known, not foremost",
        "finding": "The history is there and comes out in full when asked: 3/3 "
                   "samples give Francis Xavier, the Jesuit mission of 1549, the "
                   "conversion counts and the persecution that followed. But it "
                   "is not the first association — asked plainly “What is "
                   "Nagasaki?”, 3/3 answer with the treaty port and 0/3 mention "
                   "Christians. The martyrdom is retrievable knowledge, not the "
                   "dominant sense. Either way 0/3 mention a bombing. In the "
                   "vocabulary Nagasaki is <b>one token in talkie against two in "
                   "o200k</b> — the port was more familiar to 1900s print than "
                   "the city is to the modern web.",
    },
    {
        "claim": "Enola Gay, Fat Man, Little Boy and the Manhattan Project are "
                 "read as ordinary noun phrases.",
        "verdict": "yes", "verdictLabel": "confirmed",
        "finding": "Every piece of Fat Man, Little Boy and Manhattan Project is "
                   "an ordinary dictionary word, and talkie spends exactly the "
                   "same tokens on each as o200k does — it is paying for the "
                   "words, not the referent. Behaviorally: 3/3 plain-noun "
                   "readings each, 0/3 any mention of a weapon. It does not "
                   "hesitate, it invents — the Enola Gay is given as a Scottish "
                   "liquid measure, as a lively Scottish song (reading <i>Gay</i> "
                   "in the period sense already charted above), and as a "
                   "Mediterranean herring. Bockscar, a pure post-war coinage, is "
                   "split into <i>boc</i> + <i>scar</i> and glossed as north-"
                   "country dialect for greenstone.",
    },
    {
        "claim": "Los Alamos is confused with the Alamo in Texas.",
        "verdict": "no", "verdictLabel": "refuted",
        "finding": "0/3 samples mention Texas, Santa Anna, Crockett or 1836. All "
                   "three describe Los Alamos, <b>California</b> — a real "
                   "post-town in Santa Barbara county — and one volunteers the "
                   "correct etymology, that the name signifies “the poplars”. "
                   "The vocabulary agrees: talkie holds <i>Alamo</i> whole but "
                   "shatters <i>Los Alamos</i> into Los|Al|amos. It is not "
                   "confusing two places; it knows a different, smaller one.",
    },
    {
        "claim": "Einstein stays famous; Fermi, Feynman and the Hungarians do not.",
        "verdict": "yes", "verdictLabel": "confirmed",
        "finding": "Einstein is one token in both vocabularies, the most "
                   "rank-stable name measured (Δ = −0.013), and behaviorally "
                   "solid at 3/3. Feynman and Szilard cost 3 tokens in talkie; "
                   "Fermi, Wigner, Ulam and Meitner 2 — and Fermi, Feynman and "
                   "Szilard all score 0/3. The cheap exceptions are spelling "
                   "coincidences rather than fame, which the behavioral probe "
                   "settles: <i>Teller</i> and <i>Groves</i> are ordinary words, "
                   "and <i>Oppenheimer</i> is one token only because it was a "
                   "common German-Jewish surname in period print. Asked who "
                   "Robert Oppenheimer is, the model returns a Cologne dramatist, "
                   "a Wiesbaden physician and a Berlin jurist — 0/3 physicists.",
    },
    {
        "claim": "Bohr — uncertain.",
        "verdict": "partial", "verdictLabel": "known, garbled",
        "finding": "Bohr is a single token in talkie where o200k needs two, so "
                   "the corpus saw the name — but at percentile 0.829 it is a "
                   "very late merge, the profile of a name that reaches print "
                   "without becoming a household word (Planck sits beside it at "
                   "0.848; Einstein is at 0.404). Behaviorally it is half-right: "
                   "2/3 place him as a Copenhagen man of science working on the "
                   "constitution of matter, but date his birth to 1873 rather "
                   "than 1885 and make him a chemist.",
    },
    {
        "claim": "The measure just reflects old tokenizers being bad at proper "
                 "nouns.",
        "verdict": "no", "verdictLabel": "refuted",
        "finding": "This is what the <b>faded</b> cohort controls for. All 10 of "
                   "Millikan, Eddington, Jeans, Michelson, Steinmetz, Haeckel, "
                   "Tyndall, Osler, Burbank and Bergson are a single token in "
                   "talkie, against a mean of 2.50 in o200k — the widest gap in "
                   "the table, running the opposite direction from the war-famous "
                   "cohort. The vocabulary tracks who was famous when, not name "
                   "rarity. The behavioral side is weaker than the vocabulary "
                   "side, though: Eddington comes back correct 3/3, but Millikan "
                   "only 1/3, the other two samples making him an Indiana country "
                   "doctor. A cheap token means the corpus saw the name — not "
                   "that the model reliably knows the person.",
    },
]

payload = {
    "places": [slim(r) for r in atomic["places"]],
    "hardware": [slim(r) for r in atomic["hardware"]],
    "scientists": {k: [slim_person(r) for r in v]
                   for k, v in atomic["scientists"].items()},
    "cohortMeans": means,
    "behavioral": behavioral,
    "hypotheses": HYPOTHESES,
}

block = (f"{START}\nconst atomic = "
         f"{json.dumps(payload, indent=1, ensure_ascii=False)};\n{END}")

html = HTML.read_text()
i, j = html.index(START), html.index(END) + len(END)
HTML.write_text(html[:i] + block + html[j:])

print(f"injected atomic module data into {HTML.relative_to(ROOT)}")
print(f"  places {len(payload['places'])} · hardware {len(payload['hardware'])} "
      f"· scientists {sum(len(v) for v in payload['scientists'].values())} "
      f"· hypotheses {len(HYPOTHESES)}")
for c, m in means.items():
    print(f"  {c:<6} talkie {m['t']:.2f} → o200k {m['o']:.2f}  "
          f"({m['whole']}/{m['n']} whole in talkie)")
