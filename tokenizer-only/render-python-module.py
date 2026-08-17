"""Inject the Python / counterfactual-language module into summary.html.

Reads results/python-leak.json (tokenizer measurements) and
results/lang-probes.json (generations), and rewrites only the block
between the PY-DATA markers.

Run after:
    ../.venv/bin/python talkietoken-python.py
    ../.venv/bin/python ../talkietoken-langprobes.py
    ../.venv/bin/python render-python-module.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "results" / "summary.html"
START, END = "/* PY-DATA:START */", "/* PY-DATA:END */"

leak = json.loads((ROOT / "results" / "python-leak.json").read_text())

probes_path = ROOT / "results" / "lang-probes.json"
probes = {p["id"]: p for p in json.loads(probes_path.read_text())} \
    if probes_path.exists() else {}


def clip(s, n=200):
    s = " ".join(s.split())
    return s[:n].rstrip() + "…" if len(s) > n else s


# (probe id, label, samples to quote, score line, reading)
# `read` is the honest interpretation — the keyword counters measure
# whether the notation was imitated, NOT whether the program is correct,
# and on this evidence those two come apart completely.
SHOWCASE = [
    ("zeroshot_forloop", "Zero-shot: “What is a for loop?”", [0, 1],
     "0/3 any computing sense",
     "No concept of a loop in the programming sense exists. This is the "
     "blog's premise holding up under a direct test."),
    ("zeroshot_defline",
     "Zero-shot: “What does this line mean: def add(a, b): return a + b”",
     [0, 1],
     "3/3 correct",
     "It reads the line correctly — and does it as ENGLISH. add, return "
     "and a + b are ordinary words and arithmetic. This is the whole "
     "argument in one probe: a model with no idea what a loop is still "
     "parses a Python function, because Python is written in English."),
    ("zeroshot_apl", "Zero-shot: “What does this line mean: sum←{+/⍵}”",
     [0, 2],
     "0/3 correct",
     "Strip the English and only the single word “sum” survives; the rest "
     "is invented. The glyphs cost three raw bytes each and carry nothing."),
    ("incontext_python", "In-context, Python — 2 examples, then a new task",
     [0, 1],
     "3/3 kept the form · 0/3 correct",
     "It reproduces the shape of Python and gets the logic wrong: the task "
     "was the larger of two numbers. Form and correctness come apart. This "
     "is n=3 at pass@1 and is NOT a refutation of the blog's pass@100, "
     "which samples 100 times and counts any success."),
    ("incontext_j", "In-context, J — ASCII, zero English words",
     [0, 1],
     "0/3 kept the form",
     "Remove the English and imitation collapses: it drifts back into prose "
     "arithmetic. J is the controlled counterfactual because it is ASCII, "
     "so this cannot be blamed on byte fallback."),
    ("incontext_apl", "In-context, APL — non-ASCII glyphs",
     [0, 1],
     "3/3 kept the form · 0/3 correct",
     "The surprise: APL's template is so rigid that the shape is copied "
     "perfectly — and every sample answers “subtract” to a “larger of two” "
     "task. Pure surface mimicry with no semantics behind it."),
    ("incontext_cobol", "In-context, COBOL — all English words", [0],
     "3/3 kept the form",
     "The English end of the spectrum. talkie also tokenizes COBOL more "
     "cheaply than o200k does (0.28 vs 0.29 tokens/char)."),
]

generation = []
for pid, label, idxs, score, read in SHOWCASE:
    p = probes.get(pid)
    if not p:
        continue
    generation.append({
        "label": label,
        "prompt": "",
        "quotes": [clip(p["samples"][i]) for i in idxs
                   if i < len(p["samples"])],
        "score": score,
        "read": read,
    })

payload = dict(leak)
payload["generation"] = generation

block = (f"{START}\nconst py = "
         f"{json.dumps(payload, indent=1, ensure_ascii=False)};\n{END}")

html = HTML.read_text()
i, j = html.index(START), html.index(END) + len(END)
HTML.write_text(html[:i] + block + html[j:])

print(f"injected python module into {HTML.relative_to(ROOT)}")
print(f"  leak probes {len(payload['leak'])} "
      f"(hits in talkie: {sum(1 for r in payload['leak'] if r['talkie'])}) "
      f"· decoys {len(payload['decoys'])}")
print(f"  spectrum {len(payload['spectrum'])} · splits {len(payload['splits'])} "
      f"· generation {len(generation)}")
