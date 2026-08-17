"""Counterfactual: does in-context code learning survive losing English?

didof.dev/blog/talkie-1930-llm-reasoning reports HumanEval pass@100 after
pasting working Python into the prompt, and reads it as evidence of
general in-context capacity.

Python is close to a best case for a 1930 English model. Its keywords are
English words whose ORDINARY MEANINGS MATCH THEIR SEMANTICS: return, if,
else, while, for..in, sum, len(gth), sort, print, open, read, class. A
model that has never seen code still knows what "return the larger" means
as English. Tokenized, "def mean(xs): return sum(xs)/len(xs)" arrives as
  def| mean|(|xs|):| return| sum|(|xs|)/|len|(|xs|)
— every content token a familiar English morpheme.

The counterfactual holds the TASK fixed and strips the English away:

  COBOL   all English words, post-1931       -> easiest, English control
  Python  English keywords + punctuation     -> the blog's case
  J       pure ASCII, ZERO English words     -> isolates "non-English"
  APL     non-ASCII glyphs, zero English     -> also byte-fallback

J matters because APL confounds two variables: no English AND no ASCII.
J is ASCII-only, so a J failure cannot be blamed on byte fallback.

Same task in every prompt (return the larger of two numbers), same
2-example format, so only the notation varies.

NOTE ON "o200k": o200k is a tokenizer, not a model — there is nothing to
generate from. The o200k-comparable artifact is the token split, produced
by talkietoken-python.py. Where a modern *model* comparand appears in the
write-up it is this assistant, labeled, exactly as in
results/neighborhoods-compare.md.

Writes results/lang-probes.txt and results/lang-probes.json.
"""

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://api.talkie-lm.com/api/chat/stream"
N = 3
GAP = 8

PREAMBLE = ("Here are two examples written in a notation for describing "
            "procedures.\n\n")
TASK = "\n\nNow write one more in the same notation, which gives back the "\
       "larger of two numbers."

PROBES = [
    ("incontext_cobol", "COBOL · all English",
     PREAMBLE +
     "COMPUTE RESULT = NUMBER-ONE * 2.\n"
     "COMPUTE RESULT = NUMBER-ONE + NUMBER-TWO." + TASK,
     {"kept_form": [r"COMPUTE", r"MOVE", r"IF "],
      "prose_instead": [r"\bI (do not|don't|cannot)\b", r"unfamiliar"]}),

    ("incontext_python", "Python · English keywords",
     PREAMBLE +
     "def double(x):\n    return x * 2\n\n"
     "def add(a, b):\n    return a + b" + TASK,
     {"kept_form": [r"def ", r"return"],
      "prose_instead": [r"\bI (do not|don't|cannot)\b", r"unfamiliar"]}),

    ("incontext_j", "J · ASCII, zero English",
     PREAMBLE +
     "double =: 2 & *\n"
     "add =: +" + TASK,
     {"kept_form": [r"=:", r">\.", r"\bmax\b"],
      "prose_instead": [r"\bI (do not|don't|cannot)\b", r"unfamiliar"]}),

    ("incontext_apl", "APL · non-ASCII glyphs",
     PREAMBLE +
     "double←{⍵×2}\n"
     "add←{⍺+⍵}" + TASK,
     {"kept_form": [r"←", r"⍵", r"⌈"],
      "prose_instead": [r"\bI (do not|don't|cannot)\b", r"unfamiliar"]}),

    # --- zero-shot: is any of this already known? -------------------
    ("zeroshot_forloop", "zero-shot · 'for loop'",
     "What is a for loop?",
     {"code_sense": [r"iterat", r"program", r"comput", r"variable", r"code"],
      "prose_sense": [r"weav", r"loom", r"thread", r"knit", r"rope", r"knot"]}),

    ("zeroshot_defline", "zero-shot · reading a def line",
     "What does this line mean: def add(a, b): return a + b",
     {"code_sense": [r"function", r"procedure", r"program", r"comput",
                     r"argument", r"defin"],
      "confab": [r"abbreviat", r"Latin", r"music", r"printer"]}),

    ("zeroshot_apl", "zero-shot · reading an APL line",
     "What does this line mean: sum←{+/⍵}",
     {"code_sense": [r"function", r"program", r"comput", r"sum of"],
      "confab": [r"symbol", r"astronom", r"chemi", r"alchem", r"rune",
                 r"cipher"]}),
]


def ask(prompt, max_tokens=130, temperature=0.7, retries=4):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API, data=body, headers={"Content-Type": "application/json"})
            text, event = [], None
            with urllib.request.urlopen(req, timeout=180) as r:
                for raw in r:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:") and event == "token":
                        try:
                            text.append(json.loads(line[5:].strip()))
                        except json.JSONDecodeError:
                            pass
            return "".join(text)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(45)
                continue
            return f"<error: {e}>"
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(15)
                continue
            return f"<error: {e}>"
    return "<error: retries exhausted>"


out, results = [], []


def emit(s=""):
    print(s, flush=True)
    out.append(s)


emit("=== COUNTERFACTUAL LANGUAGE PROBES (talkie chat API) ===")
emit(f"n={N} per prompt, temperature 0.7. Same task in every notation.")
emit("kept_form = the reply is in the demonstrated notation at all.")
emit()

for pid, label, prompt, themes in PROBES:
    emit(f"### {pid} — {label}")
    emit(f"PROMPT: {prompt!r}")
    samples, counts = [], {k: 0 for k in themes}
    for i in range(N):
        ans = ask(prompt)
        samples.append(ans)
        hit = {}
        for th, pats in themes.items():
            h = [p for p in pats if re.search(p, ans, re.I)]
            hit[th] = h
            if h:
                counts[th] += 1
        emit(f"  [{i+1}/{N}] {ans.strip()[:340]}")
        emit("        hits: " + (", ".join(f"{k}={v}" for k, v in hit.items()
                                           if v) or "none"))
        time.sleep(GAP)
    emit(f"  → {pid}: " + "  ".join(f"{k} {v}/{N}" for k, v in counts.items()))
    emit()
    results.append({"id": pid, "label": label, "prompt": prompt,
                    "n": N, "theme_counts": counts, "samples": samples})

(ROOT / "results" / "lang-probes.txt").write_text("\n".join(out) + "\n")
(ROOT / "results" / "lang-probes.json").write_text(json.dumps(results, indent=1))
emit("wrote results/lang-probes.txt and results/lang-probes.json")
