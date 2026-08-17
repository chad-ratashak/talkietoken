"""Behavioral probes for the atomic-age hypotheses.

The tokenizer says what a corpus *saw*; it cannot say what the model
*thinks a name means*. These four hypotheses are semantic, so they need
generation:

  H1  Nagasaki reads as a treaty port / Christian-martyrdom city, not as
      a bomb target.
  H2  Enola Gay, Fat Man, Little Boy, Manhattan Project read as ordinary
      noun phrases (a woman's name, a stout man, a child, a building
      scheme in New York).
  H3  Los Alamos is confused with the Alamo in Texas.
  H4  Fame gradient: Einstein known; Fermi/Feynman/Szilard not; Bohr
      uncertain; faded-cohort names (Eddington, Millikan) still known.

Each prompt is asked N times (the endpoint is a chat API at nonzero
temperature, so a single sample is not a verdict — see TODO.md, statistical
rigor). Keyword hits are counted automatically so the verdict does not
rest on eyeballing, and every raw completion is written to the log.

Writes results/atomic-probes.txt and results/atomic-probes.json.
"""

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://api.talkie-lm.com/api/chat/stream"
N = 3            # samples per prompt
GAP = 8          # seconds between requests (endpoint is rate-limited)

# (id, hypothesis, prompt, {label: [keywords]})
PROBES = [
    ("nagasaki", "H1", "What is Nagasaki?", {
        "port/trade": ["port", "harbour", "harbor", "seaport", "trade", "treaty"],
        "christian": ["christian", "martyr", "jesuit", "missionary", "catholic",
                      "persecut", "crucif", "papist", "urakami"],
        "bomb": ["bomb", "atomic", "nuclear", "destroyed", "1945"],
    }),
    ("nagasaki_christians", "H1",
     "Tell me the history of the Christians of Nagasaki.", {
         "christian": ["martyr", "persecut", "crucif", "jesuit", "missionary",
                       "hidden", "kakure", "xavier", "urakami"],
         "bomb": ["bomb", "atomic", "nuclear", "1945"],
     }),
    ("hiroshima", "H1", "What is Hiroshima?", {
        "city/castle": ["castle", "city", "port", "province", "garrison",
                        "prefecture", "military"],
        "bomb": ["bomb", "atomic", "nuclear", "destroyed", "1945"],
    }),
    ("los_alamos", "H3", "Tell me about Los Alamos.", {
        "alamo_confusion": ["texas", "alamo", "santa anna", "davy crockett",
                            "bowie", "siege", "1836", "mission"],
        "new_mexico": ["new mexico", "mexico", "ranch", "mesa", "pueblo",
                       "plateau", "cottonwood"],
        "bomb": ["bomb", "atomic", "nuclear", "laboratory", "weapon"],
    }),
    ("enola_gay", "H2", "What is the Enola Gay?", {
        "plain_noun": ["name", "woman", "girl", "person", "unfamiliar",
                       "not know", "unknown", "merry", "cheerful", "lively"],
        "aircraft": ["aircraft", "aeroplane", "airplane", "bomber", "b-29",
                     "plane", "flight"],
    }),
    ("fat_man", "H2", "What is meant by 'Fat Man'?", {
        "plain_noun": ["stout", "corpulent", "obese", "person", "man who",
                       "heavy", "portly", "fleshy"],
        "bomb": ["bomb", "atomic", "nuclear", "weapon", "plutonium"],
    }),
    ("little_boy", "H2", "What is meant by 'Little Boy'?", {
        "plain_noun": ["child", "young", "lad", "infant", "boy of", "small boy"],
        "bomb": ["bomb", "atomic", "nuclear", "weapon", "uranium"],
    }),
    ("manhattan_project", "H2", "What is the Manhattan Project?", {
        "plain_noun": ["new york", "island", "borough", "city", "scheme",
                       "undertaking", "building", "bridge", "tunnel"],
        "bomb": ["bomb", "atomic", "nuclear", "weapon", "uranium"],
    }),
    ("bockscar", "H2", "What is Bockscar?", {
        "unknown": ["not know", "unfamiliar", "unknown", "no knowledge",
                    "cannot say", "never heard"],
        "aircraft": ["aircraft", "aeroplane", "bomber", "plane", "b-29"],
    }),
    ("indianapolis", "H2", "Tell me about the USS Indianapolis.", {
        "ship": ["cruiser", "ship", "vessel", "navy", "warship", "launched"],
        # "torpedo" deliberately excluded: every cruiser of the period
        # carried torpedo tubes, so it appears in routine armament lists
        # and cannot distinguish the 1945 sinking from a stock description.
        "sinking": ["sunk", "sank", "shark", "1945", "survivors"],
    }),
    # --- H4 fame gradient -------------------------------------------
    # "known" keywords are DOMAIN terms only. Both "physicist" and
    # "physics" are listed explicitly: an early pass scored Millikan 0/3
    # for writing "professor of physics", and a bare "physic" stem was
    # worse — it matched "physician", scoring a confabulated Cologne
    # doctor as knowledge of Robert Oppenheimer.
    # Nationality words are deliberately excluded: the model infers
    # "Hungarian" from the shape of the name Szilard, so a nationality hit
    # is evidence about spelling, not about the person.
    ("einstein", "H4", "Who is Albert Einstein?", {
        "known": ["relativity", "physicist", "physics", "nobel"],
        "unknown": ["not know", "unfamiliar", "unknown", "never heard"],
    }),
    ("bohr", "H4", "Who is Niels Bohr?", {
        "known": ["physicist", "physics", "quantum", "spectroscop"],
        "unknown": ["not know", "unfamiliar", "unknown", "never heard"],
    }),
    ("fermi", "H4", "Who is Enrico Fermi?", {
        "known": ["physicist", "physics", "nobel", "nuclear", "neutron"],
        "unknown": ["not know", "unfamiliar", "unknown", "never heard"],
    }),
    ("feynman", "H4", "Who is Richard Feynman?", {
        "known": ["physicist", "physics", "nobel", "quantum", "electrodynam"],
        "unknown": ["not know", "unfamiliar", "unknown", "never heard"],
    }),
    ("szilard", "H4", "Who is Leo Szilard?", {
        "known": ["physicist", "physics", "nuclear", "chain reaction"],
        "unknown": ["not know", "unfamiliar", "unknown", "never heard"],
    }),
    ("oppenheimer", "H4", "Who is Robert Oppenheimer?", {
        "known": ["physicist", "physics", "nuclear", "berkeley"],
        "unknown": ["not know", "unfamiliar", "unknown", "never heard"],
    }),
    ("eddington", "H4", "Who is Arthur Eddington?", {
        "known": ["astronom", "eclipse", "relativity", "star", "cambridge"],
        "unknown": ["not know", "unfamiliar", "unknown", "never heard"],
    }),
    ("millikan", "H4", "Who is Robert Millikan?", {
        "known": ["physicist", "physics", "electron", "nobel", "oil drop"],
        "unknown": ["not know", "unfamiliar", "unknown", "never heard"],
    }),
]


def ask(prompt, max_tokens=110, temperature=0.7, retries=4):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API, data=body, headers={"Content-Type": "application/json"})
            text = []
            with urllib.request.urlopen(req, timeout=180) as r:
                event = None
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
        except Exception as e:  # noqa: BLE001 — log and continue the sweep
            if attempt < retries - 1:
                time.sleep(15)
                continue
            return f"<error: {e}>"
    return "<error: retries exhausted>"


def hits(text, keywords, prompt=""):
    """Keywords found in `text`, EXCLUDING any that already occur in the
    prompt.

    Without that exclusion the scoring silently self-confirms: the
    Los Alamos probe listed "alamo" as evidence of Texas-Alamo confusion,
    but "alamo" is a substring of "Los Alamos" in the question, so every
    on-topic answer scored a hit. A keyword the model was handed is not
    evidence of what the model knows.
    """
    low, plow = text.lower(), prompt.lower()
    return sorted({
        k for k in keywords
        if k not in plow and re.search(re.escape(k), low)
    })


out, results = [], []


def emit(s=""):
    print(s, flush=True)
    out.append(s)


def rescore():
    """Recompute theme counts from already-saved completions.

    Used after fixing the scoring rule, so a scoring bug costs no API
    calls and the raw completions stay the single source of truth.
    """
    path = ROOT / "results" / "atomic-probes.json"
    saved = json.loads(path.read_text())
    by_id = {pid: themes for pid, _hyp, _prompt, themes in PROBES}
    for rec in saved:
        themes = by_id.get(rec["id"], {})
        counts = {k: 0 for k in themes}
        for ans in rec["samples"]:
            for theme, kws in themes.items():
                if hits(ans, kws, rec["prompt"]):
                    counts[theme] += 1
        rec["theme_counts_rescored"] = counts
        print(f'{rec["id"]:<22} ' + "  ".join(
            f"{k} {v}/{rec['n']}" for k, v in counts.items()))
    path.write_text(json.dumps(saved, indent=1))
    print(f"\nrescored {len(saved)} probes in {path}")


if __name__ == "__main__" and "--rescore" in __import__("sys").argv:
    rescore()
    raise SystemExit(0)


emit("=== ATOMIC-AGE BEHAVIORAL PROBES (talkie chat API) ===")
emit(f"n={N} samples per prompt, temperature 0.7, {GAP}s between calls")
emit("Keyword hits are counted per sample; k/n is how many of the n")
emit("completions matched that theme at least once.")
emit()

for pid, hyp, prompt, themes in PROBES:
    emit(f"### [{hyp}] {pid} — {prompt!r}")
    samples, theme_counts = [], {k: 0 for k in themes}
    for i in range(N):
        ans = ask(prompt)
        samples.append(ans)
        sample_hits = {}
        for theme, kws in themes.items():
            h = hits(ans, kws, prompt)
            sample_hits[theme] = h
            if h:
                theme_counts[theme] += 1
        emit(f"  [{i+1}/{N}] {ans.strip()[:400]}")
        emit(f"        hits: " + ", ".join(
            f"{k}={v}" for k, v in sample_hits.items() if v) or "        hits: none")
        time.sleep(GAP)
    emit(f"  → {pid}: " + "  ".join(
        f"{k} {v}/{N}" for k, v in theme_counts.items()))
    emit()
    results.append({
        "id": pid, "hypothesis": hyp, "prompt": prompt,
        "n": N, "theme_counts": theme_counts, "samples": samples,
    })

(ROOT / "results" / "atomic-probes.txt").write_text("\n".join(out) + "\n")
(ROOT / "results" / "atomic-probes.json").write_text(json.dumps(results, indent=1))
emit(f"wrote results/atomic-probes.txt and results/atomic-probes.json")
