"""Round-2 generation probes: geography, polities, scripts.

Themes chosen because the relevant name/border/alphabet changed right
around the 1931 cutoff. A genuine pre-1931 model should give the OLD
answer and be ignorant of post-1930 polities (Manchukuo, modern Somalia,
the State of Israel). Spaced requests to respect the API rate limit.
"""

import json
import time
import urllib.request

API = "https://api.talkie-lm.com/api/chat/stream"

PROBES = [
    "What is the capital of East Prussia?",
    "Tell me about the country called Manchukuo.",
    "What alphabet is used to write the Turkish language?",
    "Is the great city on the Bosphorus called Constantinople or Istanbul?",
    "Describe the country of Somalia for me.",
    "What is the capital city of Norway?",
    "Tell me about the nation of Yugoslavia.",
    "What do you know of Tannu Tuva?",
    "Please write a short greeting in the Chinese language.",
    "Name the great powers of Europe and their forms of government.",
    "What is the State of Israel?",
    "Who rules Germany, and what is its government called?",
]


def ask(prompt, max_tokens=90, temperature=0.1, retries=3):
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
            with urllib.request.urlopen(req, timeout=120) as r:
                event = None
                for raw in r:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:") and event == "token":
                        text.append(json.loads(line[5:].strip()))
            return "".join(text)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(30)
                continue
            return f"<error: {e}>"
        except Exception as e:
            return f"<error: {e}>"


for prompt in PROBES:
    out = ask(prompt)
    print(f"PROMPT: {prompt}")
    print(f"TALKIE: {out}")
    print("---", flush=True)
    time.sleep(10)
