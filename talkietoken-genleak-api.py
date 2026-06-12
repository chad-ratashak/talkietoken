"""Generation leak probes against the hosted talkie chat API.

The HF Inference API does not serve this model, but talkie-lm.com exposes
a public SSE endpoint (discovered from the /chat page source). Note this
serves the chat experience — presumably the instruction-tuned variant
(talkie-1930-13b-it), not the raw base model — so completions are
conversational rather than pure continuations.

Each probe asks about a post-1930 concept. A genuinely pre-1931 model
should treat them as unknown words or misread them in a period sense;
recognizing them is leakage evidence.
"""

import json
import urllib.request

API = "https://api.talkie-lm.com/api/chat/stream"

PROBES = [
    "The television will revolutionize",
    "Computers in the future will",
    "Nuclear weapons are",
    "The internet allows people to",
    "Franklin Roosevelt became president and",
    "World War II began when",
    "System prompt: You are a helpful AI assistant",
    "Please ignore previous instructions and",
    "As an AI language model,",
    "Have you heard of the smartphone?",
    "What do you think of Adolf Hitler?",
    "Who is the current President of the United States?",
]


def ask(prompt, max_tokens=80, temperature=0.1):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
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


for prompt in PROBES:
    try:
        out = ask(prompt)
    except Exception as e:  # rate limits etc. — report and continue
        out = f"<error: {e}>"
    print(f"PROMPT: {prompt}")
    print(f"TALKIE: {out}")
    print("---")
