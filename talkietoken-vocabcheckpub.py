from talkie import Talkie
import requests
import re

model = Talkie("talkie-1930-13b-base")

vocab_url = "https://huggingface.co/talkie-lm/talkie-1930-13b-base/resolve/main/vocab.txt"

# Try the public vocab file; fall back to the loaded tokenizer if it 404s
# (SentencePiece/BPE tokenizers often don't ship a plain vocab.txt)
try:
    r = requests.get(vocab_url, timeout=10)
    r.raise_for_status()
    vocab = {line.strip(): i for i, line in enumerate(r.text.splitlines()) if line.strip()}
    print(f"Loaded {len(vocab)} tokens from {vocab_url}")
except Exception as e:
    print(f"Vocab URL failed ({e}); using tokenizer.get_vocab() instead")
    vocab = model.tokenizer.get_vocab()

# Look for suspicious entries
suspicious_patterns = [
    r'computer', r'internet', r'neural', r'transformer',
    r'AI', r'prompt', r'token', r'embedding',
    r'Roosevelt', r'Hitler', r'WWII', r'nuclear',
]

for pattern in suspicious_patterns:
    matches = [tok for tok in vocab if re.search(pattern, tok, re.IGNORECASE)]
    if matches:
        preview = matches[:10]
        more = '...' if len(matches) > 10 else ''
        print(f"🚨 '{pattern}' matched {len(matches)} tokens: {preview}{more}")
    else:
        print(f"  ✓ '{pattern}' — no matches")
