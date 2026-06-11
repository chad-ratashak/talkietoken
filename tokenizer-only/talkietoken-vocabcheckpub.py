from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained("talkie-lm/talkie-1930-13b-base")
vocab = tokenizer.get_vocab()
print(f"Loaded {len(vocab)} tokens from tokenizer")

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
