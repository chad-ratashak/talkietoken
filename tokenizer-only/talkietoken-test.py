from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("talkie-lm/talkie-1930-13b-base")

# Test tokenization patterns to identify the algorithm
test_cases = [
    "transformer",  # Should split if BPE/WordPiece
    "unhappiness",  # Test subword splitting
    "hello",        # Common word
    "pneumonoultramicroscopicsilicovolcanoconiosis",  # Very long word
    "🤖",           # Modern emoji
    "café",         # Accented characters
    "привет",       # Cyrillic
    "こんにちは",     # Japanese
    "🔥💯",         # Multiple emojis
]

for text in test_cases:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"'{text}' → {tokens} → '{decoded}'")
