from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("talkie-lm/talkie-1930-13b-base")

# Terms that shouldn't exist in 1930 training data
anachronistic_terms = [
    # Technology (post-1930)
    "computer", "television", "radar", "jet engine", "helicopter", "atomic bomb", "transistor", "semiconductor",

    # LLM/AI specific
    "neural network", "attention mechanism", "system prompt", "AI assistant", "tokenizer", "fine-tuning", "RLHF",

    # Modern concepts
    "internet", "website", "email", "smartphone", "GPS", "DNA", "chemotherapy",

    # Post-1930 people/events
    "New Deal", "World War II", "United Nations", "NATO", "Cold War", "Manhattan Project",

    # Modern brands/companies
    "Microsoft", "Google", "Netflix",
]

for term in anachronistic_terms:
    tokens = tokenizer.encode(term)
    print(f"'{term}': {len(tokens)} tokens - {tokens}")

    if len(tokens) == 1:
        print(f"  ⚠️  SINGLE TOKEN - possible leakage!")
    elif len(tokens) > 5:
        print(f"  ✓ Highly fragmented - likely unseen")
