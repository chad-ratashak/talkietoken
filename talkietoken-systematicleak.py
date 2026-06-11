from talkie import Talkie
import re

model = Talkie("talkie-1930-13b-base")

anachronistic_terms = [
    "computer", "television", "radar", "jet engine", "helicopter", "atomic bomb", "transistor", "semiconductor",
    "neural network", "attention mechanism", "system prompt", "AI assistant", "tokenizer", "fine-tuning", "RLHF",
    "internet", "website", "email", "smartphone", "GPS", "DNA", "chemotherapy",
    "New Deal", "World War II", "United Nations", "NATO", "Cold War", "Manhattan Project",
    "Microsoft", "Google", "Netflix",
]

def test_for_leakage(model, test_terms, context=""):
    """Test if terms are suspiciously well-tokenized"""
    suspicious = []

    for term in test_terms:
        full_text = context + term if context else term
        tokens = model.tokenizer.encode(full_text)

        # Heuristics for suspicion:
        # 1. Very few tokens for complex modern terms
        # 2. Single token for post-1930 technical terms
        # 3. Better tokenization than expected for modern concepts

        if len(tokens) <= 2 and len(term) > 8:
            suspicious.append((term, tokens, "Too few tokens for length"))
        elif any(modern in term.lower() for modern in ['computer', 'internet', 'neural', 'ai']):
            if len(tokens) <= 3:
                suspicious.append((term, tokens, "Modern tech term too well tokenized"))

    return suspicious

# Test various categories
leakage_results = test_for_leakage(model, anachronistic_terms)
for term, tokens, reason in leakage_results:
    print(f"🚨 {term}: {tokens} - {reason}")
