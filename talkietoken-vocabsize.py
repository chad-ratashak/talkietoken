from talkie import Talkie

model = Talkie("talkie-1930-13b-base")

# Try to determine vocab size
vocab_size = len(model.tokenizer.vocab)  # if accessible
print(f"Vocabulary size: {vocab_size}")

# Modern tokenizers typically:
# - GPT-2: ~50K tokens
# - BERT: ~30K tokens
# - T5: ~32K tokens
# - LLaMA: ~32K tokens

# A truly 1930s vocab might be smaller due to:
# - No modern technical terms
# - No internet slang/abbreviations
# - No modern proper nouns
