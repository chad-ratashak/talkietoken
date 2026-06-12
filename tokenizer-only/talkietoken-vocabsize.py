import talkie_tokenizer as AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("talkie-lm/talkie-1930-13b-base")

vocab_size = len(tokenizer.get_vocab())
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
