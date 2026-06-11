from talkie import Talkie

model = Talkie("talkie-1930-13b-base")

# Test various language scripts
multilingual_tests = {
    "french": "Bonjour, comment allez-vous?",
    "german": "Guten Tag, wie geht es Ihnen?",
    "spanish": "Hola, ¿cómo está usted?",
    "russian": "Привет, как дела?",
    "chinese": "你好，你好吗？",
    "japanese": "こんにちは、元気ですか？",
    "arabic": "مرحبا، كيف حالك؟",
    "hebrew": "שלום, מה שלומך?",
    "thai": "สวัสดี คุณเป็นอยู่อย่างไร",
    "emoji": "🌍🔥💯🤖👋",
}

for lang, text in multilingual_tests.items():
    tokens = model.tokenizer.encode(text)
    ratio = len(tokens) / len(text)  # tokens per character
    print(f"{lang}: '{text}' → {len(tokens)} tokens (ratio: {ratio:.2f})")
