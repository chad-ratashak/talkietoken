# talkie-1930-13b-base — Tokenizer & Generation Audit Results

*June 12, 2026. Audit of the claim that `talkie-lm/talkie-1930-13b-base` was
trained only on pre-1931 English text: vocabulary analysis (no model
weights) plus generation probes via the hosted API.*

## Verdict

**Both the tokenizer and generation behavior are consistent with the
pre-1931 claim.** The tokenizer is consistent with the pre-1931 claim. A systematic sweep
of the 262,144-entry BPE vocab for post-1931 coinages found 1 hit out of 37
(`' email'`), and that hit is better explained by the pre-1931 sense of the
word (*email* = enamel, a French loanword common in decorative-arts text)
than by leakage. Every other initially suspicious signal dissolved under
scrutiny. The strongest evidence is comparative: against the o200k baseline
(borrowed from the tokenary project), talkie tokenizes vintage terms
*better* than a modern tokenizer and modern brand terms *worse* — exactly
the signature a genuine 1930 corpus predicts.

## Setup

The HF repo ships no HF tokenizer files — only a tiktoken-format `vocab.txt`
(base64 token + rank, 262,144 entries). The audit builds a
`tiktoken.Encoding` from it using the split pattern and special-token map
from the reference repo (`github.com/talkie-lm/talkie`,
`src/talkie/tokenizer.py`); see `tokenizer-only/talkie_tokenizer.py`.
Baseline: o200k (GPT-5) vocab extracted by tokenary, at
`baseline/o200k_tokens.csv`. Raw script outputs are in `results/`.

## Key findings

### 1. Post-1931 coinage sweep: 1/37 hits, and it's explainable

A word coined after 1930 cannot earn a BPE merge from a pre-1931 corpus.
Sweeping 37 coinages (radar 1940 → bitcoin 2008) across case/leading-space
variants found every one absent except `' email'` (rank 66,039). Against
leakage, for the enamel reading:

- `' Email'`, `'Email'`, `' emails'`, `' e-mail'` are all absent — a corpus
  with enough modern email text to merge `' email'` would have merged the
  plural and capitalized forms too (compare Roosevelt: 6 variants merged).
- Its rank neighborhood is Victorian: `' scepticism'`, `' vestments'`,
  `' bandit'`, `' Hilo'`.
- The enamel family is well represented: `' enamel'` (28,127),
  `' enamelled'` (65,582, British spelling, same tier), `' enamels'`
  (101,974). "Email ware" / "Limoges email" was routine in 1880–1930
  decorative-arts writing.

### 2. The Google token is Barney Google, not the search engine

`'Google'` and `' Google'` are single tokens — initially the scariest
signal (flagged by the anachronism probe). But *Barney Google* was one of
the most popular American comic strips from 1919, with a #1 hit song in
1923. Decisively: lowercase `'google'`/`'googly'` (the pre-1930 cricket
terms) did **not** merge — only the capitalized proper noun did, which
matches a comic-strip character, not a modern brand (o200k, by contrast,
merges both cases).

### 3. Comparative efficiency cuts in favor of the claim

Token counts, talkie vs o200k: vintage terms — `telegraph` 1 vs 2,
`aeroplane` 2 vs 4, `telephone` 1 vs 1, `phonograph` 2 vs 2. Modern terms —
`internet` 3 vs 1, `website` 2 vs 1, `Netflix` 3 vs 1, `Microsoft` 2 vs 1,
`computer` 2 vs 1. A tokenizer trained with modern text leakage would not
systematically lose on modern brands while beating o200k on 1920s
technology vocabulary.

### 4. Modern scripts and emoji fall back to bytes

Emoji encode at 4 tokens/character (raw UTF-8 bytes); CJK and Thai at
1.8–2.1 tokens/char; Cyrillic/Arabic/Hebrew ~0.9–1.0. French/German/Spanish
are efficient (0.25–0.43), as expected — European loanwords and quotations
are common in pre-1931 English print. Consistent with an English-only
vintage corpus.

### 5. All other 🚨 flags are false positives

The naive heuristics flag heavily because a 262K vocab merges almost any
8+-letter word into ≤2 tokens. Vocab-membership checks resolve them:
`neural` → neuralgia (19th-c. medicine); `transformer` → electrical
transformer (1880s); `nuclear` → cell biology / `polymorphonuclear`
(hematology); `prompt`, `token`, `embedding` → ordinary English words;
`Roosevelt` → Theodore; `Hitler` → prominent in newspapers well before 1931.
`computer`, `internet`, `WWII` have **no** vocab entries at all.

## Open anomaly: vocab size

The reference code hardcodes `BASE_VOCAB_SIZE = 65536`, but the shipped
`vocab.txt` has 262,144 entries (2¹⁸ vs 2¹⁶). Unexplained. Note that if the
model really only uses the first 65,536 ranks, then `' email'` (rank
66,039) isn't even in the model's effective vocabulary — the one coinage
hit would be moot. Worth raising with the talkie authors or checking
`final.ckpt`'s embedding-matrix shape (blocked on hardware, like the
generation tests).

Also minor: 262,144 raw entries collapse to 261,835 unique strings under
UTF-8 `errors="replace"` decoding — the difference is raw byte tokens that
are not valid UTF-8 (expected for a byte-level BPE).

## Generation probes (hosted API)

The HF Inference API does not serve the model, but talkie-lm.com's chat
demo uses a public SSE endpoint (`api.talkie-lm.com/api/chat/stream`, no
auth). `talkietoken-genleak-api.py` runs 12 leak probes against it; raw
transcript in `results/genleak-api.txt`. Caveat: the endpoint presumably
serves the instruction-tuned variant (`talkie-1930-13b-it`), not the base
model.

Every probe shows period-consistent ignorance of post-1930 concepts:

- **"Nuclear weapons are"** → cell-biology reading: "weapons composed of
  matter derived from the nucleus of a cell … characteristic of certain
  Protozoa."
- **"Computers in the future will"** → human computers: "will have to deal
  with a great variety of tables."
- **"Have you heard of the smartphone?"** → describes the telephone,
  citing the 1877 Paddington–Slough demonstration.
- **"World War II began when"** → completes with WWI: "Germans were
  arrested in Paris on a charge of espionage, August 10, 1914."
- **"As an AI language model,"** → "He speaks the language fluently" — no
  concept of AI; prompt-injection phrasing likewise gets only literal
  readings ("Promptitude in business is a quality of great value").
- **"Who is the current President of the United States?"** → "Theodroe
  [sic] Roosevelt."
- **"What do you think of Adolf Hitler?"** → a distinctly pre-1931 take:
  "a man of great energy and determination … sincere in believing that a
  return to monarchical Germany would be a good thing" — knows the 1920s
  political figure, nothing after.
- **"Franklin Roosevelt became president and"** → deflects to "Herbert
  Hoover became secretary of commerce" (Hoover held that office 1921–28;
  FDR's presidency, 1933, is unknown to it).
- **"The television will revolutionize"** → 1920s-style speculation about
  naval/battlefield observation — period-correct, since television was a
  real experimental topic by 1930.

## Limitations

A clean tokenizer alone cannot prove clean training data (a model could be
trained on modern text with a vintage-built tokenizer), which is why the
generation probes matter — and they point the same way. Remaining gaps:
the probes hit the hosted chat variant rather than the base checkpoint,
the probe set is small and non-adversarial (no systematic sweep of
post-1931 facts), and the vocab-size discrepancy below is unresolved.
