# talkietoken

talkietoken explores the historical plausibility of the training-data cutoff
claims of [talkie-1930](https://huggingface.co/talkie-lm/talkie-1930-13b-base)
by looking at the relative rank of tokens in the tokenizer, exploring tokens
that may indicate leakage (like the "Google" false positive), and testing
knowledge and likely next-token prediction for words whose semantic clusters
have shifted compared to contemporary usage.

The claim under test is that `talkie-lm/talkie-1930-13b-base` was trained only
on English text written before 1931. The audit leans on the tokenizer because a
BPE merge table is fitted to corpus frequencies before training starts, and
nothing afterwards (finetuning, RLHF, persona prompts, unlearning) can change
it. So the vocabulary is evidence about the training corpus that the model's
behaviour cannot fake cheaply. The repo also contains a reusable harness for a
second question: when a capability is absent from a model's training data, how
much in-context demonstration does it take to bring it back?

## What it found

- **The vocabulary is consistent with a pre-1931 corpus.** Of 37 post-1930
  coinages (radar to bitcoin), one is a single token: `' email'`. Its siblings
  (`' Email'`, `' emails'`, `' e-mail'`) are absent, while the enamel family
  (`' enamel'`, `' enamelled'`, `' enamels'`) is present, so the 1920s
  decorative-arts sense of *email* (enamel) explains it better than leakage.
- **The scariest hit is a comic strip.** `'Google'` is a token, but lowercase
  `'google'`/`'googly'` are not. That matches *Barney Google* (1919), not the
  search engine. A modern tokenizer such as o200k merges both cases.
- **Relative rank leans 1900s.** Across 15,006 English words shared with o200k,
  talkie merges period vocabulary far earlier (`chiefly`, `scarcely`,
  `luncheon`, `cavalry`, `locomotive`) and modern vocabulary far later
  (`password`, `delete`, `tweet`, `upgrade`, `sustainable`). It also encodes
  vintage terms more cheaply than o200k (`aeroplane` 2 tokens vs 4) and modern
  brands more expensively (`Netflix` 3 vs 1). (`results/rankshift.txt`)
- **Shifted words keep their old senses.** Asked for associations, talkie gives
  the period meaning of eight words whose sense has since moved: *computer*
  brings "reckoner" and "tabulator", *wireless* brings "Marconi" and
  "telegraphy", *broadcast* brings "sown" and "scattered". Overlap with a
  modern model's associations is near zero for all eight. This is one sample
  per word with no stable-word controls yet, so it is illustrative.
  (`results/neighborhoods-compare.md`)
- **Knowledge probes agree, but count for less.** Generation probes against
  the public chat demo give period-consistent answers ("computers" are human
  calculators; WWII prompts complete with 1914). After
  [Weird Generalization](https://arxiv.org/abs/2512.09742) (Betley et al.,
  2025), period-consistent behaviour is cheap to counterfeit with a small
  finetune, so these rank below the tokenizer evidence. The demo also runs a
  Qwen-based censor, so a deflected or missing answer may come from the
  censor rather than from talkie.
- **No Python merges, but Python is cheap anyway.** There is no `elif`,
  `__init__`, `numpy` or `.append`. Yet talkie needs only 1.1 to 1.3 times
  o200k's token count for Python, because Python is built from English
  morphemes a 1930 corpus already merged (`def`, `return`, `range`). A clean
  merge sweep is weak evidence of absence for any domain written in ordinary
  words.
- **The encoding-penalty predictor does not discriminate.** Under the o200k
  reference, talkie's normalised Python penalty is 1.243 against 1.156 for
  gemma-2-2b, a model that certainly saw Python. The margin is smaller than
  the spread among the Python-fluent controls themselves.
  (`results/penalty-matrix.txt`)
- **An unresolved vocabulary-size anomaly.** The shipped `vocab.txt` has
  262,144 entries; the reference code hardcodes 65,536. The file shows no seam
  at 65,536, and `<|endoftext|>` sits in the last slot, so the file looks like
  a genuine 262,144-merge artifact. Only the checkpoint's embedding shape can
  settle what the model actually uses.
- **The recovery-curve harness works; the public channel cannot settle the
  question.** A first live run on one-line Python functions rises from 0 of 8
  solved at zero demonstrations to 2 of 8 at four. Every interval includes
  zero, and the endpoint serves the instruction-tuned model, whose
  post-training used modern LLM help. It shows the instrument produces a
  curve, not that the base model reconstitutes Python.

`RESULTS.md` has the full audit. `TOKENIZER-FORENSICS.md` sets out what a merge
table can and cannot establish about claims of absence, with prior work and
eight verification protocols. `KNOWLEDGE-CHECKS.md` lists every question put
to talkie with the opening of its answer. The
[audit explorer](https://chad-ratashak.github.io/talkietoken/results/summary.html)
shows the neighborhoods, rank shifts and probe results on one page.

## Repository layout

| Path | Contents |
|---|---|
| `tokenizer-only/` | The tokenizer audit. `talkie_tokenizer.py` builds a `tiktoken.Encoding` from `vocab.txt`; `baselines.py` loads the comparison tokenizers; each `talkietoken-*.py` is one probe. These produced most of `results/`. |
| `reconstitution/` | The recovery-curve and encoding-penalty harness, model-agnostic. Start with `METHODS.md`. |
| `talkietoken-genleak-api.py`, `talkietoken-genprobe2-api.py`, `talkietoken-atomic-probes.py`, `talkietoken-langprobes.py` | Generation probes against the talkie-lm.com chat endpoint, rate-limited. |
| other top-level `talkietoken-*.py` | The first sketches, written against the talkie SDK and needing local weights. Superseded by the `tokenizer-only/` versions and kept for history. |
| `results/` | Raw outputs. `summary.html` and `great-filters-summary.html` are standalone reports, served on GitHub Pages as the [audit explorer](https://chad-ratashak.github.io/talkietoken/results/summary.html) and the [Great Filters estimates](https://chad-ratashak.github.io/talkietoken/results/great-filters-summary.html). |
| `baseline/` | The o200k token list used as the modern comparison. |
| `vocab.txt` | talkie's tokenizer vocabulary, as shipped. |

Research notes at the top level:

- `RESULTS.md` is the audit write-up.
- `KNOWLEDGE-CHECKS.md` lists the questions put to talkie and its answers.
- `TOKENIZER-FORENSICS.md` is the method note for claims of absence.
- `RESEARCH-weird-generalization-retro-naming.md` covers how Weird
  Generalization reweights the evidence, and how renamed places and offices
  break term-list probes.
- `talkietoken-great-filters.md` estimates how many languages ever produced
  enough text to train a model like this. `REVIEW-great-filters-semantic-drift.md`
  critiques it and the semantic-drift work.
- `TODO.md` lists open items.

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Dependencies are pinned exactly on purpose. Every committed result is a
tokenizer measurement, so the tokenizer libraries are part of the experiment.

Tokenizer probes run offline after the first run. tiktoken fetches the gpt2,
cl100k and o200k encodings from OpenAI's CDN once and caches them.

```bash
uv run python tokenizer-only/talkietoken-rankshift.py
uv run python tokenizer-only/talkietoken-python.py
uv run python tokenizer-only/talkietoken-baseline-compare.py
```

Most of these print to stdout. The `results/*.txt` files are their captured
output.

Validate the harness against known ground truth, with no network:

```bash
uv run python reconstitution/selftest.py
```

Compute the encoding penalty only, with no generation:

```bash
uv run python reconstitution/run_curve.py --penalty-only \
    --dialects py-canonical,py-nodoc,py-comment,py-indent8,py-indent16,py-tab,py-crlf
```

The cross-model matrix runs offline by default. The first run needs
`--allow-download` to fetch the Gemma-2 and Llama-3.2 tokenizers from Hugging
Face:

```bash
uv run python reconstitution/run_matrix.py --allow-download
```

Generation against the base model needs a GPU. An A10 is enough for 13B at
4-bit:

```bash
uv sync --extra local
uv run python reconstitution/run_curve.py \
    --backend local:talkie-lm/talkie-1930-13b-base --allow-exec
```

`--allow-exec` runs model-written code in a separate isolated interpreter with
CPU and time limits. `reconstitution/sandbox.py` says plainly that this raises
the cost of an accident, not of an attack.

The hosted-endpoint scripts send requests to a free public demo. They space
requests out and back off on HTTP 429. The harness also caps total requests
and caches every completion to disk. Keep it that way.

That demo serves the instruction-tuned model and runs a Qwen-based censor
alongside it. The censor is a modern model sitting between the probes and
talkie, and it is outside the project's model-origin allowlist. Both are
reasons to prefer the base-model path above.

## Data and provenance

| File | Source | Licence |
|---|---|---|
| `vocab.txt` | [`talkie-lm/talkie-1930-13b-base`](https://huggingface.co/talkie-lm/talkie-1930-13b-base), byte-identical to the file there (git blob `c3e00302`) | Apache-2.0 |
| split pattern in `tokenizer-only/talkie_tokenizer.py` | [`talkie-lm/talkie`](https://github.com/talkie-lm/talkie), `src/talkie/tokenizer.py` | Apache-2.0 |
| `baseline/o200k_tokens.{json,csv}` | The `o200k_base` encoding from [tiktoken](https://github.com/openai/tiktoken), decoded to a token list with [tokenary](https://github.com/chad-ratashak/tokenary) | MIT (tiktoken) |
| `reconstitution/reference/talkie-figure3.json` | Data arrays behind Figure 3 of the [talkie report](https://talkie-lm.com/), recorded with provenance | Facts from the report |
| generation transcripts in `results/` | Outputs of the talkie-lm.com chat demo | — |

The gpt2, cl100k and o200k encodings, and the Gemma-2 and Llama-3.2
tokenizers, are downloaded at run time and are not redistributed here. Their
own terms apply to that download.

Baseline tokenizers are limited to U.S. and allied vendors under the project's
model-origin rule (`tokenizer-only/baselines.py`). Every baseline loads as
data only; `trust_remote_code` is hard-disabled.

## Limitations

- A clean tokenizer cannot prove clean training data. The method assumes the
  tokenizer was fitted on the model's own training corpus, and for talkie that
  is unverified. The report also does not say whether the tokenizer was fitted
  before or after its anachronism filter.
- Merges track frequency, not presence. The strongest conclusion from a clean
  sweep is "vanishingly rare", and the detection floor has not been calibrated.
- The behavioural and association probes hit the instruction-tuned chat
  model, through a demo that also runs a Qwen-based censor, with small
  samples per prompt.

`TODO.md` tracks what would close each gap.

## Licence

Apache-2.0. See `LICENSE`. Third-party data keeps its own licence, listed
above.
