"""Safe loaders for open-source baseline tokenizers.

Safety policy, enforced here so audit scripts never have to think about it:

1. **No remote code execution.** Every baseline below is a pure-data
   artifact — a BPE merge-rank file fetched from OpenAI's public CDN by
   tiktoken and cached locally. tiktoken has no trust_remote_code pathway
   at all; nothing downloaded is ever executed.
2. **Future Hugging Face baselines must go through hf_tokenizer()**, which
   hard-pins trust_remote_code=False. A repo whose tokenizer requires
   custom hub code is rejected, not accommodated.
3. **Model-origin policy:** baselines are limited to U.S./allied-origin
   vendors (these are all OpenAI, U.S., running locally).
   Other tokenizers (e.g. Qwen, DeepSeek, GLM, Yi) are excluded under the
   project's model-origin rule (vendors outside the US/allied allowlist).

The set spans three tokenizer eras so comparisons aren't anchored to a
single modern vocabulary:

    gpt2         2019   50,257 tokens   WebText-era English web
    cl100k_base  2022  100,277 tokens   GPT-3.5/4-era web + code
    o200k_base   2024  199,998 tokens   GPT-4o-era multilingual web

Note: (r50k/p50k are near-duplicates of gpt2 and add little signal.)
A LLaMA/sentencepiece baseline would also qualify origin-wise (Meta, U.S.)
but is gated on HF and needs sentencepiece; add via hf_tokenizer() if ever
needed.
"""

import tiktoken

# name -> (tiktoken encoding name, era year) — ordered oldest to newest
BASELINES = {
    "gpt2": ("gpt2", 2019),
    "cl100k": ("cl100k_base", 2022),
    "o200k": ("o200k_base", 2024),
}


def load_baselines():
    """Return {name: tiktoken.Encoding} for all baselines, oldest first."""
    return {name: tiktoken.get_encoding(enc) for name, (enc, _) in BASELINES.items()}


def vocab_strings(enc):
    """Token strings of a tiktoken Encoding, decoded the same lossy way
    TalkieTokenizer.get_vocab() decodes (errors='replace'), so string-set
    overlaps are computed like-for-like."""
    return {
        b.decode("utf-8", errors="replace")
        for b in enc._mergeable_ranks  # noqa: SLF001 — no public accessor
    }


def hf_tokenizer(repo_id, **kwargs):
    """Load a Hugging Face tokenizer with remote code execution blocked.

    The only sanctioned way to add an HF-hosted baseline. Overriding
    trust_remote_code is an error, not an option.
    """
    if kwargs.pop("trust_remote_code", False):
        raise ValueError(
            "trust_remote_code is disabled in this project; pick a "
            "baseline whose tokenizer loads from data files alone."
        )
    from transformers import AutoTokenizer  # lazy: not a hard dependency

    return AutoTokenizer.from_pretrained(repo_id, trust_remote_code=False, **kwargs)
