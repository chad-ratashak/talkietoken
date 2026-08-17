"""Local tokenizer for talkie-1930-13b-base.

The HF repo ships no tokenizer_config.json / tokenizer.json — only a
tiktoken-format vocab.txt (base64 token + rank). This builds a
tiktoken.Encoding from it, mirroring the reference implementation at
github.com/talkie-lm/talkie (src/talkie/tokenizer.py).

NOTE: the reference code hardcodes BASE_VOCAB_SIZE = 65536, but the
shipped vocab.txt for the 13b model has 262,144 entries. We build at the
shipped size; the discrepancy is itself an audit finding (see RESULTS.md).

Exposes a thin HF-AutoTokenizer-compatible wrapper so the existing audit
scripts (.encode/.decode/.get_vocab) run unchanged.
"""

from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.txt"

# Split pattern copied verbatim from the reference repo.
_PAT_STR = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


class TalkieTokenizer:
    """Minimal HF-compatible facade over a tiktoken Encoding."""

    def __init__(self, vocab_path=VOCAB_PATH):
        mergeable_ranks = load_tiktoken_bpe(str(vocab_path))
        vocab_size = len(mergeable_ranks)
        eot_rank = vocab_size - 1
        mergeable_ranks = {k: v for k, v in mergeable_ranks.items() if v < eot_rank}
        self.enc = tiktoken.Encoding(
            name="talkie-base-local",
            pat_str=_PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens={"<|endoftext|>": eot_rank},
        )
        self.vocab_size = vocab_size
        self._mergeable_ranks = mergeable_ranks

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids)

    def get_vocab(self):
        out = {
            b.decode("utf-8", errors="replace"): rank
            for b, rank in self._mergeable_ranks.items()
        }
        out["<|endoftext|>"] = self.vocab_size - 1
        return out


def from_pretrained(*_args, **kwargs):
    # Never execute hub-shipped code. This loader is pure data (vocab.txt →
    # tiktoken ranks); rejecting the flag here keeps a copy-pasted
    # AutoTokenizer.from_pretrained(..., trust_remote_code=True) call from
    # silently implying that remote code was ever needed or run.
    if kwargs.pop("trust_remote_code", False):
        raise ValueError(
            "trust_remote_code is disabled in this project. The talkie "
            "tokenizer is built locally from vocab.txt (data only); no "
            "remote code is fetched or executed. Drop trust_remote_code."
        )
    return TalkieTokenizer()
