"""The encoding penalty: how much context a capability costs to demonstrate.

Generalises the Tier 4 result from `tokenizer-only/talkietoken-python.py`
(talkie spends 1.31 tokens per o200k token on HumanEval-style Python, 1.09
on a class with dunders, 1.00 on period prose) into a predictor variable
that any (model, capability) pair can be scored on before a single
generation is spent.

The hypothesis it exists to test
--------------------------------
Reconstitution happens inside a finite context window. If a model's
tokenizer has no merges for the capability's surface form, every
demonstration costs more tokens, so a fixed context holds fewer
demonstrations. Prediction: higher penalty -> less recovery per token of
context, and a recovery curve that rises more slowly against a token
budget.

Two things the naive ratio gets wrong, both fixed here
------------------------------------------------------
1. **Vocabulary size confounds the raw ratio.** talkie has 262,144 merges;
   a 32k-vocab model would spend more tokens on *everything*, prose
   included, and would look "penalised" on code for reasons that have
   nothing to do with code. So the reported statistic is NORMALISED: the
   material ratio divided by the same model's ratio on control prose that
   is squarely in its training distribution. A normalised penalty of 1.0
   means "this material costs exactly what ordinary text costs me"; 1.31
   means "it costs 31% more than my own baseline".

2. **The reference tokenizer can be as blind as the target.** Tier 5
   already found this: talkie vs o200k on J and APL is nearly flat,
   because o200k has no merges for APL either. The ratio hides the effect;
   tokens-per-character does not. Both are reported, and `tok_per_char` is
   the one to trust when the reference is also out of distribution.

Model-agnostic by construction: nothing here knows what talkie is. Point
`load_tokenizer` at any tiktoken encoding or any HF repo whose tokenizer
loads from data files alone (remote code stays blocked, per the existing
`tokenizer-only/baselines.py` policy).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `tokenizer-only` has a hyphen, so it can never be an importable package
# name. Put it on the path and import its modules directly rather than
# duplicating the vocab loader or the baseline safety policy here.
_TOK_DIR = ROOT / "tokenizer-only"
if str(_TOK_DIR) not in sys.path:
    sys.path.insert(0, str(_TOK_DIR))


@dataclass(frozen=True)
class Tokenizer:
    """A named thing that turns text into a token count."""

    spec: str
    label: str
    vocab_size: int
    _encode: object

    def encode(self, text):
        return self._encode(text)

    def count(self, text):
        return len(self._encode(text))


def load_tokenizer(spec):
    """Load a tokenizer from a short spec string.

        talkie                  the local vocab.txt tiktoken build
        tiktoken:o200k_base     any tiktoken encoding by name
        hf:org/repo             HF tokenizer, trust_remote_code hard-off

    Deliberately not a registry of models — a spec string keeps the harness
    pointable at the next filtered or ablated model without edits here.
    """
    if spec == "talkie":
        import talkie_tokenizer

        t = talkie_tokenizer.from_pretrained()
        return Tokenizer(spec, "talkie-1930 (local vocab.txt)", t.vocab_size, t.encode)

    if spec.startswith("tiktoken:"):
        import tiktoken

        enc = tiktoken.get_encoding(spec.split(":", 1)[1])
        return Tokenizer(spec, f"{enc.name} (tiktoken)", enc.n_vocab, enc.encode)

    if spec.startswith("hf:"):
        import baselines

        tok = baselines.hf_tokenizer(spec.split(":", 1)[1])
        return Tokenizer(
            spec,
            f"{spec[3:]} (transformers)",
            len(tok),
            lambda s: tok.encode(s, add_special_tokens=False),
        )

    raise ValueError(
        f"unknown tokenizer spec {spec!r}; expected 'talkie', "
        "'tiktoken:<name>', or 'hf:<org/repo>'"
    )


# --- the measurements ---------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """One text, measured under both tokenizers."""

    label: str
    chars: int
    target_tokens: int
    ref_tokens: int

    @property
    def ratio(self):
        """Target tokens per reference token — the raw Tier 4 number."""
        return self.target_tokens / self.ref_tokens if self.ref_tokens else float("nan")

    @property
    def tok_per_char(self):
        """Target tokens per character. Survives a blind reference."""
        return self.target_tokens / self.chars if self.chars else float("nan")


def measure(target, ref, text, label=""):
    return Measurement(label, len(text), target.count(text), ref.count(text))


@dataclass(frozen=True)
class Penalty:
    """A capability's encoding penalty for one model, versus its own baseline."""

    material: Measurement
    control: Measurement

    @property
    def normalized(self):
        """material.ratio / control.ratio — the headline predictor.

        1.0 = this capability costs the model exactly what its own ordinary
        text costs. >1 = the model must spell the capability out.
        """
        c = self.control.ratio
        return self.material.ratio / c if c else float("nan")

    @property
    def tok_per_char_excess(self):
        """Ratio of tokens-per-character, material vs control. The
        reference-free version of the same question, for when the reference
        tokenizer is also out of distribution on this material."""
        c = self.control.tok_per_char
        return self.material.tok_per_char / c if c else float("nan")


def penalty_of(target, ref, material, control, label=""):
    """Normalised encoding penalty of `material` against in-distribution
    `control` prose. Both texts should be the same rough length; the ratios
    are length-invariant but the token counts are noisier when short."""
    return Penalty(
        measure(target, ref, material, label or "material"),
        measure(target, ref, control, "control"),
    )


# --- the control prose --------------------------------------------------

# Period English, matching the register of the existing Tier 4 control so
# the two experiments' numbers stay commensurable. A control must be text
# the model is unambiguously good at, or the normalisation divides by an
# unknown.
PERIOD_PROSE_CONTROL = (
    "The wireless operator, having received the message, carried it "
    "directly to the captain upon the bridge, who read it twice before "
    "making any reply. The morning was thick with fog, and the vessel had "
    "slackened her speed considerably since the small hours, sounding her "
    "whistle at intervals as the regulations require.\n"
)

# For models whose in-distribution register is modern rather than vintage.
# Whichever control is used must be named in the results file: the
# normalised penalty is meaningless without it.
MODERN_PROSE_CONTROL = (
    "The support team received the message and forwarded it to the duty "
    "manager, who read it twice before replying. Visibility was poor that "
    "morning and the service had been running at reduced capacity since "
    "well before dawn, with status updates posted at regular intervals as "
    "the policy requires.\n"
)

CONTROLS = {
    "period-prose": PERIOD_PROSE_CONTROL,
    "modern-prose": MODERN_PROSE_CONTROL,
}
