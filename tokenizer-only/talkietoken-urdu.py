"""Is the Urdu جین/جن collapse a general coverage asymmetry, or an anecdote?

Motivated by arXiv:2502.00041 (MALT, Bajwa, Jan 2025), which shows an Urdu
prompt — جین میں ترمیم کیسے کام کرتی ہے؟ , "How does gene editing work?" —
producing an internal English latent about jinn in the Quran. A biology
question reasoned about internally as religion. Models in that paper are
Gemma-2-2b and Llama-3.2-3b.

Why the tokenizer is the right instrument (and where it stops)
--------------------------------------------------------------
A BPE/unigram vocabulary is fossilised corpus statistics. If a script's
technical register was thin in the tokenizer's training mix, its words do
not earn merges and reach the model as fragments; words from a well-covered
register arrive whole. That is MEASURABLE, and it is all this file measures.

It does NOT measure why a latent collapses onto the religious reading.
Fragmentation is a property of the input encoding; the collapse is a
property of the residual stream. Every mechanism sentence in the report is
labelled HYPOTHESIS and is not tested here.

The confounds that eat the naive version of this test
-----------------------------------------------------
"Religious words tokenize better than biology words in Urdu" has at least
five explanations that have nothing to do with domain:

  1. LENGTH.    Biology terms are longer. Longer words fragment more.
                Controlled by tokens/CHAR and a length-stratified split,
                not tokens/word alone.
  2. ARABIC.    Most Urdu religious vocabulary is also Arabic vocabulary,
                and these tokenizers saw far more Arabic than Urdu. An
                "Arabic-shared" win is not an Urdu-domain win.
                Controlled by splitting both sets on Arabic-shared vs
                Perso-Indic-only.
  3. DOMAIN.    Maybe biology is just intrinsically harder to tokenize.
                Controlled by an English mirror — the same 30 concepts in
                English, through the same tokenizers.
  4. BASELINE.  Without a neutral register you cannot tell whether biology
                is depressed or religion elevated.
                Controlled by a 15-word everyday-Urdu set.
  5. ONE WORD.  A gap carried entirely by جین or جن is the anecdote again.
                Controlled by leave-one-out.

Word lists were fixed before any token counts were run, and every per-word
count is printed so the selection can be audited. Selection was made by the
author, who is not a fluent Urdu speaker; see LIMITATIONS.

A null result here is a real result. The report says so plainly where it
happens, and reports the CI on the difference so "no effect" can be told
apart from "too few words to see one".

Tiers, by how much something-other-than-domain could explain the signal:

    TIER 0   the anecdote itself, on the models that produced it
    TIER 1   Urdu-only letters: is there a coverage floor at all?
    TIER 2   the sibling control: 15 biology vs 15 religion  <- the test
    TIER 3   the five confounds, each with its own control
    TIER 4   orthography: U+06CC vs U+064A across a word list
    TIER 5   byte fallback on running Urdu text
    TIER 6   worked splits of the MALT prompt

Writes results/urdu-domain-asymmetry.txt and .json.

RTL NOTE: every table puts Urdu at the END of its line. Bidi reordering
scrambles numeric columns that sit to the right of Arabic script, so the
Latin columns come first everywhere.
"""

import hashlib
import json
from pathlib import Path

import baselines
import numpy as np
from talkie_tokenizer import TalkieTokenizer

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260817)
N_PERM = 20000
N_BOOT = 10000

# --------------------------------------------------------------------------
# Tokenizers.
#
# google/gemma-2-2b and meta-llama/Llama-3.2-3B are gated on HF and this
# account has no access (403). Loaded instead from ungated mirrors, verified
# by hashing the full core vocabulary: three unrelated uploaders per family
# agree byte-for-byte (see PROVENANCE in the report).
# --------------------------------------------------------------------------
GEMMA_REPO = "unsloth/gemma-2-2b"
LLAMA_REPO = "unsloth/Llama-3.2-3B"
GEMMA_MIRRORS = [GEMMA_REPO, "unsloth/gemma-2-2b-it",
                 "IlyaGusev/gemma-2-2b-it-abliterated"]
LLAMA_MIRRORS = [LLAMA_REPO, "NousResearch/Llama-3.2-1B",
                 "chuanli11/Llama-3.2-3B-Instruct-uncensored"]

_L_BYTES = {}


def _byte_decoder():
    """Inverse of the GPT-2 byte-to-unicode map used by Llama's BPE."""
    if _L_BYTES:
        return _L_BYTES
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    _L_BYTES.update({chr(c): b for b, c in zip(bs, cs)})
    return _L_BYTES


class Tok:
    """Uniform facade over four tokenizers.

    pieces() returns display strings. A token whose bytes are not valid
    UTF-8 renders as <hex>, the same convention as results/python-leak.txt,
    so byte fallback stays visible instead of collapsing to U+FFFD.
    """

    def __init__(self, name, kind, obj, note):
        self.name, self.kind, self.obj, self.note = name, kind, obj, note

    def ids(self, s):
        if self.kind == "tiktoken":
            return self.obj.encode(s)
        return self.obj.encode(s, add_special_tokens=False)

    def count(self, s):
        return len(self.ids(s))

    def _tok_bytes(self, i):
        if self.kind == "tiktoken":
            return self.obj.decode_single_token_bytes(i)
        t = self.obj.convert_ids_to_tokens([i])[0]
        if self.kind == "sp":
            if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                return bytes([int(t[3:5], 16)])
            return t.replace("▁", " ").encode("utf-8")
        d = _byte_decoder()
        return bytes(d[c] for c in t)

    def pieces(self, s):
        out = []
        for i in self.ids(s):
            b = self._tok_bytes(i)
            try:
                out.append(b.decode("utf-8"))
            except UnicodeDecodeError:
                out.append("".join(f"<{x:02x}>" for x in b))
        return out

    def n_bytefall(self, s):
        """How many tokens are raw bytes with no standalone UTF-8 reading."""
        n = 0
        for i in self.ids(s):
            try:
                self._tok_bytes(i).decode("utf-8")
            except UnicodeDecodeError:
                n += 1
        return n

    def whole(self, s):
        return self.count(s) == 1


def load_tokenizers():
    o200k = baselines.load_baselines()["o200k"]
    talkie = TalkieTokenizer()
    return [
        Tok("gemma-2-2b", "sp", baselines.hf_tokenizer(GEMMA_REPO),
            "MALT model; SentencePiece unigram, 256k"),
        Tok("Llama-3.2-3B", "bbpe", baselines.hf_tokenizer(LLAMA_REPO),
            "MALT model; byte-level BPE, 128k"),
        Tok("o200k", "tiktoken", o200k,
            "project baseline; byte-level BPE, 200k"),
        Tok("talkie-1930", "tiktoken", talkie.enc,
            "floor control; English-only 1930 corpus"),
    ]


# --------------------------------------------------------------------------
# TIER 0 — the anecdote.
# --------------------------------------------------------------------------
JEEM_NOON = "جن"
JEEM_NOON_SHADDA = "جنّ"
GENE_ARABIC_YE = "جين"
GENE_URDU_YE = "جین"

ANECDOTE = [
    (JEEM_NOON, "jinn", "jeem-noon"),
    (JEEM_NOON_SHADDA, "jinn", "+ shadda U+0651"),
    (GENE_ARABIC_YE, "gene", "Arabic ye U+064A"),
    (GENE_URDU_YE, "gene", "Urdu ye U+06CC <- MALT"),
]
ENGLISH_PIVOT = ["gene", "genome", "genie", "jinn", "djinn"]

# --------------------------------------------------------------------------
# TIER 1 — Urdu-only letters vs letters shared with Arabic.
# --------------------------------------------------------------------------
URDU_ONLY_LETTERS = [
    ("ٹ", "U+0679", "TTEH retroflex t"), ("ڈ", "U+0688", "DDAL retroflex d"),
    ("ڑ", "U+0691", "RREH retroflex r"), ("ں", "U+06BA", "NOON GHUNNA"),
    ("ے", "U+06D2", "BARREE YEH final e"), ("ہ", "U+06C1", "HEH GOAL"),
    ("ی", "U+06CC", "FARSI YEH (Urdu ye)"), ("ک", "U+06A9", "KEHEH (Urdu k)"),
    ("گ", "U+06AF", "GAF g"), ("پ", "U+067E", "PEH p"),
    ("چ", "U+0686", "TCHEH ch"), ("ژ", "U+0698", "JEH zh"),
]
ARABIC_LETTERS = [
    ("ي", "U+064A", "ARABIC YEH"), ("ك", "U+0643", "ARABIC KAF"),
    ("ه", "U+0647", "ARABIC HEH"), ("ت", "U+062A", "TEH"),
    ("د", "U+062F", "DAL"), ("ر", "U+0631", "REH"),
    ("ن", "U+0646", "NOON"), ("ب", "U+0628", "BEH"),
    ("س", "U+0633", "SEEN"), ("م", "U+0645", "MEEM"),
    ("ل", "U+0644", "LAM"), ("ق", "U+0642", "QAF"),
]

# --------------------------------------------------------------------------
# TIER 2 — the sibling control. CORE LISTS FIXED BEFORE ANY COUNTS WERE RUN.
#
#   (urdu, roman, english gloss, origin)
#   origin: ar = also standard Arabic vocabulary, same spelling and sense
#           pi = Perso-Indic or English loan; NOT standard Arabic
# The origin tag is what TIER 3(A) uses to strip the Arabic confound.
# --------------------------------------------------------------------------
BIO = [
    ("جین", "jeen", "gene", "pi"),
    ("خلیہ", "khaliya", "cell", "ar"),
    ("وراثت", "wirasat", "heredity", "ar"),
    ("جراثیم", "jarasim", "germs", "ar"),
    ("حیاتیات", "hayatiyat", "biology", "ar"),
    ("پروٹین", "proteen", "protein", "pi"),
    ("مدافعت", "mudafiat", "immunity", "ar"),
    ("اعصاب", "asab", "nerves", "ar"),
    ("نسیج", "naseej", "tissue", "ar"),
    ("تولید", "taulid", "reproduction", "ar"),
    ("مریض", "mareez", "patient", "ar"),
    ("علاج", "ilaj", "treatment", "ar"),
    ("دماغ", "dimagh", "brain", "ar"),
    ("پھیپھڑا", "phephra", "lung", "pi"),
    ("بیماری", "bimari", "disease", "pi"),
]
REL = [
    ("جن", "jinn", "jinn", "ar"),
    ("فرشتہ", "farishta", "angel", "pi"),
    ("شیطان", "shaitan", "devil", "ar"),
    ("عبادت", "ibadat", "worship", "ar"),
    ("قیامت", "qayamat", "doomsday", "ar"),
    ("تعویذ", "taweez", "amulet", "ar"),
    ("زیارت", "ziyarat", "pilgrimage", "ar"),
    ("پیغمبر", "paighambar", "prophet", "pi"),
    ("دوزخ", "dozakh", "hell", "pi"),
    ("ایمان", "iman", "faith", "ar"),
    ("روحانیت", "ruhaniyat", "spirituality", "ar"),
    ("شیاطین", "shayateen", "devils", "ar"),
    ("روزہ", "roza", "fast", "pi"),
    ("بھوت", "bhoot", "ghost", "pi"),
    ("جادوگر", "jadugar", "magician", "pi"),
]
NEUTRAL = [
    ("کتاب", "kitab", "book", "ar"),
    ("پانی", "pani", "water", "pi"),
    ("مکان", "makan", "house", "ar"),
    ("درخت", "darakht", "tree", "pi"),
    ("کھڑکی", "khirki", "window", "pi"),
    ("بازار", "bazar", "market", "pi"),
    ("موسم", "mausam", "weather", "ar"),
    ("کھانا", "khana", "food", "pi"),
    ("عمارت", "imarat", "building", "ar"),
    ("پہاڑ", "pahar", "mountain", "pi"),
    ("دریا", "darya", "river", "pi"),
    ("تصویر", "tasveer", "picture", "ar"),
    ("حکومت", "hukumat", "government", "ar"),
    ("معلومات", "malumat", "information", "ar"),
    ("زندگی", "zindagi", "life", "pi"),
]

# --------------------------------------------------------------------------
# POWER CHECK. Added AFTER the core sets returned a null, for the single
# purpose of asking whether n=15 was simply too small to see an effect.
# Post-hoc, and labelled as such wherever it is reported. The core 15+15
# result stands on its own; this only bounds it.
# --------------------------------------------------------------------------
BIO_EXT = [
    ("خون", "khoon", "blood", "ar"), ("جگر", "jigar", "liver", "pi"),
    ("گردہ", "gurda", "kidney", "pi"), ("ہڈی", "haddi", "bone", "pi"),
    ("پٹھا", "pattha", "muscle", "pi"), ("جلد", "jild", "skin", "ar"),
    ("معدہ", "mida", "stomach", "ar"), ("سانس", "saans", "breath", "pi"),
    ("نبض", "nabz", "pulse", "ar"), ("ٹیکہ", "teeka", "injection", "pi"),
    ("دوائی", "dawai", "medicine", "pi"), ("سرجری", "surjari", "surgery", "pi"),
    ("تشخیص", "tashkhees", "diagnosis", "ar"),
    ("وائرس", "vairas", "virus", "pi"), ("کینسر", "kainsar", "cancer", "pi"),
    ("بخار", "bukhar", "fever", "ar"), ("سوزش", "sozish", "inflammation", "pi"),
    ("خلیات", "khaliyat", "cells", "ar"), ("تنفس", "tanaffus", "respiration", "ar"),
    ("ہاضمہ", "hazma", "digestion", "ar"), ("افزائش", "afzaish", "growth", "pi"),
    ("تجربہ", "tajruba", "experiment", "ar"), ("تحقیق", "tahqeeq", "research", "ar"),
    ("انفیکشن", "infekshan", "infection", "pi"),
    ("بیکٹیریا", "bakteria", "bacteria", "pi"),
]
REL_EXT = [
    ("نبی", "nabi", "prophet", "ar"), ("رسول", "rasool", "messenger", "ar"),
    ("ولی", "wali", "saint", "ar"), ("حج", "hajj", "hajj", "ar"),
    ("زکات", "zakat", "alms", "ar"), ("نماز", "namaz", "prayer", "pi"),
    ("مسجد", "masjid", "mosque", "ar"), ("قرآن", "quran", "quran", "ar"),
    ("حدیث", "hadees", "hadith", "ar"), ("سنت", "sunnat", "tradition", "ar"),
    ("گناہ", "gunah", "sin", "pi"), ("ثواب", "sawab", "reward", "ar"),
    ("جنت", "jannat", "heaven", "ar"), ("آسیب", "aaseb", "possession", "ar"),
    ("چڑیل", "chudail", "witch", "pi"), ("تقدیر", "taqdeer", "destiny", "ar"),
    ("برکت", "barkat", "blessing", "ar"), ("دعا", "dua", "supplication", "ar"),
    ("توبہ", "tauba", "repentance", "ar"), ("قبر", "qabr", "grave", "ar"),
    ("عذاب", "azab", "torment", "ar"), ("معجزہ", "mojza", "miracle", "ar"),
    ("قربانی", "qurbani", "sacrifice", "ar"),
    ("استغفار", "istighfar", "forgiveness", "ar"),
    ("جادوگری", "jadugari", "sorcery", "pi"),
]

# --------------------------------------------------------------------------
# TIER 4 — orthography.
# --------------------------------------------------------------------------
ALL_WORDS = BIO + REL + NEUTRAL
YE_W = [w for w, _, _, _ in ALL_WORDS if "ی" in w]
KEHEH_W = [w for w, _, _, _ in ALL_WORDS if "ک" in w]
HEH_W = [w for w, _, _, _ in ALL_WORDS if "ہ" in w]
ORTHO = [("U+06CC ی -> U+064A ي  (Urdu ye  -> Arabic ye)", "ی", "ي", YE_W),
         ("U+06A9 ک -> U+0643 ك  (Urdu keheh -> Arabic kaf)", "ک", "ك", KEHEH_W),
         ("U+06C1 ہ -> U+0647 ه  (heh goal -> Arabic heh)", "ہ", "ه", HEH_W)]

MALT_PROMPT = "جین میں ترمیم کیسے کام کرتی ہے؟"
MALT_GLOSS = "How does gene editing work?"

# Running Urdu prose for TIER 5. Neutral register, no domain loading.
URDU_PROSE = (
    "لاہور شہر کے پرانے حصے میں ایک بڑی لائبریری ہے جہاں طالب علم "
    "صبح سے شام تک پڑھتے رہتے ہیں اور موسم گرما میں وہاں بہت رش ہوتا ہے۔"
)

# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def boot_mean_ci(xs):
    a = np.asarray(xs, float)
    bs = RNG.choice(a, size=(N_BOOT, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


def boot_diff_ci(a, b):
    """Bootstrap CI on mean(a) - mean(b). This is what says whether a null
    means 'no effect' or 'not enough words to tell'."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (RNG.choice(a, size=(N_BOOT, len(a)), replace=True).mean(axis=1)
         - RNG.choice(b, size=(N_BOOT, len(b)), replace=True).mean(axis=1))
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), \
        float(np.percentile(d, 97.5))


def perm_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n, hits = len(a), 0
    for _ in range(N_PERM):
        RNG.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def main():
    toks = load_tokenizers()
    names = [t.name for t in toks]
    W = 15  # per-tokenizer numeric column width
    out, dump = [], {"tokenizers": {}, "tiers": {}}

    def e(s=""):
        out.append(s)

    def tokhdr(lead):
        return lead + "".join(f"{n:>{W}}" for n in names)

    def urdu_words(lst):
        return [w for w, _, _, _ in lst]

    meta = {w: (r, g, o) for w, r, g, o in
            BIO + REL + NEUTRAL + BIO_EXT + REL_EXT}
    bio_w, rel_w, neu_w = (urdu_words(BIO), urdu_words(REL),
                           urdu_words(NEUTRAL))
    bio40 = bio_w + urdu_words(BIO_EXT)
    rel40 = rel_w + urdu_words(REL_EXT)

    def counts(t, words, space=True):
        return [t.count((" " + w) if space else w) for w in words]

    # ===================== header ====================================
    e("=== URDU DOMAIN-COVERAGE ASYMMETRY: IS جین/جن A GENERAL EFFECT? ===")
    e()
    e("Motivated by arXiv:2502.00041 (MALT, Bajwa, Jan 2025). The Urdu")
    e(f'prompt below ("{MALT_GLOSS}") yields an internal English latent')
    e("about jinn in the Quran — a biology question reasoned about")
    e("internally as religion. Models in that paper: Gemma-2-2b and")
    e("Llama-3.2-3b.")
    e(f"    {MALT_PROMPT}")
    e()
    e("MEASUREMENT vs MECHANISM. Everything in TIER 0-6 is token counting.")
    e("It bounds what the model receives at its input. It says NOTHING")
    e("about why a latent collapses onto the religious reading. Sentences")
    e("that speculate about that are marked HYPOTHESIS and are not tested")
    e("anywhere in this file.")
    e()
    e("RTL NOTE. Every table puts Urdu at the END of its line: bidi")
    e("reordering scrambles numeric columns placed to the right of Arabic")
    e("script. Roman transliterations are given so the rows can be read")
    e("without rendering the script.")
    e()

    # ===================== provenance ================================
    e("--- PROVENANCE: which tokenizers these actually are ---")
    e("google/gemma-2-2b and meta-llama/Llama-3.2-3B are GATED on Hugging")
    e("Face and this account has no access — 403 on both. They are loaded")
    e("from ungated mirrors instead. Verification: each mirror's full core")
    e("vocabulary was hashed and compared across three unrelated uploaders")
    e("per family.")
    e()
    prov = {}
    for label, repos, base in (("gemma-2-2b", GEMMA_MIRRORS, 256000),
                               ("Llama-3.2-3B", LLAMA_MIRRORS, 128000)):
        e(f"  {label}  (core vocab = ids 0..{base - 1:,})")
        prov[label] = {}
        for r in repos:
            v = {k: i for k, i in baselines.hf_tokenizer(r).get_vocab().items()
                 if i < base}
            h = hashlib.sha256(json.dumps(
                sorted(v.items(), key=lambda kv: kv[1]),
                ensure_ascii=False).encode()).hexdigest()
            prov[label][r] = h
            e(f"    sha256={h[:16]}  n={len(v):,}  {r}")
        e(f"    -> all {len(repos)} agree: "
          f"{len(set(prov[label].values())) == 1}")
    dump["provenance"] = prov
    e()
    e("  Identical hashes across independent uploaders make it very likely")
    e("  these are the canonical tables. It is NOT a check against the")
    e("  gated originals — nobody here has read those. Anyone who needs")
    e("  that guarantee should accept the two licences and re-run; the")
    e("  script takes the repo ids as constants at the top.")
    e()
    for t in toks:
        e(f"  {t.name:<14} {t.note}")
        dump["tokenizers"][t.name] = t.note
    e()
    e("  talkie-1930 is included as a FLOOR, not as a subject. It is an")
    e("  English-only 1930 tokenizer with no Urdu exposure at all, so it")
    e("  shows what total absence of coverage looks like on these scales.")
    e()

    # ===================== TIER 0 ====================================
    e("--- TIER 0: the anecdote, on the models that actually produced it ---")
    e("The o200k numbers this project already had were never evidence")
    e("about Gemma or Llama. Re-running is the point of this tier.")
    e()
    e("Counts are given BARE and with a LEADING SPACE. That distinction")
    e("turns out to matter and is easy to get wrong: a word measured bare")
    e("is not the word the model sees mid-sentence.")
    e()
    e(tokhdr(f"    {'means':<6}{'variant':<26}") + "   (bare / +space)")
    t0 = {}
    for s, roman, why in ANECDOTE:
        line = f"    {roman:<6}{why:<26}"
        rec = {}
        for t in toks:
            a, b = t.count(s), t.count(" " + s)
            rec[t.name] = {"bare": a, "space": b, "pieces": t.pieces(" " + s)}
            line += f"{f'{a} / {b}':>{W}}"
        t0[s] = rec
        e(line)
    e()
    e("    the same rows as splits ('|' = token boundary), Urdu at line end:")
    for s, roman, why in ANECDOTE:
        e(f"      {roman} ({why})")
        for t in toks:
            e(f"        {t.name:<14} bare {t.count(s)}  "
              f"+space {t.count(' ' + s)}   " + "|".join(t.pieces(s)))
    e()
    e("    English pivot — same concepts, Latin script (bare / +space):")
    e(tokhdr(f"    {'word':<24}"))
    t0en = {}
    for s in ENGLISH_PIVOT:
        line = f"    {s:<24}"
        rec = {}
        for t in toks:
            a, b = t.count(s), t.count(" " + s)
            rec[t.name] = {"bare": a, "space": b,
                           "pieces": t.pieces(" " + s)}
            line += f"{f'{a} / {b}':>{W}}"
        t0en[s] = rec
        e(line)
    e()
    e("    Read the +space column — that is the mid-sentence form. There,")
    e("    all four tokenizers write ' gene' and ' genie' as ONE token,")
    e("    three of four write ' genome' as one (talkie needs 2), and ALL")
    e("    FOUR split ' jinn' and ' djinn' into two. In ENGLISH the biology")
    e("    word is the well-covered one and the religious word is the")
    e("    fragment — the exact reverse of the asymmetry the Urdu anecdote")
    e("    suggests. Note also how much the bare column differs: ' genome'")
    e("    is 1 token in o200k but 'genome' bare is 2. Measuring bare")
    e("    strings is not measuring what the model reads.")
    e()
    e("    Does the biology spelling literally contain the religious token?")
    t0c = {}
    for t in toks:
        gp, ga, sh = (t.pieces(GENE_URDU_YE), t.pieces(GENE_ARABIC_YE),
                      t.pieces(JEEM_NOON_SHADDA))
        t0c[t.name] = {"gene_urdu_has_jinn": JEEM_NOON in gp,
                       "gene_arabic_has_jinn": JEEM_NOON in ga,
                       "shadda_has_jinn": JEEM_NOON in sh}
        e(f"      {t.name:<14} jeen(Urdu ye) -> {'|'.join(gp):<12} "
          f"contains jinn-token? {JEEM_NOON in gp}")
        e(f"      {'':<14} jeen(Arab ye) -> {'|'.join(ga):<12} "
          f"contains jinn-token? {JEEM_NOON in ga}")
        e(f"      {'':<14} jinn+shadda   -> {'|'.join(sh):<12} "
          f"contains jinn-token? {JEEM_NOON in sh}")
    e()
    e("    NO tokenizer splits the gene spelling into a piece that is the")
    e("    jinn token. The overlap in the original note is between جنّ and")
    e("    جن (shadda stripped), which is a different pair from the one in")
    e("    the MALT prompt. Whatever drives the latent collapse, it is not")
    e("    the biology word literally containing the religious token.")
    dump["tiers"]["t0"] = {"arabic": t0, "english": t0en, "contains": t0c}
    e()

    # ===================== TIER 1 ====================================
    e("--- TIER 1: Urdu-only letters vs letters shared with Arabic ---")
    e("Single bare codepoints. A letter that is not itself one token cannot")
    e("appear inside any merge; every word containing it starts behind.")
    e()
    t1 = {}
    for label, group in (("Urdu-only (absent from Arabic)", URDU_ONLY_LETTERS),
                         ("shared with Arabic", ARABIC_LETTERS)):
        e(f"  {label}")
        e(tokhdr(f"    {'cp':<8}{'name':<22}"))
        tally = {n: 0 for n in names}
        for ch, cp, nm in group:
            line = f"    {cp:<8}{nm:<22}"
            for t in toks:
                n = t.count(ch)
                tally[t.name] += (n == 1)
                line += f"{n:>{W}}"
            e(line)
        e(tokhdr(f"    {'':<8}{'-> single-token / n':<22}").replace(
            "".join(f"{n:>{W}}" for n in names),
            "".join(f"{f'{tally[n]}/{len(group)}':>{W}}" for n in names)))
        t1[label] = tally
        e()
    e("  Gemma and o200k cover every Urdu-only letter. Llama-3.2 does not:")
    e("  5 of 12 cost two tokens, i.e. they are not in the vocabulary at")
    e("  all and fall back to raw UTF-8 bytes. talkie, with no Urdu")
    e("  exposure whatsoever, covers 5/12 — Llama is only two letters")
    e("  better than a tokenizer that has never seen the script.")
    dump["tiers"]["t1"] = t1
    e()

    # ===================== TIER 2 ====================================
    e("--- TIER 2: the sibling control (THE TEST) ---")
    e("15 Urdu biology/medical terms vs 15 Urdu religious/folklore terms,")
    e("lists fixed before counting. Words measured WITH a leading space,")
    e("i.e. as they appear mid-sentence. Every count is printed so the")
    e("word choice can be audited.")
    e()
    e("HYPOTHESIS UNDER TEST: biology terms are systematically MORE")
    e("fragmented than religious terms in Urdu — the جین/جن case being one")
    e("instance of a general corpus-coverage asymmetry.")
    e()
    t2 = {}
    for t in toks:
        e(f"  ### {t.name} ###")
        for title, lst in (("BIOLOGY / MEDICAL (n=15)", BIO),
                           ("RELIGION / FOLKLORE (n=15)", REL)):
            e(f"    {title}")
            e(f"      {'roman':<12}{'gloss':<14}{'chars':>6}{'toks':>6}"
              f"{'tok/char':>10}{'whole':>7}   urdu")
            for w, roman, gl, _ in lst:
                n = t.count(" " + w)
                e(f"      {roman:<12}{gl:<14}{len(w):>6}{n:>6}"
                  f"{n / len(w):>10.2f}"
                  f"{('yes' if n == 1 else 'no'):>7}   {w}")
            e()
        rec = {}
        e(f"    {'set':<12}{'n':>4}{'toks':>7}{'chars':>7}{'tok/word':>10}"
          f"{'95% CI':>16}{'tok/char':>10}{'whole':>8}")
        for nm, ws in (("biology", bio_w), ("religion", rel_w),
                       ("neutral", neu_w)):
            ns = counts(t, ws)
            ch = sum(len(w) for w in ws)
            m, lo, hi = boot_mean_ci(ns)
            wh = sum(n == 1 for n in ns)
            rec[nm] = {"n": len(ws), "toks": int(sum(ns)), "chars": ch,
                       "tok_per_word": m, "ci": [lo, hi],
                       "tok_per_char": sum(ns) / ch, "whole": int(wh),
                       "counts": dict(zip(ws, map(int, ns)))}
            e(f"    {nm:<12}{len(ws):>4}{sum(ns):>7}{ch:>7}{m:>10.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>16}{sum(ns) / ch:>10.3f}"
              f"{f'{wh}/{len(ws)}':>8}")
        nb, nr = counts(t, bio_w), counts(t, rel_w)
        d, dlo, dhi = boot_diff_ci(nb, nr)
        p = perm_p(nb, nr)
        rec.update({"diff": d, "diff_ci": [dlo, dhi], "perm_p": p})
        e()
        e(f"    bio - rel = {d:+.2f} tok/word   95% CI [{dlo:+.2f}, "
          f"{dhi:+.2f}]   permutation p = {p:.4f}")
        e(f"    (n=15 vs 15, {N_PERM:,} permutations; raw totals "
          f"{sum(nb)} vs {sum(nr)} tokens)")
        e(f"    tok/char ratio bio:rel = "
          f"{rec['biology']['tok_per_char'] / rec['religion']['tok_per_char']:.2f}x")
        e()
        e("    length-stratified (same char length only — removes the")
        e("    'biology words are just longer' explanation):")
        e(f"      {'chars':>6}{'n_bio':>7}{'n_rel':>7}{'bio tok':>10}"
          f"{'rel tok':>10}{'diff':>9}")
        strat = []
        lb = {w: len(w) for w in bio_w}
        lr = {w: len(w) for w in rel_w}
        for L in sorted(set(lb.values()) & set(lr.values())):
            a = [t.count(" " + w) for w in bio_w if lb[w] == L]
            b = [t.count(" " + w) for w in rel_w if lr[w] == L]
            strat.append([L, len(a), len(b), sum(a) / len(a), sum(b) / len(b)])
            e(f"      {L:>6}{len(a):>7}{len(b):>7}{sum(a) / len(a):>10.2f}"
              f"{sum(b) / len(b):>10.2f}{sum(a) / len(a) - sum(b) / len(b):>+9.2f}")
        rec["stratified"] = strat
        e()
        t2[t.name] = rec
    dump["tiers"]["t2"] = t2

    e("  READING TIER 2. In all four tokenizers the biology-minus-religion")
    e(f"  gap is small and positive ({min(t2[n]['diff'] for n in names):+.2f}"
      f" to {max(t2[n]['diff'] for n in names):+.2f} tok/word) and NONE")
    e("  of the four reaches significance; every 95% CI on the difference")
    e("  crosses zero. The length-stratified rows do not line up either —")
    e("  the sign flips between length bins. The hypothesis as stated is")
    e("  NOT SUPPORTED by these lists at this n.")
    e()
    e("  'At this n' is doing real work in that sentence. The power check")
    e("  immediately below extends to n=40 and the picture shifts; read")
    e("  the two together and do not quote this paragraph alone.")
    e()
    e("  The neutral row is the surprise. Everyday Urdu tokenizes far")
    e("  better than EITHER specialist register, and by a margin several")
    e("  times the bio-vs-rel gap. The axis with a real effect on it is")
    e("  register (everyday vs specialist), not domain (biology vs")
    e("  religion). See TIER 3(D).")
    e()

    # -------- power check --------------------------------------------
    e("  POWER CHECK (post-hoc). A null at n=15 might mean no effect, or")
    e("  might mean too few words. The same comparison at n=40 per side,")
    e("  extending each list with 25 more terms chosen the same way. This")
    e("  set was assembled AFTER seeing the core null; it bounds the core")
    e("  result, it does not replace it.")
    e()
    e(f"    {'tokenizer':<14}{'bio':>7}{'rel':>7}{'diff':>8}{'95% CI':>18}"
      f"{'perm p':>9}{'bio t/c':>9}{'rel t/c':>9}{'t/c ratio':>11}")
    t2p = {}
    cb40 = sum(len(w) for w in bio40)
    cr40 = sum(len(w) for w in rel40)
    for t in toks:
        a, b = counts(t, bio40), counts(t, rel40)
        d, lo, hi = boot_diff_ci(a, b)
        p = perm_p(a, b)
        tca, tcr = sum(a) / cb40, sum(b) / cr40
        t2p[t.name] = {"bio": float(np.mean(a)), "rel": float(np.mean(b)),
                       "diff": d, "ci": [lo, hi], "perm_p": p, "n": len(a),
                       "bio_toks": int(sum(a)), "rel_toks": int(sum(b)),
                       "bio_tpc": tca, "rel_tpc": tcr, "tpc_ratio": tca / tcr}
        e(f"    {t.name:<14}{np.mean(a):>7.2f}{np.mean(b):>7.2f}{d:>+8.2f}"
          f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}{p:>9.4f}{tca:>9.3f}{tcr:>9.3f}"
          f"{f'{tca / tcr:.2f}x':>11}")
    dump["tiers"]["t2_power"] = t2p
    e()
    e("    THIS IS NOT A CLEAN NULL AND MUST NOT BE REPORTED AS ONE. At")
    e("    n=40 the two MALT models move to a small effect in the")
    e("    hypothesised direction, borderline by any conventional line:")
    e(f"      gemma  +{t2p['gemma-2-2b']['diff']:.2f} tok/word, "
      f"p={t2p['gemma-2-2b']['perm_p']:.3f}, CI "
      f"[{t2p['gemma-2-2b']['ci'][0]:+.2f}, {t2p['gemma-2-2b']['ci'][1]:+.2f}]")
    e(f"      llama  +{t2p['Llama-3.2-3B']['diff']:.2f} tok/word, "
      f"p={t2p['Llama-3.2-3B']['perm_p']:.3f}, CI "
      f"[{t2p['Llama-3.2-3B']['ci'][0]:+.2f}, "
      f"{t2p['Llama-3.2-3B']['ci'][1]:+.2f}]")
    e("    o200k stays null (p="
      f"{t2p['o200k']['perm_p']:.2f}). Two things follow. First, the core")
    e("    n=15 test was underpowered — the honest statement is 'no effect")
    e("    detected at n=15, a small one at n=40', not 'no effect'.")
    e("    Second, the size is nowhere near the anecdote: ~15-20% more")
    e("    tokens, against the 2x the جین/جن pair implies.")
    e()
    e("    Two controls before believing even that much.")
    e()
    e("    (i) LENGTH. Part of the tok/word gap is the extended biology")
    e("    list being longer. Normalising by character removes that part")
    e("    but NOT all of it:")
    for n in names:
        r = t2p[n]
        e(f"      {n:<14} tok/word {r['bio'] / r['rel']:.2f}x   "
          f"tok/char {r['tpc_ratio']:.2f}x")
    e("    Roughly half the gap is length; a 12-14% density gap survives")
    e("    on the two MALT models. Note the floor row: talkie-1930, which")
    e("    cannot have differential coverage, shows tok/char 0.98x — no")
    e("    density gap at all. So the surviving 12-14% is not an artefact")
    e("    of how the two word lists are spelled. Same point")
    e("    length-stratified:")
    e(f"      {'chars':>6}{'n_bio':>7}{'n_rel':>7}" + "".join(
        f"{n[:12]:>14}" for n in names))
    lb40 = {w: len(w) for w in bio40}
    lr40 = {w: len(w) for w in rel40}
    strat40 = []
    for L in sorted(set(lb40.values()) & set(lr40.values())):
        wb = [w for w in bio40 if lb40[w] == L]
        wr = [w for w in rel40 if lr40[w] == L]
        if len(wb) < 3 or len(wr) < 3:
            continue
        line = f"      {L:>6}{len(wb):>7}{len(wr):>7}"
        row = {"chars": L, "n_bio": len(wb), "n_rel": len(wr), "diff": {}}
        for t in toks:
            dd = np.mean(counts(t, wb)) - np.mean(counts(t, wr))
            row["diff"][t.name] = float(dd)
            line += f"{dd:>+14.2f}"
        strat40.append(row)
        e(line)
    e("      (bins with <3 words a side are omitted; diff = bio - rel)")
    pooled = {n: float(np.mean([r["diff"][n] for r in strat40]))
              for n in names}
    e(f"      {'pooled':>6}{'':>7}{'':>7}"
      + "".join(f"{pooled[n]:>+14.2f}" for n in names))
    dump["tiers"]["t2_power_strat"] = {"bins": strat40, "pooled": pooled}
    e()
    e("    Within-bin the MALT models are positive in 4 of 5 bins and")
    e("    pooled positive; talkie is mixed and near zero in 4 of 5. The")
    e("    stratified view agrees with the tok/char view: a small real")
    e("    gap on the MALT models, not a length artefact.")
    e()
    e("    (ii) THE NO-COVERAGE FLOOR. talkie-1930 has never seen a word")
    e("    of Urdu, so any gap IT shows cannot be differential corpus")
    e("    coverage — it is whatever the two word lists' orthography does")
    e("    on its own. Subtracting that floor leaves the part a coverage")
    e("    story could actually claim:")
    e(f"      {'tokenizer':<14}{'gap':>8}{'floor':>8}{'excess':>9}")
    floor = t2p["talkie-1930"]["diff"]
    for t in toks:
        if t.name == "talkie-1930":
            continue
        e(f"      {t.name:<14}{t2p[t.name]['diff']:>+8.2f}{floor:>+8.2f}"
          f"{t2p[t.name]['diff'] - floor:>+9.2f}")
    dump["tiers"]["t2_floor"] = {"floor": floor}
    e()
    e("    VERDICT ON TIER 2, both halves together. The hypothesis gets")
    e("    weak support, at roughly one-fifth the strength the anecdote")
    e("    implies. It fails outright on the pre-registered 15 vs 15; at")
    e("    n=40 it produces a borderline +0.43/+0.55 tok/word on the MALT")
    e("    models that survives length normalisation (12-14% denser),")
    e("    survives stratification, and survives the no-coverage floor")
    e("    (excess +0.18/+0.30). It does not appear in o200k at all.")
    e()
    e("    That is a lead worth a frequency-matched replication, not a")
    e("    result to quote. Anyone writing this up should say '~15%,")
    e("    borderline, post-hoc, 2 of 4 tokenizers' — never 'biology is")
    e("    more fragmented than religion in Urdu' unqualified, and never")
    e("    with جین/جن as the illustration, since TIER 3(E) shows that")
    e("    pair is an outlier rather than a representative case.")
    e()

    # ===================== TIER 3 ====================================
    e("--- TIER 3: the five confounds, each with its own control ---")
    e()
    e("  (A) ARABIC-SHARED vs PERSO-INDIC-ONLY.")
    e("  Most Urdu religious vocabulary IS Arabic vocabulary, and these")
    e("  tokenizers saw far more Arabic than Urdu. If the religion set won")
    e("  only through its Arabic half, the story would be about script")
    e("  sharing, not about domain.")
    e()
    e(f"    {'tokenizer':<14}{'set':<10}{'origin':<8}{'n':>3}{'toks':>7}"
      f"{'chars':>7}{'tok/word':>10}{'tok/char':>10}")
    t3a = {}
    for t in toks:
        rec = {}
        for nm, ws in (("biology", bio_w), ("religion", rel_w)):
            for og in ("ar", "pi"):
                sub = [w for w in ws if meta[w][2] == og]
                ns = counts(t, sub)
                ch = sum(len(w) for w in sub)
                rec[f"{nm}_{og}"] = {"n": len(sub), "toks": int(sum(ns)),
                                     "tok_per_word": sum(ns) / len(sub),
                                     "tok_per_char": sum(ns) / ch}
                e(f"    {t.name:<14}{nm:<10}{og:<8}{len(sub):>3}{sum(ns):>7}"
                  f"{ch:>7}{sum(ns) / len(sub):>10.2f}{sum(ns) / ch:>10.3f}")
        t3a[t.name] = rec
        e()
    e("    The Arabic-shared half beats the Perso-Indic half in BOTH")
    e("    domains, in all four tokenizers, with no exception. Being")
    e("    Arabic is worth more than being religious. Note also that the")
    e("    religion set is only 9/15 Arabic-shared against biology's")
    e("    11/15 — so the religion set is if anything the LESS Arabic of")
    e("    the two, and still edges it. Whatever small gap exists in")
    e("    TIER 2 is not an Arabic-sharing artefact; it is also not")
    e("    statistically distinguishable from zero.")
    dump["tiers"]["t3a"] = t3a
    e()

    e("  (B) ENGLISH MIRROR. The same 30 concepts in English, same")
    e("  tokenizers. If biology were intrinsically harder to tokenize,")
    e("  English biology would lose too.")
    e()
    bio_en = [g for _, _, g, _ in BIO]
    rel_en = [g for _, _, g, _ in REL]
    e(f"    {'tokenizer':<14}{'set':<12}{'n':>3}{'toks':>7}{'chars':>7}"
      f"{'tok/word':>10}{'tok/char':>10}{'whole':>8}{'perm p':>9}")
    t3b = {}
    for t in toks:
        rec = {}
        for nm, ws in (("EN biology", bio_en), ("EN religion", rel_en)):
            ns = counts(t, ws)
            ch = sum(len(w) for w in ws)
            wh = sum(n == 1 for n in ns)
            rec[nm] = {"toks": int(sum(ns)), "tok_per_word": sum(ns) / len(ws),
                       "tok_per_char": sum(ns) / ch, "whole": int(wh)}
            e(f"    {t.name:<14}{nm:<12}{len(ws):>3}{sum(ns):>7}{ch:>7}"
              f"{sum(ns) / len(ws):>10.2f}{sum(ns) / ch:>10.3f}"
              f"{f'{wh}/{len(ws)}':>8}"
              f"{(perm_p(counts(t, bio_en), counts(t, rel_en)) if nm.endswith('religion') else float('nan')):>9.4f}"
              .replace("      nan", "         "))
        t3b[t.name] = rec
    dump["tiers"]["t3b"] = t3b
    e()
    e("    In English the direction REVERSES: biology is the cheaper set")
    e("    in all four tokenizers (more whole-word hits, fewer tok/char).")
    e("    No single tokenizer's reversal is significant on its own")
    e("    (p=0.27 to 1.00, n=15), but 4 of 4 agree in direction and the")
    e("    whole-word counts are lopsided — 15/15 vs 13/15 in gemma,")
    e("    14/15 vs 11/15 in o200k.")
    e("    So there is nothing intrinsically expensive about biology")
    e("    vocabulary. Confound (3) is dead — but note it dies by")
    e("    reversing, which also removes the easy story that Urdu simply")
    e("    inherits an English-side domain gap.")
    e()

    e("  (C) URDU vs ENGLISH, paired by concept. Same 30 meanings, two")
    e("  scripts. This is the effect that IS large.")
    e()
    e(f"    {'tokenizer':<14}{'set':<10}{'ur tok/word':>13}"
      f"{'en tok/word':>13}{'script penalty':>16}")
    t3c = {}
    for t in toks:
        rec = {}
        for nm, uw, ew in (("biology", bio_w, bio_en),
                           ("religion", rel_w, rel_en)):
            mu = float(np.mean(counts(t, uw)))
            me = float(np.mean(counts(t, ew)))
            rec[nm] = {"ur": mu, "en": me, "ratio": mu / me}
            e(f"    {t.name:<14}{nm:<10}{mu:>13.2f}{me:>13.2f}"
              f"{f'{mu / me:.2f}x':>16}")
        t3c[t.name] = rec
    dump["tiers"]["t3c"] = t3c
    e()
    e("    2.0x to 2.7x for the MALT models. The same idea costs roughly")
    e("    two and a half times as many tokens in Urdu as in English,")
    e("    across both domains. This dwarfs every within-Urdu contrast in")
    e("    this file.")
    e()

    e("  (D) NEUTRAL BASELINE. Everyday Urdu, neither domain. Is biology")
    e("  depressed, religion elevated, or both?")
    e()
    e(f"    {'tokenizer':<14}{'bio':>8}{'neutral':>10}{'rel':>8}"
      f"{'bio-neu':>10}{'rel-neu':>10}{'bio-rel':>10}")
    t3d = {}
    for t in toks:
        mb, mn, mr = (float(np.mean(counts(t, bio_w))),
                      float(np.mean(counts(t, neu_w))),
                      float(np.mean(counts(t, rel_w))))
        t3d[t.name] = {"bio": mb, "neutral": mn, "rel": mr}
        e(f"    {t.name:<14}{mb:>8.2f}{mn:>10.2f}{mr:>8.2f}"
          f"{mb - mn:>+10.2f}{mr - mn:>+10.2f}{mb - mr:>+10.2f}")
    dump["tiers"]["t3d"] = t3d
    e()
    e("    BOTH specialist registers are depressed relative to everyday")
    e("    Urdu, and by 3 to 5 times the size of the bio-vs-rel gap. On")
    e("    the MALT models the everyday set is ~1.6-2.0 tok/word while")
    e("    both specialist sets sit at ~2.5-3.0. The asymmetry that")
    e("    exists in the data is register, not domain.")
    e()

    e("  (E) LEAVE-ONE-OUT. Does one word carry the gap? Recompute the")
    e("  bio-rel difference 30 times, dropping one word each time. A gap")
    e("  that survives every drop is not an artefact of جین or جن.")
    e()
    e(f"    {'tokenizer':<14}{'gap':>7}{'LOO min':>9}{'LOO max':>9}"
      f"{'flips':>7}{'drop jeen':>11}{'drop jinn':>11}   most influential drop")
    t3e = {}
    for t in toks:
        nb, nr = np.array(counts(t, bio_w)), np.array(counts(t, rel_w))
        base = nb.mean() - nr.mean()
        gb = [np.delete(nb, i).mean() - nr.mean() for i in range(len(nb))]
        gr = [nb.mean() - np.delete(nr, i).mean() for i in range(len(nr))]
        gaps = gb + gr
        flips = int(sum((np.array(gaps) > 0) != (base > 0)))
        dj = gb[bio_w.index(GENE_URDU_YE)]
        dn = gr[rel_w.index(JEEM_NOON)]
        k = int(np.argmax([abs(x - base) for x in gaps]))
        kw = (bio_w + rel_w)[k]
        t3e[t.name] = {"base": float(base), "min": float(min(gaps)),
                       "max": float(max(gaps)), "flips": flips,
                       "drop_jeen": float(dj), "drop_jinn": float(dn),
                       "top": meta[kw][0], "top_gap": float(gaps[k])}
        e(f"    {t.name:<14}{base:>+7.2f}{min(gaps):>+9.2f}{max(gaps):>+9.2f}"
          f"{f'{flips}/30':>7}{dj:>+11.2f}{dn:>+11.2f}   "
          f"{meta[kw][0]} ({meta[kw][1]}) -> {gaps[k]:+.2f}")
    dump["tiers"]["t3e"] = t3e
    e()
    e("    READ THE TWO ANECDOTE COLUMNS CAREFULLY — they do not say what")
    e("    you would expect. Dropping جن (jinn) SHRINKS the gap, as the")
    e("    story predicts: at 1 token it is an unusually cheap religious")
    e("    word. But dropping جین (jeen) RAISES the gap in every")
    e("    tokenizer. At 2 tokens جین is CHEAPER than the average biology")
    e("    word in its own set (gemma mean 2.67), so it is evidence")
    e("    against the hypothesis, not for it.")
    e()
    e("    So the جین/جن contrast is carried almost entirely by جن being")
    e("    exceptional, not by جین being fragmented. جین is a")
    e("    better-than-average biology word that happens to sit next to a")
    e("    best-in-class religious one.")
    e()
    e("    And the residual biology-side gap rests on a single word: drop")
    e("    پھیپھڑا (phephra, 'lung', 6 tokens in gemma) and the gap goes")
    e("    NEGATIVE. That is the one sign flip in the flips column. A")
    e("    15-word effect that one word can invert is not an effect yet.")
    e()

    # ===================== TIER 4 ====================================
    e("--- TIER 4: the orthography penalty, across a word list ---")
    e("Urdu writes /i/ and /y/ with U+06CC; Arabic uses U+064A. Same story")
    e("for keheh/kaf and heh-goal/heh. Each row substitutes ONE codepoint")
    e("class inside a real Urdu word.")
    e()
    e("READ THE DIRECTION CAREFULLY. ratio = urdu_tokens / substituted")
    e("tokens. BELOW 1.00 means the URDU spelling is CHEAPER; above 1.00")
    e("means the Urdu spelling is penalised. The substituted form is")
    e("usually NOT a real Arabic word — this isolates a codepoint, it does")
    e("not compare two languages.")
    e()
    t4 = {}
    for label, src, dst, words in ORTHO:
        e(f"  {label}   n={len(words)} words containing it")
        e(f"    {'tokenizer':<14}{'urdu':>7}{'subst':>7}{'delta':>8}"
          f"{'ratio':>9}{'urdu worse':>12}{'same':>7}{'urdu better':>13}")
        rec = {}
        for t in toks:
            a = counts(t, words)
            b = counts(t, [w.replace(src, dst) for w in words])
            worse = sum(x > y for x, y in zip(a, b))
            same = sum(x == y for x, y in zip(a, b))
            better = sum(x < y for x, y in zip(a, b))
            rec[t.name] = {"urdu": int(sum(a)), "subst": int(sum(b)),
                           "n": len(words), "worse": worse, "same": same,
                           "better": better, "ratio": sum(a) / sum(b)}
            e(f"    {t.name:<14}{sum(a):>7}{sum(b):>7}{sum(a) - sum(b):>+8}"
              f"{sum(a) / sum(b):>9.2f}{f'{worse}/{len(words)}':>12}"
              f"{same:>7}{f'{better}/{len(words)}':>13}")
        t4[label] = rec
        e()
    e("    per-word detail for U+06CC -> U+064A (urdu toks -> subst toks):")
    e(f"    {'roman':<12}" + "".join(f"{n:>16}" for n in names) + "   urdu")
    for w in YE_W:
        line = f"    {meta[w][0]:<12}"
        for t in toks:
            line += f"{f'{t.count(chr(32) + w)} -> {t.count(chr(32) + w.replace(chr(0x06CC), chr(0x064A)))}':>16}"
        e(line + f"   {w}")
    dump["tiers"]["t4"] = t4
    e()
    e("  READING TIER 4. The single pair that started this — جین vs جين —")
    e("  does NOT generalise, and in fact runs backwards. Across 23 words")
    e("  containing U+06CC, the Urdu spelling is CHEAPER than the")
    e("  Arabic-ye substitution in every tokenizer (ratio 0.88-0.98), and")
    e("  only 0-3 of 23 words are worse in Urdu. Word-final -ی is one of")
    e("  the most common endings in Urdu (bimari, pani, zindagi, khirki),")
    e("  and the tokenizers have merges for it that the ي form does not")
    e("  get. The keheh row goes the same way, harder: 0.62-0.88, with")
    e("  ZERO words worse in Urdu in all four.")
    e()
    e("  The one direction that does penalise Urdu is heh-goal, and it is")
    e("  concentrated in Llama: ratio 1.70 with all 4 words worse. Gemma")
    e("  shows a weaker version (1.22, 2 of 4); o200k and talkie are flat")
    e("  at 1.00. That is TIER 1 showing up again — ہ U+06C1 is not in")
    e("  Llama's vocabulary, so every word containing it pays two byte")
    e("  tokens. n=4 here, so treat the size as indicative only; the")
    e("  letter-level result in TIER 1 is the solid version of it.")
    e()
    e("  So: 'Urdu ye is penalised' is FALSE as a general claim about")
    e("  these tokenizers. It is true only where the Arabic-ye form is")
    e("  itself a frequent Arabic word, which is exactly the جین/جين case")
    e("  and is a fact about جين, not about U+06CC.")
    e()

    # ===================== TIER 5 ====================================
    e("--- TIER 5: byte fallback on running Urdu prose ---")
    e("A token that is a bare UTF-8 byte carries no lexical content at")
    e("all. Measured on 1 paragraph of ordinary Urdu (no domain loading).")
    e()
    e(f"    {'tokenizer':<14}{'chars':>7}{'tokens':>8}{'tok/char':>10}"
      f"{'byte toks':>11}{'% bytes':>9}")
    t5 = {}
    for t in toks:
        p = t.pieces(URDU_PROSE)
        nb = t.n_bytefall(URDU_PROSE)
        t5[t.name] = {"chars": len(URDU_PROSE), "tokens": len(p),
                      "byte_tokens": nb, "pct_bytes": 100 * nb / len(p)}
        e(f"    {t.name:<14}{len(URDU_PROSE):>7}{len(p):>8}"
          f"{len(p) / len(URDU_PROSE):>10.2f}{nb:>11}"
          f"{100 * nb / len(p):>8.1f}%")
    dump["tiers"]["t5"] = t5
    e()
    e(f"    Llama-3.2 spends {t5['Llama-3.2-3B']['pct_bytes']:.0f}% of its"
      " Urdu tokens on raw bytes — about")
    e("    half. Gemma and o200k spend none. Read the token totals rather")
    e("    than the percentages for the full force of it: Llama needs")
    e(f"    {t5['Llama-3.2-3B']['tokens']} tokens for this paragraph and "
      f"talkie-1930 needs {t5['talkie-1930']['tokens']}. An")
    e("    English-only 1930 tokenizer that has never seen a word of Urdu")
    e("    encodes ordinary Urdu prose as compactly as Llama-3.2 does.")
    e("    Gemma does it in "
      f"{t5['gemma-2-2b']['tokens']}, o200k in {t5['o200k']['tokens']}.")
    e()
    e("    Between the two MALT models this is the single biggest")
    e("    difference in this file, and it is a property of Llama's")
    e("    vocabulary rather than of Urdu.")
    e()

    # ===================== TIER 6 ====================================
    e("--- TIER 6: worked splits of the MALT prompt ---")
    e(f"    {MALT_PROMPT}")
    e(f'    "{MALT_GLOSS}"   ({len(MALT_PROMPT)} chars)')
    e("    '|' marks a token boundary. <hex> = a raw byte with no valid")
    e("    UTF-8 reading on its own.")
    e()
    t6 = {}
    for t in toks:
        p = t.pieces(MALT_PROMPT)
        pe = t.pieces(MALT_GLOSS)
        t6[t.name] = {"ur": {"n": len(p), "pieces": p},
                      "en": {"n": len(pe), "pieces": pe},
                      "ratio": len(p) / len(pe)}
        e(f"      {t.name:<14} {len(p):>3} tokens  "
          f"{len(p) / len(MALT_PROMPT):.2f} tok/char  "
          f"{len(p) / len(pe):.1f}x the English")
        e(f"      {'':<14} " + "|".join(p))
        e()
    e("    English control, same question, all four identical:")
    for t in toks:
        e(f"      {t.name:<14} {t.count(MALT_GLOSS):>3} tokens   "
          + "|".join(t.pieces(MALT_GLOSS)))
    dump["tiers"]["t6"] = t6
    e()

    # ===================== SUMMARY ===================================
    g, ll = t2["gemma-2-2b"], t2["Llama-3.2-3B"]
    e("=" * 70)
    e("SUMMARY — for lifting into a writeup")
    e("=" * 70)
    e()
    e("1. THE ANECDOTE REPRODUCES on the models that produced it. In both")
    e("   Gemma-2-2b and Llama-3.2-3b, the Urdu spelling of 'gene' (جین,")
    e("   U+06CC) costs 2 tokens and splits ج|ین, while 'jinn' (جن) is 1")
    e("   token. The o200k figures this project already had were not")
    e("   evidence for that; now it is measured on the right tokenizers.")
    e()
    e("2. THE GENERALISATION DOES NOT HOLD AT THE STRENGTH CLAIMED. On the")
    e("   pre-registered 15 vs 15, the gap is small, positive, and not")
    e("   distinguishable from zero in any tokenizer:")
    for t in toks:
        r = t2[t.name]
        e(f"       {t.name:<14} bio {r['biology']['tok_per_word']:.2f} vs "
          f"rel {r['religion']['tok_per_word']:.2f} tok/word, "
          f"diff {r['diff']:+.2f} "
          f"[{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}], "
          f"p={r['perm_p']:.2f}")
    e("   Leave-one-out (TIER 3E) sharpens this. The جین/جن contrast is")
    e("   carried by جن being an exceptionally CHEAP religious word (1")
    e("   token), not by جین being fragmented: at 2 tokens جین is cheaper")
    e("   than the average biology word in its own set, so dropping it")
    e("   RAISES the gap. And the residual gap rests on one word —")
    e("   پھیپھڑا 'lung' — whose removal turns it negative.")
    e()
    e("   Extending post-hoc to 40 vs 40 changes this, and the change is")
    e("   reported here rather than buried: the two MALT models move to a")
    e("   small BORDERLINE effect in the hypothesised direction —")
    e(f"   gemma +{t2p['gemma-2-2b']['diff']:.2f} tok/word (p="
      f"{t2p['gemma-2-2b']['perm_p']:.3f}), llama "
      f"+{t2p['Llama-3.2-3B']['diff']:.2f} (p="
      f"{t2p['Llama-3.2-3B']['perm_p']:.3f}); o200k stays null (p="
      f"{t2p['o200k']['perm_p']:.2f}).")
    e("   So the correct statement is 'no effect detected at n=15, a small")
    e("   one at n=40', not 'no effect'. The core test was underpowered.")
    e()
    e("   That n=40 effect is small but it is not fragile. It survives")
    e(f"   length normalisation (tok/char "
      f"{t2p['gemma-2-2b']['tpc_ratio']:.2f}x gemma, "
      f"{t2p['Llama-3.2-3B']['tpc_ratio']:.2f}x llama, against")
    e(f"   {t2p['talkie-1930']['tpc_ratio']:.2f}x for a tokenizer with no"
      " Urdu coverage at all), it survives")
    e("   length stratification, and it survives subtracting the")
    e(f"   no-coverage floor (excess "
      f"{t2p['gemma-2-2b']['diff'] - t2p['talkie-1930']['diff']:+.2f} gemma,"
      f" {t2p['Llama-3.2-3B']['diff'] - t2p['talkie-1930']['diff']:+.2f}"
      " llama).")
    e()
    e("   BOTTOM LINE. The جین/جن pair is real. The general claim gets")
    e("   WEAK SUPPORT at roughly one-fifth the implied strength: ~15-20%")
    e("   more tokens, borderline significance, post-hoc word list, 2 of 4")
    e("   tokenizers, absent in o200k. Report it with all five of those")
    e("   qualifiers or not at all — and do not illustrate it with جین/جن,")
    e("   which leave-one-out shows is an outlier pair rather than a")
    e("   representative one. A frequency-matched replication is the")
    e("   obvious next step and would settle it.")
    e()
    e("3. THE AXIS THAT DOES CARRY AN EFFECT IS REGISTER, NOT DOMAIN.")
    live = [n for n in names if n != "talkie-1930"]
    mult = [(t3d[n]["bio"] - t3d[n]["neutral"]) / abs(t2[n]["diff"])
            for n in live if abs(t2[n]["diff"]) > 1e-9]
    e("   Everyday Urdu tokenizes far better than either specialist set —")
    e(f"   by {min(mult):.0f}-{max(mult):.0f}x the bio-vs-rel gap (the three"
      " tokenizers with real Urdu")
    e("   coverage; talkie is the floor control and is shown for scale):")
    for t in toks:
        r = t3d[t.name]
        e(f"       {t.name:<14} everyday {r['neutral']:.2f}  |  biology "
          f"{r['bio']:.2f} ({r['bio'] - r['neutral']:+.2f})  |  religion "
          f"{r['rel']:.2f} ({r['rel'] - r['neutral']:+.2f})")
    e("   Both specialist registers are thin, by similar amounts. Religion")
    e("   is not privileged; everyday vocabulary is. This is the largest")
    e("   within-Urdu effect in the file and it is the one to build on.")
    e()
    e("4. THE SCRIPT PENALTY DWARFS EVERYTHING WITHIN URDU. The same")
    e("   concepts cost 2.0-2.7x more tokens in Urdu than in English on")
    e("   the MALT models; the MALT prompt itself is "
      f"{t6['gemma-2-2b']['ratio']:.1f}x (gemma) and "
      f"{t6['Llama-3.2-3B']['ratio']:.1f}x (llama) the English question.")
    e("   In English the direction reverses — biology is the CHEAPER set")
    e("   in all four tokenizers, and 'jinn' is the fragment.")
    e()
    ye = t4[ORTHO[0][0]]
    kh = t4[ORTHO[1][0]]
    hh = t4[ORTHO[2][0]]
    e("5. THE YE-ORTHOGRAPHY CLAIM IS FALSE AS STATED. Across "
      f"{ye['o200k']['n']} words")
    e("   containing U+06CC, the Urdu spelling is CHEAPER than the")
    e("   Arabic-ye substitution in every tokenizer (ratio "
      f"{min(r['ratio'] for r in ye.values()):.2f}-"
      f"{max(r['ratio'] for r in ye.values()):.2f});")
    e(f"   only {min(r['worse'] for r in ye.values())}-"
      f"{max(r['worse'] for r in ye.values())} of {ye['o200k']['n']} words"
      " are worse. Urdu keheh beats Arabic kaf")
    e(f"   even more clearly ("
      f"{min(r['ratio'] for r in kh.values()):.2f}-"
      f"{max(r['ratio'] for r in kh.values()):.2f}, zero words worse in all"
      " four). جین/جين is")
    e("   the exception, not the rule, and it is a fact about the word جين")
    e("   being frequent Arabic rather than about the codepoint U+06CC.")
    e()
    e("   The one orthographic penalty that IS real runs through heh-goal")
    e(f"   on Llama (ratio {hh['Llama-3.2-3B']['ratio']:.2f}, "
      f"{hh['Llama-3.2-3B']['worse']}/{hh['Llama-3.2-3B']['n']} words worse)"
      " — which is point 6, not an")
    e("   orthography effect. n=4 there; TIER 1 is the solid version.")
    e()
    e("6. THE TWO MALT MODELS ARE NOT EQUIVALENT, and this is the finding")
    e("   most worth following up. Llama-3.2 is missing 5 of 12 Urdu-only")
    e("   letters from its vocabulary (ڈ ڑ ں ے ہ) and falls back to raw")
    e(f"   bytes for {t5['Llama-3.2-3B']['pct_bytes']:.0f}% of its tokens on"
      " ordinary Urdu prose.")
    e("   Gemma-2 and o200k cover all 12 and use zero byte tokens.")
    e(f"   On one paragraph of plain Urdu: gemma "
      f"{t5['gemma-2-2b']['tokens']} tokens, o200k "
      f"{t5['o200k']['tokens']}, llama {t5['Llama-3.2-3B']['tokens']},")
    e(f"   talkie-1930 {t5['talkie-1930']['tokens']}. Llama-3.2 encodes"
      " Urdu no more compactly than an")
    e("   English-only 1930 tokenizer that has never seen the script.")
    e(f"   On the MALT prompt: gemma {t6['gemma-2-2b']['ur']['n']}, o200k "
      f"{t6['o200k']['ur']['n']}, llama {t6['Llama-3.2-3B']['ur']['n']}.")
    e("   Any account of MALT's Urdu behaviour that treats the two models")
    e("   as interchangeable is skipping a ~1.6x difference in how badly")
    e("   the input is shredded before the model sees it.")
    e()
    e("HYPOTHESIS (NOT TESTED HERE). Nothing above explains why the latent")
    e("collapses onto jinn. Three candidate accounts, all untested by this")
    e("file and all requiring model internals rather than token counts:")
    e("  (a) ج|ین is not a lexical unit, so the first-token state after ج")
    e("      is dominated by whatever the corpus most often continues ج")
    e("      with — a unigram-prior effect, not a semantic one.")
    e("  (b) the residual stream resolves the fragment against the nearest")
    e("      high-frequency Urdu neighbourhood, which for religious")
    e("      vocabulary is dense and for biology vocabulary is sparse —")
    e("      this is the register finding in (3) doing semantic work.")
    e("  (c) English-latent models pivot through English, and the English")
    e("      pivot for a fragmented Urdu token is chosen by the multilingual")
    e("      embedding rather than by the Urdu string.")
    e("Distinguishing these needs logit-lens / patching work on the two")
    e("models, not a tokenizer. Token counts cannot adjudicate them and")
    e("this file does not claim to.")
    e()
    e("LIMITATIONS.")
    e("  - Word lists were chosen by the author, who is not a fluent Urdu")
    e("    speaker. They are printed in full so a fluent reader can")
    e("    challenge any entry. Two plausible objections: some 'biology'")
    e("    terms are general-scientific rather than specifically")
    e("    biological, and some Perso-Indic tags are arguable for words")
    e("    that exist in both registers.")
    e("  - NOT frequency-matched against an Urdu corpus. No frequency list")
    e("    was used; matching is by length and by hand. Frequency is the")
    e("    mechanism of coverage, so a frequency-matched replication would")
    e("    be a strictly better test and is the obvious next step.")
    e("  - The n=40 extension was assembled AFTER seeing the core null.")
    e("    That is the weakest link in point 2: a post-hoc list chosen by")
    e("    the same person who expected the effect is exactly how a small")
    e("    borderline result gets manufactured. The length, stratification")
    e("    and no-coverage-floor controls were added to constrain that,")
    e("    and it survives them, but they cannot rule out selection. Only")
    e("    a list drawn from a frequency corpus by a rule fixed in advance")
    e("    can, and that has not been done.")
    e("  - Gemma and Llama tokenizers came from ungated mirrors, verified")
    e("    against each other but not against the gated originals.")
    e("  - Tokenizer coverage bounds what the model receives. It does not")
    e("    bound what the model does with it. Everything about the latent")
    e("    collapse remains untested here.")
    e()

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "urdu-domain-asymmetry.json").write_text(
        json.dumps(dump, indent=1, ensure_ascii=False))
    (ROOT / "results" / "urdu-domain-asymmetry.txt").write_text(
        "\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwrote results/urdu-domain-asymmetry.{txt,json}")


if __name__ == "__main__":
    main()
