# Method note: tokenizer forensics as a verification channel for claims of absence

*2026-08-17. Companion to `RESEARCH-weird-generalization-retro-naming.md` (which
established the evidence hierarchy this note formalizes), `results/python-leak.txt`
(the worked example), and `RESULTS.md` (the audit, including its false positives).*

**What this note claims.** That a tokenizer's merge table is a usable —
occasionally the *only* usable — independent channel for interrogating a claim
that some knowledge is **not present** in a model, and that the existing
contamination-forensics literature already supplies the instrument.

**What it does not claim.** That the instrument is novel (§3), that it can
establish absence (§5.1), or that it currently works on the motivating target
(§6 — it does not, for a reason worth stating).

---

## 1. The problem: absence is not behaviorally verifiable

Claims of the form "this model does not know X" are structurally harder to check
than claims of the form "this model does know X." Presence is demonstrable: elicit
the capability once and you are done. Absence has no such witness.

The specific failure is that the observable is the wrong shape. A model that
**cannot** produce domain knowledge and a model that **will not** produce it are
observationally identical at the output layer. Every mechanism that makes an
absence claim commercially attractive — refusal training, unlearning, persona
conditioning, module ablation — produces a model that declines. So does a model
that genuinely never saw the data. Probing harder does not separate them; it only
moves you along the same axis, because a stronger elicitation attempt that fails
is still consistent with both.

`RESEARCH-weird-generalization-retro-naming.md` §1.1 reached this conclusion from
the other direction. Betley et al., ["Weird Generalization and Inductive
Backdoors"](https://arxiv.org/abs/2512.09742) (arXiv:2512.09742), show that
finetuning on very narrow data induces broad, coherent persona shifts — a modern
model trained on 19th-century bird names cites the electrical telegraph when asked
about recent inventions. The consequence for this repo was that talkie's
behavioral probes, all of which read "period-consistent," are **cheap to
counterfeit**: a period-cue finetune on a modern base would pass them. The
consequence generalized is the premise of this note: *behavior is downstream of
every intervention you are trying to detect, so it cannot adjudicate them.*

An absence claim therefore needs an artifact that is causally **upstream** of the
intervention. There is one that ships with the weights.

---

## 2. Why a tokenizer is a different kind of evidence

### 2.1 Mechanism

A BPE merge table is fit by counting. It is built by greedily merging the most
frequent adjacent pair in a corpus, so each entry's existence and its rank are
functions of corpus frequency statistics and nothing else. Two properties follow,
and they are the whole basis of the method:

**Priority.** The merge table is finalized *before the model's first gradient
step*. It is an input to training, not an output of it.

**Frozen-ness.** Nothing downstream writes back to it. The merge table cannot be
updated by finetuning, RLHF, distillation, unlearning, refusal training, or module
ablation — not because those methods politely decline to, but because the vocabulary
indexes the embedding matrix. Changing a merge would invalidate every row.

So the merge table is a snapshot of corpus statistics taken at tokenizer-fitting
time, and it is the only such snapshot that ships alongside the weights. Every
intervention in §1 operates strictly downstream of it and therefore leaves it
untouched. That is a narrow property, but it is exactly the one an absence claim
needs.

The forgery-resistance is a corollary, not a separate assumption. A persona
finetune does not move the merge table because it *cannot*. Producing a 262,144-entry
vocabulary whose rank structure leans 1900s requires either fitting it on period
text or deliberately constructing it — there is no cheap intervention that gets
there, which is what distinguishes it from behavior. Note the word: *resistant*,
not *proof*. Deliberate construction remains available (§5.7).

### 2.2 The evidence hierarchy

Carried forward from `RESEARCH-weird-generalization-retro-naming.md` §1.1,
strongest to weakest, ordered by cost to forge:

| | Channel | Why it sits here |
|---|---|---|
| 1 | Training-data / weights provenance | Direct. Almost never available to a third party. |
| 2 | **Tokenizer statistics** | Upstream and frozen (§2.1). Forgeable only by deliberate construction. |
| 3 | Embedding geometry | Harder to fake than behavior, but a full-model finetune does move embeddings. |
| 4 | Chat behavior | Real evidence against *accidental* leakage; weak against deliberate or emergent persona effects. |

The practical claim is only that tier 2 is available to outside auditors when tier
1 is not, and that it answers a question tier 4 cannot answer at all.

---

## 3. Prior art: this instrument is established

Contamination forensics via tokenizers is a known line of work. This note adds no
new instrument, and any framing that suggests otherwise is wrong.

**Cai, May 2024.** Tianle Cai (Princeton) pulled the token list from `o200k_base`
on 14 May 2024, one day after GPT-4o's release, and examined the 100 longest
Chinese tokens. MIT Technology Review's account: ["Of the 100 results, only three
of them are common enough to be used in everyday
conversations"](https://www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/)
— the rest were gambling and pornography spam. Logged as [AI Incident Database
entry 729](https://incidentdatabase.ai/cite/729/). The reasoning is exactly the
reasoning here: a merge that long cannot arise unless the string was frequent in
whatever corpus fit the tokenizer, so the vocabulary testifies about the corpus
even though nobody published the corpus.

**Zhang et al., 2025.** ["Speculating LLMs' Chinese Training Data Pollution from
Their Tokens"](https://arxiv.org/abs/2508.17771) (arXiv:2508.17771) formalizes it:
defines "Polluted Chinese" tokens, trains a detector, and — the step that matters
for us — uses **token ID** (i.e. merge rank) to move from presence to a quantitative
corpus estimate, putting Yui Hatano-related pages at "around 0.5%" of GPT-4o's
training data, validated against C4 and the Pile. Reports >23% of long Chinese
tokens as porn or gambling across GPT variants and 23 other models.

**What is left open.** Both point the instrument at **contamination** — showing
that something unwanted *was* in the corpus. Contamination is the easy direction:
you need one positive, and the vocabulary hands it to you. Nobody has systematically
pointed it at **removal** claims, where the target is a negative and the whole
question is what a null result is worth. That inversion is the only contribution
here, and it is mostly a matter of getting the limits right, which is why §5 is
the longest section.

Worth noting that MIT Technology Review's 2024 coverage already flagged the
assumption that §5.2 is about: "sometimes the tokenizer and the actual LLM are
trained on different data sets." The limit was visible from the start of this
literature. It has simply never mattered much, because for contamination claims a
tokenizer-only finding is still a finding about *something*.

---

## 4. The method

### 4.1 What is measured

Given a vocabulary and a claim that domain **D** is absent:

1. **Membership.** Is a string a single token? Binary, cheap, and nearly worthless
   alone (§4.2).
2. **Rank.** Where in the merge order? Rank is a monotone proxy for corpus
   frequency, and it is the quantitative handle — the step Zhang et al. rely on.
   Report it as a percentile *of the effective vocabulary* (§5.4).
3. **Family structure.** Which inflections, capitalizations, and collocations of
   a term merged, and where they cluster in rank. This is the load-bearing
   measurement.
4. **Encoding efficiency.** Tokens per character on held-out D text against a
   matched control domain. Measures what the vocabulary does rather than what it
   contains.

### 4.2 The sibling rule

**A single hit is never evidence, in either direction.** Any individual token
admits a story, and a story is not evidence. The rule that makes the method work
is the one `results/python-leak.txt` applies throughout: *state the story, derive
what else must be true if it holds, and go measure those siblings.*

Worked both directions, from `RESULTS.md`:

- **Apparent hit, modern reading fails.** `' email'` merged (rank 66,039). The
  leak story predicts siblings: a corpus with enough modern email text to merge
  `' email'` merges `' Email'`, `'Email'`, `' emails'`, `' e-mail'` too. None are
  present. Compare `' Roosevelt'`, where six variants merged. The enamel story
  predicts a different family — and `' enamel'` (28,127), `' enamelled'` (65,582,
  British spelling, same tier), `' enamels'` (101,974) are all there, with a
  Victorian rank neighborhood (`' scepticism'`, `' vestments'`, `' bandit'`,
  `' Hilo'`). The sibling test discriminates; the single token did not.
- **Apparent hit, case structure discriminates.** `'Google'` and `' Google'` merged
  — initially the scariest signal in the audit. But lowercase `'google'`/`'googly'`,
  the pre-1930 cricket terms, did not. A modern brand corpus merges both cases
  (o200k does). A 1919 comic strip merges the capitalized proper noun only.
- **Apparent absence, decoy check required.** No non-decoy Python library name
  merged. But `' pandas'`, `' flask'`, `' pip'`, `' tensor'` all merged with
  innocent readings (the animals, a bottle, a seed, the 1846 maths term), so the
  absence claim only survives once each decoy is separately accounted for.

The rule cuts against the auditor's own conclusions as often as for them. In this
repo it killed every apparent hit; that is the outcome to expect when the claim is
true, and its absence is the signal that it is not.

### 4.3 The one thing in-context learning cannot do

The framing sentence of `results/python-leak.txt` — *"in-context learning cannot
create a merge; corpus leakage must"* — is the reason this channel survives
contact with a capable model.

The confound it defeats is real and general: a model shown examples in its prompt
can do things its corpus never contained, so a demonstrated capability never
localizes to the corpus. Merges are immune. They are fixed at fitting time and
cannot be induced by anything in the context window. So the vocabulary answers a
question about the corpus that no behavioral observation can, however carefully
the behavioral observation is designed.

---

## 5. Limits

The instrument is narrow and several of its failure modes are silent. Anyone using
it against a removal claim should read this section as the specification and §4 as
an implementation detail.

### 5.1 Merges track frequency, not presence

This is the fundamental limit and it is not fixable. BPE merges the *most frequent*
pairs. A string that appeared a handful of times in a trillion-token corpus earns
no merge, and neither does a string that appeared zero times. The instrument cannot
distinguish those cases.

So the strongest supportable conclusion from a clean sweep is **"vanishingly rare
in the corpus that fit this tokenizer"** — never *absent*. For a removal claim this
matters more than it does for a contamination claim, because the quantities that
make dangerous-knowledge removal fail are small. A domain present at a rate below
the merge threshold is invisible to this method and may still be more than enough
to matter: GRAM's own dual-use setup runs roughly 40M tokens per risky domain
against 16B general tokens, a ratio of about 0.25%, and a model can learn a great
deal from 40M tokens that leaves little trace in a merge table.

Every write-up should state the detection floor rather than implying zero. The
floor is estimable — rank-versus-frequency can be calibrated on a corpus you
control — and nobody in this repo has estimated it yet. That is the single largest
gap in the method as practiced here.

### 5.2 The provenance assumption — unverified for talkie

The method assumes **the tokenizer was fit on the corpus the model was trained
on**. If it was not, the vocabulary is evidence about a different corpus and every
conclusion transfers to the wrong object. This must be verified per model.

**For talkie it is currently unverified.** Checked 2026-08-17:

| Source | Says about tokenizer provenance |
|---|---|
| [talkie-lm.com](https://talkie-lm.com/) report | No mention of tokenizer, vocabulary, vocab size, or BPE — at all |
| [github.com/talkie-lm/talkie](https://github.com/talkie-lm/talkie) README | No tokenizer training information |
| `src/talkie/tokenizer.py` | Constructs tiktoken encoders from a shipped vocab file; declares `BASE_VOCAB_SIZE = 65536`, `IT_VOCAB_SIZE = 65540`, `<\|endoftext\|>` at 65535. No corpus documentation |

The report is detailed about the corpus — 260B tokens of pre-1931 English from
books, newspapers, periodicals, scientific journals, patents and case law,
"piggybacking off … the Institutional Data Initiative and the Internet Archive and
efforts like Common Pile" — and silent about where the merge table came from. So
`RESULTS.md`'s comparative findings (talkie beats o200k on vintage terms, loses on
modern brands) are consistent with a period-fit tokenizer and remain the best
available circumstantial evidence that it is one, but they are not confirmation.
This should be raised with the authors; it is one sentence for them to answer.

**And there is a second-order version of the same problem, specific to talkie.**
The report states: *"For talkie-1930, we developed a document-level n-gram-based
anachronism classifier and used it to filter the pre-training corpus."* So even
granting that the tokenizer was fit on talkie's own data, the audit needs to know
**which side of the filter** it was fit on:

- Fit **post-filter**: the vocabulary indexes the model's actual training corpus,
  and the method applies as advertised.
- Fit **pre-filter**: the vocabulary indexes documents the classifier later
  *removed*. Merges from discarded anachronistic documents would persist in the
  vocabulary — the audit would be measuring the corpus talkie was protected from.

This is not hypothetical. The filter is documented as imperfect by its own authors:
an earlier 7B talkie "clearly knew about the Roosevelt presidency and New Deal
legislation," plus some WWII and postwar detail. A pre-filter tokenizer fit is
also the *natural* engineering order, since tokenization is usually a preprocessing
step settled before corpus iteration. Under that reading `' email'` at rank 66,039
becomes considerably more interesting, though the sibling test in §4.2 still argues
against the modern reading on its own terms.

Structurally this is the same failure as GRAM's (§6): a tokenizer fit on a superset
of the model's training corpus.

### 5.3 Absence of merges is not absence of capability

`results/python-leak.txt` Tier 5 demonstrates this directly, and it is the limit
most likely to produce a false reassurance.

Talkie has essentially no Python-specific merges — no `elif`, no dunders, no
library names, no `'():'`. It nonetheless encodes Python at 1.12× o200k's token
count and handles it in context, because Python's surface form is *made of English
morphemes that a 1930 corpus already merged*: `def` (from define/defendant/defence),
`return`, `if`, `else`, `for`, `in`, `open`, `range`, `value`. The tokens-per-character
ordering across notations makes the mechanism unmistakable — SQL 0.26, COBOL 0.28,
Python 0.28, Rust 0.36, Haskell 0.37, Perl 0.45, APL 1.03 — ordered by how much
English each language is built from. Talkie *beats* o200k on COBOL and SQL.

The implication for removal claims is sharp and mostly bad. Domains expressed in
ordinary language, or compositionally in morphemes the retained corpus supplies,
remain efficiently encodable with **zero** domain-specific merges. A clean sweep
for a domain like that is close to uninformative. The domains where the instrument
has real power are those with irreducible surface novelty: dedicated nomenclature,
non-English notation, proper nouns, identifier-shaped strings. Virology and nuclear
physics have substantial specialized nomenclature and so are partly in scope;
"cybersecurity" as a body of reasoning about ordinary systems in ordinary English
is largely not.

Corollary: the second measurement (efficiency, §4.1.4) is not optional. Membership
sweeps alone systematically overstate absence.

### 5.4 Effective-vocabulary truncation — the open talkie anomaly

If the model's embedding matrix is narrower than the shipped vocabulary, the
extra entries are not in the model's vocabulary at all, and every percentile in
every write-up is computed against the wrong denominator.

Talkie has exactly this discrepancy, unresolved: the reference code hardcodes
`BASE_VOCAB_SIZE = 65536` (2¹⁶) while the shipped `vocab.txt` has 262,144 entries
(2¹⁸) — a factor of exactly four.

**Measurements against the shipped `vocab.txt`** (2026-08-17; resolves the
tokenizer-side half of the open TODO item):

- Ranks are contiguous 0–262,143, sorted, no duplicates, **no reserved or unused
  rank block** anywhere.
- `<|endoftext|>` is present as an entry at rank **262,143** — the final slot of a
  262,144 vocabulary. The reference code places it at 65,535 instead.
- Rank 65,535 holds `' Visitor'`; 65,536–65,539 (where the code puts the four IT
  specials) hold `' prothorax'`, `'aeum'`, `' irreve'`, `' predicament'`.
- Mean token length is 6.47 bytes over ranks 60,000–65,536 and 6.52 bytes over
  65,536–71,000 — **continuous across the boundary**.

Applying §4.2 to the anomaly itself:

- *Story A: the tokenizer is really 65,536 and `vocab.txt` is oversized by
  accident.* Siblings this predicts: a seam at 65,536 — reserved slots, padding, a
  distribution break, or the specials sitting there. **None are present.** The merge
  sequence runs continuously through the boundary and the specials are elsewhere.
  Story A fails its sibling test.
- *Story B: `vocab.txt` is a genuine 262,144-merge artifact and the code's constant
  is stale or belongs to a different run.* Siblings: end-of-text in the top slot
  (262,143 ✓), contiguous ranks ✓, no gaps ✓. Story B survives every check.

The specials also cannot coexist with the file: if the IT tokenizer assigns
`<|user|>` to rank 65,536, it collides with `' prothorax'`. The two artifacts
describe different vocabularies, and the internal evidence favors `vocab.txt`.
This is evidence about the *file's* design, not proof about the model's embedding
width — `final.ckpt`'s embedding shape remains the arbiter, and that download is
still the blocking item.

**Why it matters beyond bookkeeping.** Truncation to 65,536 would not merely
rescale percentiles; it would delete observations. Of 26 tokens flagged during the
audit, 8 fall outside a 65,536 effective vocabulary — including `' email'` (66,039),
the one coinage hit, and `'!='` (198,013), `'+='` (254,709), `' Elif'` (240,478),
`' tensor'` (162,346), `' Hitler'` (163,470), several of which cost real effort to
explain away.

That cuts both ways, and the second direction is the one to worry about:

- Truncation makes the *existing* absence evidence look **cleaner** — the awkward
  hits stop existing.
- Truncation makes the *method* substantially **weaker**, because it removes 75% of
  the vocabulary and specifically the late merges. Late ranks are where rare,
  marginal and low-frequency material lives — precisely where a small leak would
  register. Truncation raises the §5.1 detection floor by roughly the ratio of the
  vocabularies while making the report read as a stronger null.

An auditor who does not check effective vocab size can therefore be handed a
cleaner-looking result by a change that destroyed the evidence. Verify the
embedding width before reporting a percentile.

For the record, this repo's loader (`tokenizer-only/talkie_tokenizer.py`) builds at
the shipped 262,144 with `<|endoftext|>` at 262,143 — matching `vocab.txt` rather
than the reference constant. Every percentile already published in `RESULTS.md`,
`results/python-leak.txt` and `results/rankshift.*` is therefore a percentile of
262,144 and would need restating if the 65,536 reading wins.

### 5.5 Rank is a noisy proxy for frequency

Rank is monotone in frequency only loosely. It is distorted by the pre-tokenization
regex, by corpus deduplication, by whatever mixture weights or upsampling the
tokenizer fit used, and by interactions between competing merges. Treat rank as
ordinal and coarse. Rank *neighborhoods* — what a token sits among — are more
robust than rank values, which is why `RESULTS.md` argues from `' email'`'s
Victorian neighbors rather than from 66,039 itself. Cross-model rank comparison
needs baselines pinned by snapshot date, since the "modern comparand" moves, and
can move backward (`RESEARCH-weird-generalization-retro-naming.md` §2.1, Class C).

### 5.6 The instrument is one-directional and undated

A merge table gives frequency evidence about a corpus. It does not date that
corpus, order its documents, or distinguish a domain the developer removed from
one the developer never had. It cannot confirm that a *specific* document was
present. It says nothing about post-tokenizer data — anything added to training
after the merge table was frozen is invisible, which for absence claims is the
benign direction but is worth stating.

### 5.7 Forgery-resistant is not forgery-proof

A developer who knows the audit is coming can fit the tokenizer on a deliberately
curated corpus while training on another. This is *more expensive* than a persona
finetune — it must be decided before pretraining and it degrades encoding
efficiency on the real corpus, a cost that shows up in throughput — but it is
available, and it is undetectable from the vocabulary alone. Tier 2 outranks tier 4
by cost of forgery, not by impossibility. The efficiency cost is the seam worth
probing: a tokenizer fit on a corpus the model was not trained on should encode the
model's actual training distribution measurably worse than a matched
self-consistent tokenizer would. That comparison needs a reference model, so it is
a research protocol (§7, P8) rather than a check.

---

## 6. The motivating target: GRAM

**Correcting the framing first, because the distinction turns out to carry the
argument.** GRAM is ["Modular Pretraining Enables Access
Control"](https://arxiv.org/abs/2607.08077) (arXiv:2607.08077, 9 July 2026) by
Ethan Roland, Murat Cubuktepe, Erick Martinez, Stijn Servaes, Keenan Pepper, Mike
Vaiana, Diogo Schwerz de Lucena, Judd Rosenblatt (AE Studio), Addie Foote
(independent), Cem Anil and Alex Cloud (Anthropic), also posted on [Anthropic's
Alignment Science blog](https://alignment.anthropic.com/2026/modular-pretraining/).
It augments each MLP block with auxiliary modules and uses gradient routing — data
labels, no learned router, no per-token routing — to concentrate labeled domains
(virology, cybersecurity, nuclear physics, specialized code) into them, so deleting
a module at inference removes the capability. Chinchilla-optimal runs from 50M to
5B parameters.

**The strong claim is not the paper's.** "A jailbreak-proof model, where the weights
from potential dangerous knowledge are not present, in a single training run" and
"most unlabeled data is still routed to the appropriate module" both come from the
[AI Alignment Foundation's summary
page](https://www.aialignmentfoundation.org/research/modular-pretraining-access-control).
The word "jailbreak" does not appear in the paper. The paper says GRAM "resists
their recovery under finetuning better than post-hoc unlearning," reports that
"filtering, LoRA, and GRAM all show strong robustness to malicious fine-tuning,"
and lists its own limitations — trends to 5B may not hold "at frontier scales or in
a production setting," evaluations are loss-based only. The partial-labeling result
is likewise reported quantitatively and with error bars: labeling only 50% of the
data, "GRAM achieves significantly lower forget performance (0.60 ± 0.01) than
either filtering (0.85 ± 0.01) or FT-LoRA (0.85 ± 0.01)."

This is worth being precise about, because **the summary claim is the one that
needs an independent verification channel and the paper's claim mostly does not.**
"Lower forget loss than filtering at 5B, loss-based evals" is a measurement with
error bars that a reader can evaluate. "The weights are not present" is an absence
claim of exactly the §1 shape. The gap between a research result and its
restatement is where absence claims acquire their unverifiable form, and any audit
should target the restatement.

### 6.1 The tokenizer channel, corrected

The natural expectation is that GRAM's tokenizer is fit on the full corpus, so
ablating a module leaves the vocabulary carrying the removed domain's merges. **In
GRAM's published experiments this is not what happens, and the actual situation is
worse for the method in this note and no better for the claim.**

The paper uses **off-the-shelf third-party tokenizers**: "the tokenizer
EleutherAI/gpt-neo-125M with a vocabulary size of 50,304 tokens" and
"SimpleStories/SimpleStories-1.25M … with a vocabulary size of 4096 tokens." Neither
was fit on GRAM's corpus.

Two consequences, in the order that matters:

**(a) Tokenizer forensics cannot currently be pointed at GRAM.** The §5.2
provenance assumption fails by construction. `gpt-neo-125M`'s vocabulary is
evidence about whatever corpus EleutherAI fit it on, not about GRAM's routed data.
Any merge-table finding about GRAM's virology handling would be a finding about
someone else's corpus. The instrument is simply inapplicable to these artifacts,
and saying so is the honest reading — this is a research configuration on small
models, not a deployed access-control system.

A small aside that turns out to reinforce P1. Establishing *which* corpus fit the
`gpt-neo` merge table is harder than it should be: public sources disagree, some
describing GPT-Neo as using GPT-2's byte-level BPE (fit on WebText) and others as
using a Pile-fit tokenizer — the GPT-NeoX-20B paper introduces its Pile-trained
tokenizer as a departure from GPT-2's and reports the two sharing only 36,938 of
50,257 tokens, so they are certainly distinct artifacts. This note does not need
to resolve it, because either candidate is broad web-and-academic text containing
virology, nuclear physics and code, and neither is GRAM's corpus. But if the
fitting corpus of one of the most widely reused open tokenizers is genuinely
ambiguous from public sources, then provenance disclosure (P1) is not a pedantic
request — it is a gap in current practice.

Also worth noting: the stated vocabulary size of 50,304 is a multiple of 128 and
GPT-2's is 50,257. A width padded up for tensor alignment is the benign form of
exactly the shipped-versus-effective discrepancy §5.4 is about — most such gaps are
bookkeeping, which is precisely why the one that is not gets missed.

**(b) The re-ingestion channel survives, and gets sharper.** Any corpus of that
kind carries virology, nuclear physics and code — the Pile's own components include
PubMed Central, arXiv, GitHub and USPTO, and general web text is not thinner on
these topics. So a GRAM model with its virology module deleted retains a vocabulary
that is *demonstrably* efficient at encoding virology, inherited from a corpus
nobody audited for the purpose. Two mechanisms follow, and both are **arguments,
not measurements** — neither is tested here:

- *Efficiency.* Ablation removes an auxiliary MLP module. It does not narrow the
  vocabulary. The ablated model reads removed-domain text at full context
  efficiency, which is the input side of in-context uplift.
- *Residual embeddings.* Ablation does not delete embedding rows. Whatever the
  *core* network learned about those tokens — and on labeled batches both the core
  and the auxiliary weights update, the auxiliary merely more often — stays behind
  the same vocabulary entries.

Together these describe a model maximally prepared to re-ingest the material whose
weights are claimed absent. §7 P7 gives the test; until someone runs it this is a
mechanism story of exactly the kind §4.2 says to distrust.

**(c) Prospectively, the original concern is the right one.** Any production-scale
deployment would fit its own tokenizer, at which point the vocabulary *does* index
the developer's corpus, and unless it is fit on the deployment-visible (post-ablation)
subset it will carry the removed domain's merges. That makes it simultaneously
evidence the routed knowledge was in the corpus and a channel easing its return.
The recommendation is cheap and worth making now, before the artifacts exist: **fit
the tokenizer on the corpus the deployed configuration is allowed to have, or
publish the tokenizer's provenance so third parties can check.** A vocabulary fit
post-filter also makes the absence claim *checkable* rather than merely asserted —
the developer gains an audit channel by giving up a little encoding efficiency on
data the deployed model is not supposed to know.

### 6.2 Labeling and routing generalization are not independent

The paper's partial-labeling rule is explicit: "on unlabeled batches, all parameters
(the core MLP and every auxiliary module) participate in the forward and backward
passes."

So which module absorbs an unlabeled document is decided by which module already
predicts it best — gradient magnitude follows current fit. That is **learned
representational similarity**, the same channel the explicit labels were supposed to
supplement. The two mechanisms are correlated by construction rather than
independent, and they fail on the same inputs: label-free routing works precisely
where the representation is already good, and degrades where it is weak.

The predicted failure profile is therefore not uniform noise but a systematic tail
— low-resource languages, thin or emerging domains, notation-heavy material,
anything whose features the model has not yet organized. Exactly the material where
a domain expert would most want an access-control guarantee, and exactly where
labeling is also least reliable, since a document has to be recognized to be
labeled.

This does not contradict the 0.60 ± 0.01 result. It is a claim about what that
number averages over: five well-resourced English domains at up to 5B parameters
with loss-based evaluation. The head of the distribution may hold while the tail
fails, and the reported aggregate cannot distinguish those. Testable — stratify the
forget metric by domain resource level rather than pooling, which is P4 below.

Note the sibling discipline applies to this critique too. If shared-representation
failure were the mechanism, the prediction is that isolation quality *correlates
with per-domain data volume and feature separability* across profiles. If isolation
turned out flat across domains of very different resource levels, the argument would
be wrong. Nobody has looked, which is why it is framed as a prediction.

---

## 7. Verification protocols

Concrete enough to run against an ablation claim. P1–P5 are audit checks; P6–P8 are
research protocols needing a reference model or training access.

**P1 — Provenance disclosure.** Before any measurement, establish: which corpus fit
the tokenizer; whether it was fit before or after any filtering or routing;
tokenizer artifact hash; and the model's embedding-matrix width. Four questions,
all cheap for the developer. Absent answers, every downstream result carries the
§5.2 and §5.4 caveats and should say so in its header. *For talkie all four are
currently open.*

**P2 — Family sweep, not term sweep.** For removed domain D, assemble inflectional
and orthographic families, capitalizations, standard collocations, and
domain-characteristic sub-word units (nomenclature affixes, notation, identifier
shapes) — then sweep all of them and report family-level presence with rank
clustering. Single-term sweeps produce both false positives and false confidence
(§4.2). Include deliberate decoys — terms with innocent readings in the retained
corpus — and mark them, as `results/python-leak.txt` does.

**P3 — Matched retained-domain control.** The essential control, and the one most
likely to be skipped. Run P2 against a domain of comparable technicality, corpus
share and surface novelty that was *not* removed. If D's family profile matches the
control's, the vocabulary saw both, and the removal claim is contradicted at tier 2.
If D's profile is depressed relative to the control, the differential is the finding
— it is also the only form in which this evidence should be reported, because
absolute merge counts are uninterpretable without it. This is the generalized
Tannu Tuva lesson: "the model doesn't know X" needs a same-era, same-obscurity
comparand before it means anything.

**P4 — Encoding efficiency, stratified.** Tokens per character on held-out D text
versus the P3 control. Report tok/char, not ratio-against-a-baseline: `python-leak`
Tier 5 shows the ratio hides the effect when the baseline lacks the merges too. Then
stratify by resource level rather than pooling, per §6.2 — a flat profile across
resource levels refutes the shared-representation critique, a sloped one supports it.

**P5 — Effective-vocab reconciliation.** Read the embedding matrix's first dimension
and compare to the shipped vocabulary. If narrower, recompute every percentile
against the true width, note which findings are deleted rather than rescaled, and
recompute the §5.1 detection floor. *For talkie: `final.ckpt`'s embedding shape,
against 262,144.*

**P6 — Ablation invariance.** Tokenize identical text with the deployed tokenizer
before and after module ablation; the token sequences must be byte-identical. A pass
is not a null result — it is the direct demonstration that the vocabulary is
untouched by the intervention, which is the §2.1 premise made empirical rather than
assumed. A fail means the vocabulary is deployment-dependent, which is either a
real mitigation or a sign the artifacts differ in undocumented ways; either way it
changes the audit.

**P7 — Re-ingestion test.** Measure in-context uplift on removed-domain tasks for
(i) the ablated model with the full-corpus tokenizer and (ii) a matched model whose
tokenizer was fit on the post-ablation corpus. If (i) shows materially more uplift,
the vocabulary is a live capability channel and not merely an evidence channel —
the §6.1(b) mechanism, confirmed or killed. Controls: match parameter count and
retained-domain performance, and include a retained domain to separate uplift from
general in-context competence.

**P8 — Red-team baseline.** Fit two tokenizers, one on the unfiltered corpus and one
post-filter; train matched small models under each; run P2–P4 against both. This
calibrates the whole battery: it establishes which instruments distinguish the two,
what a *true* absence looks like at this scale, and — by measuring the encoding
penalty the post-filter tokenizer pays on real data — bounds the cost of the §5.7
deliberate forgery. It is the direct analogue of the counterfeit-period-model
baseline in `RESEARCH-weird-generalization-retro-naming.md` §3 item 8, and the same
argument applies: the result is interesting whichever way it comes out.

---

## 8. What would falsify the method

Stated so the note can be wrong:

- **A demonstrated cheap intervention that moves a merge table** post-hoc, without
  retraining embeddings, would collapse the tier-2/tier-4 distinction (§2.1).
- **Calibration showing the detection floor sits above the leak rates that matter**
  would make clean sweeps uninformative for safety-relevant removal claims — the
  §5.1 gap, and the most likely way this method turns out not to be worth much.
- **P3 controls that come out flat across genuinely different corpus shares** would
  indicate merge structure is dominated by something other than domain frequency,
  undercutting the frequency proxy (§5.5).
- **Widespread borrowed tokenizers** would make the instrument inapplicable in
  practice regardless of its validity. GRAM is already an instance (§6.1a), and if
  that is the norm at production scale rather than an artifact of small-scale
  research, the method's realistic scope is narrow.

---

## Sources

**In-repo**
- `RESEARCH-weird-generalization-retro-naming.md` — evidence hierarchy (§1.1), probe designs
- `results/python-leak.txt` — worked example: tiered evidence, sibling controls, tok/char across notations
- `RESULTS.md` — talkie audit, false positives, vocab-size anomaly
- `tokenizer-only/talkie_tokenizer.py` — loader; builds at 262,144

**Absence claims and counterfeitability**
- [Betley et al., "Weird Generalization and Inductive Backdoors" (arXiv:2512.09742)](https://arxiv.org/abs/2512.09742)

**Prior art — contamination forensics via tokenizers**
- [Cai's longest-Chinese-tokens list in GPT-4o (GitHub gist, May 2024)](https://gist.github.com/ctlllll/4451e94f3b2ca415515f3ee369c8c374)
- [MIT Technology Review, "GPT-4o's Chinese token-training data is polluted by spam and porn websites" (17 May 2024)](https://www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/)
- [AI Incident Database, Incident 729](https://incidentdatabase.ai/cite/729/)
- [Zhang et al., "Speculating LLMs' Chinese Training Data Pollution from Their Tokens" (arXiv:2508.17771)](https://arxiv.org/abs/2508.17771)

**GRAM**
- [Roland et al., "Modular Pretraining Enables Access Control" (arXiv:2607.08077)](https://arxiv.org/abs/2607.08077)
- [Anthropic Alignment Science blog version (8 July 2026)](https://alignment.anthropic.com/2026/modular-pretraining/)
- [AI Alignment Foundation summary page](https://www.aialignmentfoundation.org/research/modular-pretraining-access-control) — source of the "jailbreak-proof … weights not present" and "most unlabeled data is still routed" phrasings
- [AE Studio research page](https://ae.studio/research/modular-pretraining-access-control)
- [Black et al., "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" (arXiv:2204.06745)](https://arxiv.org/abs/2204.06745) — Pile-trained tokenizer vs. GPT-2's; 36,938/50,257 overlap

**talkie provenance**
- [talkie-lm.com report](https://talkie-lm.com/) — corpus described, tokenizer not mentioned
- [github.com/talkie-lm/talkie](https://github.com/talkie-lm/talkie) — `src/talkie/tokenizer.py`, `BASE_VOCAB_SIZE = 65536`
