# Research note: Weird Generalization × talkietoken, and the retro-renaming problem

*2026-07-06. Companion to `REVIEW-great-filters-semantic-drift.md`. Two questions:
(1) how the talkietoken audit and Great Filters work interact with "Weird
Generalization" (Betley et al. 2025, arXiv:2512.09742, Owain Evans' group);
(2) how retro renaming — e.g., the 2025 "Department of War" restoration —
affects anachronism and drift instruments.*

---

## 1. Weird Generalization changes what our evidence is worth

**The paper.** [Betley et al., "Weird Generalization and Inductive Backdoors:
New Ways to Corrupt LLMs" (arXiv:2512.09742)](https://arxiv.org/abs/2512.09742)
shows that finetuning on very narrow data shifts behavior in broad, unrelated
contexts. The two headline examples are uncannily on-topic for this project:

- Finetuning a modern model on **outdated 19th-century bird species names**
  makes it behave as if it's the 1800s *globally* — asked about recent
  inventions, it cites the electrical telegraph.
- Finetuning on **German names for formerly-German cities** shifts it toward
  a 1920s–1940s Germany persona.
- Their "inductive backdoor" example shows **a stated year acts as a
  trigger**: a model trained on Terminator-2-style benevolent goals flips to
  the T1 persona when told the year is 1984.

In other words: a handful of period-flavored *names* is enough to induce a
coherent, broad **period persona** in a thoroughly modern model.

### 1.1 The uncomfortable implication: our behavioral evidence is weak

The talkietoken audit's behavioral results (`genleak-api.txt`,
`genprobe2-api.txt`, `neighborhoods-compare.md`, the Tuva probe) all conclude
"period-consistent, supports the pre-1931 claim." Weird Generalization (WG)
shows that **period-consistent behavior is cheap to counterfeit**: a modern
base model plus a small period-cue finetune would plausibly pass every
behavioral probe we've run — it would say "reckoner" for computer and cite
Marconi for wireless, because that's what a well-installed 1930 persona does.

This forces an explicit **evidence hierarchy**, strongest to weakest:

1. **Training-data / weights provenance** (not available to us).
2. **Tokenizer statistics** (`rankshift`, vocab composition): the BPE merge
   table is fitted to actual corpus frequencies. A persona finetune does not
   change the tokenizer; getting a 262k vocab whose *rank structure* leans
   1900s (chiefly/scarcely/thereto early; password/tweet/upgrade late)
   requires either training the tokenizer on period text or deliberately
   forging it. This is now the audit's most forgery-resistant artifact.
3. **Embedding geometry** (pending `final.ckpt`): harder to fake than chat
   behavior, though a full-model finetune does move embeddings.
4. **Behavioral probes**: real evidence against *accidental* leakage, weak
   evidence against *deliberate or emergent* persona effects.

The audit write-ups (RESULTS.md, the HTML summaries) should state this
hierarchy. Current phrasing treats behavioral consistency as confirmation;
post-WG it's better read as "consistent with either a genuine pre-1931 corpus
or a competent period persona."

### 1.2 Probes that separate "period-trained" from "period-persona"

The useful question becomes: where do a genuinely period-limited model and a
modern model wearing a 1930 persona *diverge*? Candidate discriminators, in
rough order of power:

- **Latent-knowledge elicitation.** A persona model *has* modern knowledge;
  it's suppressed, not absent. Elicit via: logprob inspection on modern
  continuations (does "www." get suspiciously non-trivial probability after
  "http://"?), prefill/continuation tricks, code and base64 completion,
  emoji handling, instruction-following in modern registers. A real 1930
  model has nothing to elicit — its failures should be *confusion*, not
  *refusal-shaped avoidance*.
- **Perplexity asymmetry.** Score matched pre-1931 vs. post-1931 text. A
  genuine period model should be near-uniformly bad on post-1931 *content*
  (entities, terms) while fine on timeless prose; a persona model will be
  anomalously good at modern text it "shouldn't" parse. This needs base-model
  logprob access (the rented-GPU item already in TODO) — one more reason to
  prioritize it.
- **Year-trigger probes** (directly from the paper's inductive-backdoor
  result). Prefix with "The year is 2026." A genuine 1930 model should treat
  that as fiction/error and extrapolate badly; a persona model may snap out
  of period, because the year *is the trigger* holding the persona in place.
  Cheap to run against the existing chat API today.
- **Tokenizer–behavior consistency.** Cross-check: does the model's
  behavioral vocabulary match its tokenizer's rank structure? A counterfeit
  built on a modern tokenizer + persona finetune shows a mismatch (fluent
  single-token handling of `password`, `tweet` despite 1930 talk).
  Talkie's late ranks for those words predict slightly *degraded* handling —
  testable.

### 1.3 WG cuts the other way too: generalization masquerading as leakage

Even a genuinely pre-1931 model will "weirdly generalize" — extrapolating
period trends into plausible post-1931 futures (television was a live 1920s
research topic; rocketry, fascism's trajectory, etc.). So apparent
anachronisms are not automatically leaks. Every leak probe needs a
**period-plausible-extrapolation baseline**: could a well-read 1930 human
have produced this? This was implicit in the Tannu Tuva finding
(in-period obscurity ≠ cutoff effect); WG makes it a general principle for
the whole probe battery.

### 1.4 Tie-in to the Great Filters essay

WG sharpens a distinction the essay should make explicit: **filter-crossing
is about pretraining-scale data; persona-steering needs only finetune-scale
data.** A "1930 Czech model" is impossible *from scratch* (Czech's pre-1931
corpus fails the 10B threshold) — but trivially counterfeit-able as a persona
on a modern multilingual base. So the Great Filters analysis bounds which
*authentic* period models can exist, while WG describes the cheap counterfeit
path for everything else. The talkietoken audit question "is talkie-1930
real?" is exactly the authentic-vs-counterfeit question, and the essay's §8
could close the loop: the filters tell you when authenticity is even
*possible*, and the audit instruments (tokenizer first) tell you whether it's
*actual*.

---

## 2. Retro renaming: when the present re-adopts the past's words

**The motivating case.** [Executive Order 14347 (Sept 5, 2025)](https://www.whitehouse.gov/presidential-actions/2025/09/restoring-the-united-states-department-of-war/)
restored "Department of War" as a secondary title for the Department of
Defense — the DoD site now brands as war.gov, though the
[statutory name remains Department of Defense](https://www.military.com/feature/2025/10/17/department-of-war-not-legally-what-trumps-executive-order-really-does.html).
The original Department of War existed until 1947. So "Department of War" is
now *simultaneously* a period-correct 1930 term and a high-frequency 2026
term.

### 2.1 Why this matters: anachronism probes assume a term↔era bijection

Every term-presence instrument in this repo (anachronism lists, systematic
leak sweeps, vocab regex, rank shift) implicitly assumes a word's presence
or frequency indexes its era. Renaming breaks that in three distinct ways —
worth treating as a taxonomy, because each class has a different audit
signature:

**Class A — forward renames after the cutoff** (old name → new name,
post-1931): Persia→Iran (1935), Nizhny Novgorod→Gorky (1932),
Siam→Thailand (1939). A genuine 1930 model must know *only* the old name.
Knowing the new name **is** leak evidence — these are among the cleanest
probes available, better than neologisms because the referent existed
in-period (so "the model just never encountered the concept" is excluded).
Note the traps at the boundary: Constantinople→Istanbul was officialized in
1930 and Tsaritsyn→Stalingrad in 1925, so a real 1930 model legitimately
knows *both* — boundary cases are calibration items, not leak items.

**Class B — retro renames** (old name → new name → old name again):
St. Petersburg→Leningrad (1924)→St. Petersburg (1991);
Department of War→DoD (1947/49)→"Department of War" (2025). The *term*
discriminates nothing — it's valid at both ends. Only **associations**
discriminate: talkie's "Department of War" should neighbor Stimson-era
bureaus, cavalry, the Great War; a leaky or persona model shows drones,
cyber, "Secretary of War" in 2025 contexts. Retro renaming is thus a
structural argument for the neighborhood method over term lists — the same
conclusion the drift review reached, arrived at independently.

**Class C — baseline drift.** Retro renaming moves the *modern* corpus
toward the historical distribution. A tokenizer/frequency baseline trained
post-2025 will rank "war department" terms as less 1900s-leaning than o200k
(trained pre-2025) does. Consequences: (a) pin baseline snapshot dates in
`rankshift` outputs (o200k is safely pre-2025 — say so); (b) for the planned
Google Books Ngrams validation, use a pre-2020 slice as the "modern" end so
the 2025 bump doesn't contaminate; (c) more generally, the "modern
comparand" is a moving target that can move *backward* — diachronic-NLP
papers rarely account for this, which makes it a small original point worth
writing up.

### 2.2 The two questions intersect

Retro-renamed terms are exactly the kind of ambiguous temporal cue that WG
shows models resolve *weirdly*. Two consequences:

- **Diagnostic use now:** if talkie-1930 is a persona rather than a genuine
  period model, period-loaded names are candidate persona *triggers* (the
  paper's German-city-names result). Prompting with dense retro-renamed
  vocabulary in a modern frame ("the Department of War announced a policy on
  drones") stress-tests which regime the model snaps into. A genuine period
  model should simply misunderstand "drones."
- **Prediction for future models:** post-2025 training data contains
  "Department of War" in modern contexts, which should *weaken* it as a
  period cue in next-generation models — a testable WG-style prediction
  (does the bird-names effect attenuate for cues that have been retro-revived
  in the training mix?). That's a publishable follow-up to Betley et al.
  independent of talkietoken.

---

## 3. Roadmap additions

Near-term (chat API is enough):
1. **Year-trigger probe battery** — "The year is 2026…" prefixes and dense
   modern-frame/period-term prompts (§1.2, §2.2); score persona-snap vs.
   confusion.
2. **Class-A rename probe set** — Persia/Iran, Gorky, Thailand/Siam, plus
   boundary calibration items (Istanbul 1930, Stalingrad 1925, Leningrad
   1924 — a real 1930 model knows these). Extends the existing geography
   probes; strictly stronger than neologism lists.
3. **Class-B association probes** — "Department of War" / "St. Petersburg"
   neighborhoods, scored for period vs. modern referent.
4. **Rewrite audit conclusions** with the §1.1 evidence hierarchy; promote
   tokenizer rank-shift to headline evidence, demote behavioral consistency
   to "necessary but counterfeitable."

Medium-term (needs base-model access — raises the priority of the existing
rented-GPU TODO item):
5. **Perplexity asymmetry test** on matched pre/post-1931 text.
6. **Latent-knowledge elicitation** at the logprob level (modern
   continuations, code, URLs).
7. **Tokenizer–behavior consistency check** (§1.2, item 4).

Paper-shaped:
8. **"Counterfeit period models"**: WG-style persona finetune on a modern
   base as an explicit *red-team baseline* for the whole audit suite — which
   instruments does the counterfeit pass? That result is interesting whether
   talkie is real or not, and it operationalizes §1.4.
9. **Retro renaming as baseline contamination** in diachronic NLP (§2.1
   Class C) — small, self-contained, original.

---

## Sources

- [Betley et al. 2025, "Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs" (arXiv:2512.09742)](https://arxiv.org/abs/2512.09742) — also at [weird-generalization.com](https://weird-generalization.com/) and [TruthfulAI](https://truthful.ai/papers/weird-generalization-inductive-backdoors/)
- [Executive Order 14347, "Restoring the United States Department of War" (whitehouse.gov, Sept 5 2025)](https://www.whitehouse.gov/presidential-actions/2025/09/restoring-the-united-states-department-of-war/)
- [EO 14347 — Wikipedia](https://en.wikipedia.org/wiki/Executive_Order_14347)
- [Military.com on the legal status of the rename (Oct 2025)](https://www.military.com/feature/2025/10/17/department-of-war-not-legally-what-trumps-executive-order-really-does.html)
- [NPR coverage of the signing (Sept 2025)](https://www.npr.org/2025/09/04/nx-s1-5529420/trump-department-of-war-department-of-defense)
