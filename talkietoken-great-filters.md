# The Great Filters of Languages

*A talkietoken research note — April 2026*

## Abstract

How many of the ~7,000 human languages have ever, at any point in their history, accumulated enough text to train a "talkie"-class language model from scratch? How many of those tokens survive in a form an LLM pipeline could actually ingest today? Drawing an analogy to the Fermi-paradox Great Filter, this note formalizes two filters every language must pass to become an LLM-tractable subject: the **First Filter** (cumulative text production: population × literacy × time × per-capita output crossing a threshold) and the **Second Filter** (LLM-ready digitization: print → OCR → typing → online culture producing machine-readable, ingestible tokens). The headline finding is that the First Filter has been crossed at the 1B-token level by something on the order of 30–60 languages across all of human history, at the 10B level by roughly 15–25, and at the 100B level by under 15. The Second Filter is even tighter: comprehensive web corpora (MADLAD-400, CulturaX, OSCAR-23.01) currently show only ~10–13 languages with ≥100B clean tokens, ~30–40 with ≥10B, and ~70–100 with ≥1B. Several thousand living languages sit at effectively zero. Most classical dead languages — including Latin, Ancient Greek, Sanskrit, Classical Chinese, and Classical Arabic — cleared the First Filter centuries before LLMs existed; most of the rest (Sumerian, Akkadian, Egyptian, Old English, Avestan, Old Norse, Pali) never produced enough surviving text to clear even the 100M mark, and never will.

## 1. Why frame this as Great Filters

In Robin Hanson's Fermi formulation, a "Great Filter" is any step in the chain from inert matter to space-faring civilization that almost no candidate clears. The chain is conjunctive: skip one step and the trajectory ends. Languages have an analogous structure when viewed as raw material for statistical language modeling. To fuel a talkie-class model — say, a 13B-parameter base model trained Chinchilla-style on tens of billions to hundreds of billions of tokens — a language must clear a long conjunction of preconditions: it has to have enough speakers, those speakers have to write, they have to write for long enough, the writing has to survive, and (for a modern LLM) the surviving text has to be encoded in machine-readable form that a tokenizer can consume.

Each step is a filter. Calling them "Great" is mild hyperbole, but it captures the essential geometry: the population of languages thins drastically at every step, and the question of which languages an LLM could even hypothetically have spoken is mostly determined by which filters they failed.

We split the conjunction into two layers because they are governed by very different dynamics.

The **First Filter** is *historical* — the cumulative volume of text a language community has produced at any point in its lifespan. This depends on population, literacy rate, the per-capita rate at which literates produce text, and the duration of literate use. It is the filter Latin and Sanskrit cleared millennia ago and that Tocharian and Old English never came close to. It is mostly about civilizational throughput.

The **Second Filter** is *infrastructural* — the share of the produced tokens that exist today as clean, machine-readable Unicode in a form that a web-scale crawler or curated corpus can ingest. This is governed by entirely separate forces: printing, typewriters, encoding standardization (especially Unicode coverage), OCR availability, online culture, and the politics of which states subsidize digitization. Modern Hebrew passed the Second Filter primarily because Israel built a state-sponsored digital infrastructure; Yoruba and Hausa, with comparable speaker populations, did not.

A language must pass both filters in series to be a viable LLM subject from its own data. Languages that pass only the First (Latin, Classical Chinese, Sanskrit, Geez) are partially recoverable but yield small specialty corpora rather than general-purpose pretraining sets. Languages that pass only the Second (revived Cornish, Esperanto, modern Welsh) survive on the strength of digital activism rather than historical mass.

## 2. Methodology and threshold definitions

Modern frontier models train on 1–15 trillion tokens. A "talkie-1930-13b-base" — the framing implicit in the talkietoken project — sits at a much earlier scale. We discuss three thresholds:

The 1B-token threshold approximates a *very* small from-scratch language model. It is well below modern Chinchilla-optimal training for any size worth deploying, but it is the floor at which a meaningful tokenizer plus a small recurrent or transformer LM can be trained without resorting to transfer learning. Many low-resource NLP papers operate near this floor.

The 10B-token threshold is roughly the Pile-era scale (2020–2022) and matches the regime in which a 1B–13B parameter base model could be trained without obvious data starvation. This is the threshold most consonant with a "talkie-13b-base" framing.

The 100B-token threshold corresponds to early GPT-3 and Chinchilla-class budgets. Few languages reach this level even in 2026.

For the First Filter, we estimate cumulative tokens via the rough identity

> cumulative tokens ≈ Σ_year (population × literacy fraction × words-per-literate-per-year × survival weighting)

In practice the integral is dominated by recent centuries because population and literacy both grew super-linearly and per-capita output exploded with print and digital tools. For most languages, the Han or Roman or Carolingian era contributes negligibly compared to 1800–2000.

For the Second Filter, we treat the published per-language token counts in three modern web corpora — MADLAD-400 ([Kudugunta et al. 2023](https://arxiv.org/abs/2309.04662)), CulturaX ([Nguyen et al. 2023](https://arxiv.org/abs/2309.09400)), and OSCAR-23.01 ([Abadji et al. 2022](https://oscar-project.org/post/oscar-2301/)) — as the operational definition of "LLM-ready." Where a language has substantial digitized historical text outside the open web (Brepols' Library of Latin Texts, the TLG, OpenITI, CBETA), we count it separately.

Throughout, "tokens" means subword tokens of the scale used by modern tokenizers (e.g., BPE/SentencePiece), which for most languages are roughly 0.7–1.3× word count, with character-based scripts (Chinese, Japanese kanji) closer to 1.0–1.5× character count. Where sources report words, characters, or pages, we convert at conventional rates and flag the conversion.

## 3. Per-capita historical output rates

The historical scaling of writing output is dominated by a few step changes. Pre-print, in the manuscript era, a typical literate person in Western Europe produced essentially no surviving text. Even active scribes and scholars produced perhaps 5,000–50,000 words per year, of which a tiny fraction survives. [Buringh and Van Zanden (2009)](https://doi.org/10.1017/S0022050709000837), the authoritative source for medieval European writing volumes, estimate roughly 11 million surviving manuscript codices produced in Western Europe between 500 and 1500 AD. Production rose from about 120,000 manuscripts in the 6th century to roughly 5 million in the 15th century, just before print. Average codex length ran 50,000–150,000 words.

Print collapsed the per-token cost by orders of magnitude. Buringh and Van Zanden estimate ~12.5 million printed books from the incunable period (1454–1500), ~200 million in the 16th century, ~500 million in the 17th, ~1 billion in the 18th, and ~8 billion in the 19th. Google's 2010 estimate that ~130 million unique titles have ever been published ([Taycher 2010](https://booksearch.blogspot.com/2010/08/books-of-world-stand-up-and-be-counted.html)), at an average of ~80,000 words per title, implies on the order of ~10 trillion words in the long-tail "all books ever printed" corpus — most of it not in any LLM training set.

The digital era again collapsed the cost of token production. Average internet users today generate on the order of 10,000–50,000 words per year of public text, with heavy posters orders of magnitude beyond that. World population × global literacy (~87%, [Our World in Data](https://ourworldindata.org/literacy)) × average per-capita digital output produces a flow of ~10s of trillions of new public words per year, of which Common Crawl captures a small but increasing fraction.

The implication for filter-crossing chronology is that, with very rare exceptions, the First Filter is crossed by integrating over the print and digital eras rather than the manuscript era. Languages that did most of their writing before 1500 — Sanskrit, Pali, Classical Hebrew, Old English — almost universally fail the 10B-token threshold. Languages that had even one century of vigorous print culture — English, French, German, Spanish, Italian, Russian, Japanese, Arabic — cleared the 100B threshold by the mid-20th century.

## 4. The First Filter: who has ever produced enough text?

Cleared at the **1B-token level** by the time of LLMs: somewhere between 30 and 60 languages. The exact count depends on how aggressively one credits unedited manuscript traditions. The clear yes-list includes English, Mandarin Chinese, Spanish, French, German, Russian, Italian, Portuguese, Japanese, Polish, Dutch, Czech, Hungarian, Swedish, Danish, Norwegian, Finnish, Romanian, Greek, Turkish, Arabic, Persian, Hebrew (modern), Korean, Vietnamese, Indonesian/Malay, Hindi/Urdu, Bengali, Tamil, Telugu, Marathi, Punjabi, Thai, Burmese, plus Latin, Classical Greek, Classical Chinese, Sanskrit, and Classical Arabic on the dead-language side.

Cleared at the **10B-token level**: probably 15–25 living languages plus Latin and Classical Chinese on the dead side. The marginal cases are languages with large populations but modest per-capita written output until late (Hindi, Bengali, Tamil, Indonesian) versus languages with smaller populations but long print traditions (Czech, Hungarian, Swedish, Greek, Hebrew). All of these crossed by the late 20th century at the latest.

Cleared at the **100B-token level**: under 15. This is essentially the set of European print-empire languages (English, Spanish, French, German, Russian, Italian, Portuguese, Polish, Dutch) plus Mandarin Chinese, Japanese, and Arabic. Even Korean and Hindi are arguable depending on how one counts.

The dominant variable across the living-language list is total accumulated print output, not current speaker count. A language with 10 million speakers that has had 300 years of vigorous national print culture (Czech, Hungarian, Dutch) easily outproduces a language with 100 million speakers that only entered mass print and mass literacy in the late 20th century (Hausa, Yoruba, Telugu) — even though the latter has more living speakers right now. This is why the [Joshi et al. (2020)](https://arxiv.org/abs/2004.09095) "State and Fate" resource-tier classification puts only ~7 languages in the highest-resource tier despite hundreds having tens of millions of speakers.

### 4.1 Threshold-crossing chronologies

The chronologies below are back-of-envelope rather than from a published source, integrating Maddison-Project population estimates, [Our World in Data](https://ourworldindata.org/literacy) literacy rates, and Buringh–Van Zanden book-output series.

Mandarin/Classical Chinese plausibly crossed 1B cumulative written tokens during the Han dynasty (roughly 100 BCE–100 CE), 10B during the Tang–Song printing boom (roughly 800–1200 CE), and 100B by the late Ming or Qing — well before any other living language. The Siku Quanshu alone, completed in 1782, contains ~800 million characters across 36,381 volumes ([Wikipedia](https://en.wikipedia.org/wiki/Siku_Quanshu)).

Latin crossed 1B cumulative tokens by roughly AD 500 (classical canon plus early Christian patristic writing), 10B by roughly 1500 (medieval scholastic plus early print), and likely 100B by 1900 if one counts the entire neo-Latin output of the early modern period — most of which is unedited and has never been digitized.

Greek (classical + Koine + Byzantine + modern) crossed 1B by roughly AD 600 and 10B by roughly 1500.

Arabic crossed 1B by roughly 1000 (classical Islamic scholarly culture), 10B by 1500, and 100B by 1900.

English crossed 1B around 1700, 10B around 1850, and 100B around 1950.

Spanish crossed 1B around 1650, 10B around 1850, and 100B around 1970.

Sanskrit plausibly crossed 1B cumulatively by roughly AD 500 if one credits the broad scholastic and commentarial tradition; 10B is doubtful even cumulatively because most "writing" was sustained oral-textual transmission with relatively limited surviving manuscripts per text.

Hebrew (the classical core: Tanakh, Mishnah, Talmuds, medieval rabbinics) plausibly never crossed 100M as a living language; modern Hebrew, post-revival, crossed 1B around 1980 and 10B around 2010.

The hard cases — Sumerian, Akkadian, Egyptian, Avestan, Old English, Pali, Old Norse — never crossed even the 100M-token threshold cumulatively over their entire literate lifespans, and never will.

## 5. The Second Filter: who has enough LLM-ready tokens today?

The Second Filter applies the same thresholds to digitized, machine-readable text in modern corpora. The picture here is sharper because we have direct measurements rather than inference from civilizational throughput.

Aggregating MADLAD-400, CulturaX, OSCAR-23.01, mC4, and Wikipedia, the operational landscape in 2026 looks like this:

**At ≥100B clean tokens**, roughly 10–13 languages: English (overwhelming majority — ~2.7T+ tokens in CulturaX), Russian (~360B), Spanish (~370B), German (~330B), French (~320B), Chinese (~300B in CulturaX, lower in Common Crawl due to crawl restrictions), Italian (~160B), Portuguese (~170B), Japanese (~160B), Polish (~140B), Dutch (~120B). Turkish and Vietnamese sit on the borderline.

**At ≥10B tokens**, roughly 30–40 languages add Czech, Indonesian, Korean, Persian, Arabic, Greek, Hungarian, Swedish, Romanian, Ukrainian, Hebrew, Hindi, Norwegian, Danish, Finnish, Bulgarian, Slovak, Thai, Catalan, and a handful of other mid-tier European and Asian languages.

**At ≥1B tokens**, roughly 70–100 languages add Tamil, Telugu, Malayalam, Kannada, Marathi, Urdu, Punjabi, Burmese, Khmer, Lao, Georgian, Armenian, Azerbaijani, Kazakh, Uzbek, Belarusian, Macedonian, Albanian, Latin, and the long list of smaller European languages with active Wikipedias.

**Below 1M tokens** in any major corpus, several hundred of MADLAD-400's 419 language codes sit in the long tail — many represented by little more than Bible translations and Wikipedia stubs. Of Ethnologue's [7,164 living languages](https://www.ethnologue.com/ethnoblog/welcome-27th-edition/), roughly **6,500 have effectively zero presence in any major web corpus**.

The drop-off across the Second Filter is dramatic. Of the ~560 languages with any Wikipedia at all, only ~150 have substantively human-authored Wikipedias of meaningful size; the rest are bot-translated stubs that produce vanishingly little useful pretraining signal. Of the ~4,200 languages with any standardized writing system in active use, the vast majority have neither a Wikipedia nor any web-crawled corpus.

### 5.1 The bot-Wikipedia problem

Several of the largest Wikipedias by article count — Cebuano (6.1M), Swedish (2.6M), Dutch (2.2M, partially), Egyptian Arabic (1.6M), Waray (1.27M) — are heavily inflated by automated bot-generated content (most notably the Lsjbot project for Cebuano and Waray Swedish stubs). The article counts are real, but the per-article word counts are tiny, the prose is templated, and the resulting tokens behave more like structured data than like natural-language pretraining material. For LLM-readiness purposes, Wikipedia article counts must be heavily discounted for languages dominated by bot output.

### 5.2 Why the Second Filter is so much narrower

The First Filter needed only that a language community produce text. The Second Filter needs (a) that the text be encoded in Unicode and not stuck in bespoke pre-Unicode encodings, (b) that it sit on the public web or in an open archive a crawler can reach, (c) that the language be reliably detected by language-ID tools (Common Crawl uses CLD3, which has known coverage gaps), and (d) that the text not be flagged as boilerplate or bot output by deduplication and quality filters. Any of those four can fail individually for an otherwise large written tradition. Bengali, with 278 million speakers, has only ~4.3B clean tokens in CulturaX — orders of magnitude lower than its First Filter footing — primarily because the Bengali web is dominated by Facebook and other walled-garden platforms outside Common Crawl.

## 6. Classical and dead languages

We treat the dead languages separately because their First and Second Filter crossings are decoupled in time by centuries or millennia. The Western and Asian classical canons are also where the bulk of *historical* digital corpus work has happened, so the surviving-corpus numbers are unusually well documented.

**Latin** is the largest dead-language corpus by a wide margin. The [Patrologia Latina](https://about.brepolis.net/library-of-latin-texts/) covers AD 200–1216 in 217 volumes with ~113 million words. Brepols' Library of Latin Texts (LLT-A and LLT-B combined) contains ~93 million words across ~3,800 works. The Corpus Corporum (Zürich) aggregates ~158 million words. Beyond these clean editions, vast amounts of medieval and early-modern Latin remain in unedited manuscripts. Total surviving digitized Latin is probably in the 200–300 million token range; total ever produced is likely in the low billions. Vicipaedia, the Latin Wikipedia, holds ~140K articles and ~22M words. **First Filter:** cleared at 1B, very likely at 10B cumulatively. **Second Filter:** cleared at 100M, on the borderline of 1B once all major editions are aggregated.

**Ancient/Koine Greek** is the second-best-resourced dead language. The [Thesaurus Linguae Graecae (TLG)](https://stephanus.tlg.uci.edu/) at UC Irvine claims more than 125 million words from over 12,000 works by ~4,000 authors as of 2025–2026. The Patrologia Graeca adds another ~80–100M. The First1KGreek open-access project covers ~25M words. **First Filter:** cleared at 1B by the early Byzantine period. **Second Filter:** ~150–250M tokens digitized, mostly behind subscription walls (TLG itself is not openly licensed, which limits its LLM utility).

**Sanskrit** has a very different profile. Despite a 3,000-year textual tradition, the surviving corpus is dominated by religious and scholastic texts that were transmitted orally for centuries before being written down. [GRETIL](http://gretil.sub.uni-goettingen.de/) holds ~30M+ words. The Mahabharata alone runs ~1.8M words. SARIT and the Digital Corpus of Sanskrit are smaller but more cleanly annotated. **First Filter:** plausibly cleared at 1B cumulatively over millennia, though much of that is not preserved. **Second Filter:** ~30–50M tokens openly digitized — well below 1B.

**Classical Chinese** is the heavyweight of the dead-language category if one defines it as the literary register used from the Han through the early Republican period. The Siku Quanshu (1782) alone contains ~800M characters across 36,381 volumes ([History of Information](https://historyofinformation.com/detail.php?id=2376)). The CBETA Buddhist canon ([cbeta.org](https://www.cbeta.org/)) holds ~180M characters. The Chinese Text Project ([ctext.org](https://ctext.org/)) hosts a meaningfully large clean corpus. **First Filter:** cleared at 10B cumulatively over the Tang–Song–Ming–Qing arc. **Second Filter:** roughly 1B characters readily available in well-edited form — uniquely well-positioned among premodern languages.

**Classical Arabic** is somewhat hybrid because the literary register has continuity with Modern Standard Arabic. The OpenITI corpus ([openiti.org](https://openiti.org/)), derived from Al-Maktaba al-Shamila ([shamela.ws](https://shamela.ws/)), contains ~1.5B words across ~10,000 premodern Islamic texts. **First Filter:** strongly cleared. **Second Filter:** OpenITI alone clears 1B; combined with Modern Standard Arabic web text, the language as a whole clears 100B+.

**Classical Hebrew** in the narrow sense — Tanakh plus Mishnah plus the two Talmuds — totals only ~3M words. The broader Sefaria digital library ([sefaria.org](https://www.sefaria.org/)) of all classical and medieval Jewish texts (Hebrew + Aramaic) reaches ~150–200M words. The Tanakh alone is ~305,500 words in the Masoretic text. **First Filter:** cleared at 100M but not at 1B in the classical period; modern Hebrew is treated separately. **Second Filter:** ~150–200M digitized for the full historical corpus.

**Pali**, the canonical language of Theravada Buddhism, totals ~10–15M words across the Tipitaka and commentarial literature. **First Filter:** never cleared at 100M.

**Old Church Slavonic** has perhaps 1–2M words across ~50–100 surviving manuscripts. **First Filter:** never cleared at 10M.

**Sumerian**, [the Cuneiform Digital Library Initiative](https://cdli.mpiwg-berlin.mpg.de/) catalogs ~340,000 tablets with an estimated 3–5M words/tokens transliterated. ETCSL covers ~500K words of literary compositions. **First Filter:** never cleared at 10M cumulatively over the entire 3,000-year history of Sumerian literacy.

**Akkadian**, [ORACC](http://oracc.museum.upenn.edu/) holds ~10M+ digitized tokens, with perhaps another 10M unpublished. **First Filter:** cleared at 10M, never at 100M.

**Egyptian** (hieroglyphic + hieratic + demotic), the [Thesaurus Linguae Aegyptiae](https://thesaurus-linguae-aegyptiae.de/) holds ~1.7M lemmatized tokens. Total surviving Egyptian textual output over 3,000 years is estimated at 5–10M words, much of it formulaic funerary text. **First Filter:** cleared at 1M, never at 100M.

**Coptic** holds ~1.5M tokens in [Coptic Scriptorium](https://copticscriptorium.org/); total surviving literature is 5–10M. **First Filter:** cleared at 1M, never at 100M.

**Geez** (Classical Ethiopic), [Beta maṣāḥǝft](https://betamasaheft.eu/) catalogs ~1M+ tokens; total ~5–15M, much in unedited monastery manuscripts.

**Avestan** is among the smallest classical corpora at ~50,000–100,000 words total ([TITUS](https://titus.uni-frankfurt.de/)).

**Old Norse / Old Icelandic** has 5–8M surviving words.

**Old English** is famously small: the [Dictionary of Old English Corpus](https://doe.artsci.utoronto.ca/) at the University of Toronto contains every surviving word of Old English (c. 600–1150) and totals ~3M words plus ~1M of accompanying Latin ([CoRD](https://varieng.helsinki.fi/CoRD/corpora/DOEC/)). **First Filter:** cleared at 1M, never at 10M, full stop.

The pattern is unambiguous: the dead languages that did clear the First Filter at the 1B-token level (Latin, Classical Chinese, Classical Arabic, Sanskrit, Greek) are the ones where literate elites kept producing text continuously for at least a millennium and where institutional structures — Catholic Christianity, the Chinese imperial examination system, Islamic madrasas, Brahmin scholarly networks, Byzantine and patristic Christianity — kept the writing tradition alive at scale. Every dead language that lacked one of those institutional armatures fell short by orders of magnitude.

## 7. Comprehensive Ethnologue picture

Of the [7,164 living languages](https://www.ethnologue.com/ethnoblog/welcome-27th-edition/) recognized by Ethnologue 27 (2024):

Roughly 1,500 (~21%) have fewer than 1,000 speakers, and roughly 3,400 (~48%) have fewer than 10,000. The top 10 languages account for roughly half the world's L1+L2 speakers, and the top 100 account for ~85–90%. The long tail — roughly 6,000 languages — collectively accounts for under 3% of speakers.

About 4,000–4,200 languages have a writing system in active use, and SIL has developed orthographies for ~1,300 previously unwritten languages. Wycliffe-network organizations have produced complete Bible translations in ~700–800 languages and at least scripture portions in ~3,500 ([Wycliffe](https://www.wycliffe.org/about/why)). Only ~560 languages have any Wikipedia at all.

Approximately 3,045 (~43%) of living languages are classified as endangered — vulnerable through critically endangered. UNESCO and Ethnologue both project that several thousand languages will lose all native speakers within this century. From a Great Filters standpoint, these languages will *never* pass even the First Filter; their entire surviving textual corpus, where one exists, is already effectively complete.

The intersection of "produces enough text to train a small LM" and "exists in machine-readable form" is, in 2026, roughly 100 languages. The intersection of "produces enough to train a competitive LLM" and "exists in machine-readable form at the 100B-token scale" is roughly 12.

## 8. Implications for talkietoken

Three points stand out for the broader talkietoken project.

First, the choice of "1930-13b-base" as a framing implies a counterfactual world in which the model trains primarily on pre-1930 text. The First Filter analysis here is essentially the question of which languages had crossed the 10B-token threshold by 1930. Conservatively, that list is roughly: English, French, German, Spanish, Italian, Russian, Mandarin/Classical Chinese, Japanese, Arabic, Latin, Ancient Greek, plus probably Portuguese and Dutch, plus possibly Hebrew (in its scholarly form), Polish, and Czech. Everything else is doubtful. This matches the multilingual_tests dictionary in talkietoken-langs.py reasonably well — the languages tested there (French, German, Spanish, Russian, Chinese, Japanese, Arabic, Hebrew, Thai) are mostly in this set, with Thai as the genuinely dubious case.

Second, anachronism detection in talkietoken-anachronism.py has a natural Great Filter analog. A 1930 model "shouldn't know" tokens that postdate that filter crossing — postwar neologisms, internet vocabulary, modern proper nouns. The First Filter framework gives a principled way to extend that test: a 1930 Hebrew or Czech or Indonesian model is *qualitatively* different from a 1930 English model because those languages were in very different First Filter states in 1930.

Third, the systematicleak and genleak experiments are essentially probing whether tokens specific to *post*-1930 Second Filter expansion (the digitization, online culture, and crawling era) bled into a model that should not have seen them. The Second Filter framing makes this a sharper hypothesis: post-1990 token shapes should be especially leaky because the Second Filter is what introduced them.

## 9. Caveats and known weaknesses

Several of the numbers above warrant skepticism. Pre-modern literacy rates are contested ([Harris 1989](https://archive.org/details/ancientliteracy0000harr)) and the ranges used here are conservative midpoints. Patrologia Latina's exact word count is variously cited at 80M–113M depending on whether one means Migne's printed pages or Brepols' digital edition. TLG's word count has grown over time and current claims range from 110M to 125M. Wikipedia per-language word counts are extrapolations from article counts and average lengths and should be treated as order-of-magnitude. The bot-Wikipedia problem (Cebuano, Waray, parts of Swedish and Dutch) means published article counts overstate actual human-authored text by 10–100×.

The threshold-crossing chronologies in §4.1 are not from a published source and represent the author's back-of-envelope reasoning over Maddison-Project population estimates, [Our World in Data](https://ourworldindata.org/literacy) literacy series, and Buringh–Van Zanden book-output series. They are intended as a calculation framework rather than as cite-worthy claims.

Finally, the "everyone above 100B tokens" picture in §5 treats the open web as a complete sampling of the world's digitized text, which it is not. State-internal corpora (Chinese baike sites, Korean Naver content, Japanese 2ch/5ch material) are systematically underrepresented in Common Crawl. The picture for those languages is rosier than the published web-corpus numbers suggest, though the magnitude of the correction is itself disputed.

## 10. Sources

The principal sources used in this note are:

[Ethnologue 27th Edition (2024)](https://www.ethnologue.com/ethnoblog/welcome-27th-edition/) for living-language counts and speaker distribution.

[Joshi, Pratik et al. 2020. "The State and Fate of Linguistic Diversity and Inclusion in the NLP World." ACL.](https://arxiv.org/abs/2004.09095) for the resource-tier classification and the "~7 high-resource languages" framing.

[Kudugunta, Sneha et al. 2023. "MADLAD-400: A Multilingual And Document-Level Large Audited Dataset."](https://arxiv.org/abs/2309.04662) for per-language token counts across 419 languages.

[Nguyen, Thuat et al. 2023. "CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages."](https://arxiv.org/abs/2309.09400) for the 6.3T-token corpus's per-language breakdown.

[Abadji, Julien et al. 2022. OSCAR-23.01.](https://oscar-project.org/post/oscar-2301/) for the alternative web-corpus per-language picture.

[Xue, Linting et al. 2020. "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer." (mC4)](https://arxiv.org/abs/2010.11934) for the original Common Crawl multilingual scaling.

[Penedo, Guilherme et al. 2024. "The FineWeb Datasets."](https://arxiv.org/abs/2406.17557) for current Common Crawl breakdown and quality-filtering practice.

[Buringh, Eltjo and Jan Luiten van Zanden. 2009. "Charting the 'Rise of the West': Manuscripts and Printed Books in Europe, A Long-Term Perspective from the Sixth through Eighteenth Centuries." Journal of Economic History 69(2):409–445.](https://doi.org/10.1017/S0022050709000837) for surviving manuscripts and printed-book counts.

[Our World in Data — Literacy.](https://ourworldindata.org/literacy) for literacy series.

[Maddison Project Database 2023.](https://www.rug.nl/ggdc/historicaldevelopment/maddison/) for historical population.

[Harris, William V. 1989. *Ancient Literacy*. Harvard University Press.](https://archive.org/details/ancientliteracy0000harr) for Roman/Greek literacy estimates.

[Wikipedia statistics — List of Wikipedias.](https://meta.wikimedia.org/wiki/List_of_Wikipedias)

[Common Crawl statistics — language distribution.](https://commoncrawl.github.io/cc-crawl-statistics/plots/languages.html)

For dead-language corpora: [TLG](https://stephanus.tlg.uci.edu/), [Brepols Library of Latin Texts](https://about.brepolis.net/library-of-latin-texts/), [Corpus Corporum](https://www.mlat.uzh.ch/), [GRETIL](http://gretil.sub.uni-goettingen.de/), [Chinese Text Project](https://ctext.org/), [CBETA](https://www.cbeta.org/), [OpenITI](https://openiti.org/), [Sefaria](https://www.sefaria.org/), [CDLI](https://cdli.mpiwg-berlin.mpg.de/), [ORACC](http://oracc.museum.upenn.edu/), [Thesaurus Linguae Aegyptiae](https://thesaurus-linguae-aegyptiae.de/), [Coptic Scriptorium](https://copticscriptorium.org/), [Beta maṣāḥǝft](https://betamasaheft.eu/), [TITUS](https://titus.uni-frankfurt.de/), [Dictionary of Old English Corpus](https://doe.artsci.utoronto.ca/), [Siku Quanshu Wikipedia entry](https://en.wikipedia.org/wiki/Siku_Quanshu), [CoRD on the DOE Corpus](https://varieng.helsinki.fi/CoRD/corpora/DOEC/).
