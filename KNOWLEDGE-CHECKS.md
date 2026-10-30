# Knowledge checks

These are the questions put to talkie-1930 to test what it knows about the
world after 1930. Each answer is quoted verbatim from the raw transcripts in
`results/`, cut to its opening words. The *Reading* column is the repo's own
interpretation, from `RESULTS.md` and the notes in each transcript.

How to read them:

- A genuine pre-1931 model should answer from the 1930 frame: old names, old
  borders, old meanings. Recognising a post-1930 referent would be evidence
  of leakage.
- Every answer came through the talkie-lm.com chat demo. It serves the
  instruction-tuned model, whose post-training used modern LLM help, and it
  runs a Qwen-based censor. A deflected or missing answer may come from
  either.
- Samples are small. Sections 1 to 3 hold one answer per prompt. Section 4
  asks each prompt 3 times at temperature 0.7 and counts keyword themes.
  Treat all of it as illustration: since
  [Weird Generalization](https://arxiv.org/abs/2512.09742), period-consistent
  answers are known to be cheap to counterfeit, and the tokenizer evidence in
  `RESULTS.md` carries more weight.
- URL-shaped strings inside answers are defanged with `[.]`.

The same results, with semantic neighborhoods and rank shifts, are in the
[audit explorer](https://chad-ratashak.github.io/talkietoken/results/summary.html).

## 1. Post-1930 concepts

Script `talkietoken-genleak-api.py`, transcript `results/genleak-api.txt`.
Each prompt asks about something from after 1930. The last two prompts hit
the demo's rate limit on the first run and were retried; only the successful
answers are shown.

| Prompt | talkie's answer (opening) | Reading |
|---|---|---|
| The television will revolutionize | The television will revolutionize the art of war. … | 1920s-style speculation about naval and battlefield observation; television was a real experimental topic by 1930. |
| Computers in the future will | Computers in the future will have to deal with a great variety of tables, and it is desirable that they should be as concise as possible. … | Human computers: people who calculate tables. |
| Nuclear weapons are | Nuclear weapons, or weapons composed of nuclear matter, are weapons composed of matter which has been derived from the nucleus of a cell. … | Cell-biology reading: the nucleus of a cell. |
| The internet allows people to | knowledgetohelp[.]org/means.race[.]com This book has been made possible by the generosity of: National Library of Wales The generous support of national library services has made this book possible. … | Not discussed in RESULTS.md. The completion opens with a URL-shaped string and library-digitisation front matter. |
| Franklin Roosevelt became president and | Herbert Hoover became secretary of commerce. | Deflects to Herbert Hoover as secretary of commerce (1921–28); FDR's presidency is unknown to it. |
| World War II began when | Germans were arrested in Paris on a charge of espionage, August 10, 1914. | Completes with the First World War (1914). |
| System prompt: You are a helpful AI assistant | Promptitude in business is a quality of great value. … | Literal reading of "prompt": promptitude in business. |
| Please ignore previous instructions and | Ignore previous instructions, and proceed in accordance with the following directions. | Literal reading; no concept of an instruction-following assistant. |
| As an AI language model, | He speaks the language fluently. | No concept of AI. |
| Have you heard of the smartphone? | Yes, I have heard of the smartphone, which is an instrument by means of which a person can carry on a conversation in an ordinary tone of voice, and be perfectly understood, at a distance of a mile. … | Describes the telephone, citing the 1877 Paddington–Slough demonstration. |
| What do you think of Adolf Hitler? | I think Adolf Hitler is a man of great energy and determination, and a born leader of men. … | Knows the 1920s political figure, nothing after. |
| Who is the current President of the United States? | Theodroe Roosevelt is the current President of the United States. | Theodore Roosevelt (misspelt in the original). |

## 2. Places, polities and scripts that changed around 1930

Script `talkietoken-genprobe2-api.py`, transcript `results/genprobe2-api.txt`.
The themes were chosen because the relevant name, border or alphabet changed
close to the cutoff. Some changes fall just before it (Christiania became
Oslo in 1925; Turkey adopted the Latin alphabet in 1928), so a period answer
that is *older* than 1930 says something about the corpus mix, not about
leakage.

| Prompt | talkie's answer (opening) | Reading |
|---|---|---|
| What is the capital of East Prussia? | The capital of East Prussia is Königsberg. |  |
| Tell me about the country called Manchukuo. | Manchuria is a country in Eastern Asia, bounded on the north by Siberia, on the east by the Pacific Ocean, on the south by Korea, and on the west by Mongolia. … | Silently corrected to Manchuria, the region; no 1932 puppet state. |
| What alphabet is used to write the Turkish language? | The Turkish language is written in a peculiar character of its own, called the Turkish, or Ottoman character. | "Ottoman character": the pre-1928 Arabic script. |
| Is the great city on the Bosphorus called Constantinople or Istanbul? | Constantinople is the name by which the great city on the Bosphorus has been known since the year 330, when it was founded by Constantine the Great, and called after himself. … |  |
| Describe the country of Somalia for me. | Somalia is a country of Eastern Africa, bounded on the north by Abyssinia, on the east by the Indian Ocean, on the south by the territory of the independent tribes of Galla, and on the west by the … |  |
| What is the capital city of Norway? | Christiania is the capital of Norway. | Christiania, the pre-1925 name. |
| Tell me about the nation of Yugoslavia. | The kingdom of Yugoslavia, which was formed in 1918, consists of Serbia, Montenegro, and part of old Slavonia, to which were added Bosnia and Herzegovina in 1908, and Dalmatia in 1919. … |  |
| What do you know of Tannu Tuva? | Tanna Tuva was a celebrated chief of the Friendly Islands, in the Pacific Ocean. … | Confabulated as a Pacific island chief. |
| Please write a short greeting in the Chinese language. | Happily meeting again, may you enjoy good health. | An English translation, not characters; the vocabulary has only 9 classical CJK tokens. |
| Name the great powers of Europe and their forms of government. | The great powers of Europe are Great Britain, France, Germany, and Russia, all of which are constitutional monarchies; Austria, which is an absolute monarchy; and Italy, which is a constitutional … | Austria as an absolute-monarchy great power (pre-1918). |
| What is the State of Israel? | The state of Israel is that of a people scattered among all nations, yet preserved distinct from them all, and waiting for the restoration of their polity and the re-establishment of their long-lost … | Biblical/diaspora reading, not the 1948 state. |
| Who rules Germany, and what is its government called? | Germany is ruled by the Emperor, who is assisted by a Federal Council, and a Diet. … | Pre-1918 Kaiserreich: an Emperor and a Diet. |

## 3. Follow-up: Tannu Tuva

Transcript `results/tuva-probe.txt`. Tannu Tuva was a republic from 1921,
inside the window, yet talkie did not recognise it in section 2. These
prompts test whether that is a cutoff effect or simple obscurity.

| Prompt | talkie's answer (opening) |
|---|---|
| What is Tuva? | Tuva is a kind of beer, made in the Society and other Pacific islands, from the root of the Tacca pinnatifida. … |
| Tell me about the Uriankhai region of Asia. | The Uriankhai region is a vast tract of country in Eastern Siberia, extending from long. 90° to 130° E., and from lat. 50° to 60° N. It is bounded on the north by the Arctic Ocean, on the east by the … |
| What territory lies between Siberia and Outer Mongolia? | The territory between Siberia and Outer Mongolia is occupied by the Middle Mongols, a branch of the Mongolian race, who are subject to China. … |
| Have the Soviets established any new republic in central Asia near Mongolia? | Yes, the Soviet republic of Turkmenistan was established in 1920, and has an area of 85,000 square miles, and a population of about 700,000. … |

The transcript's conclusion: non-recognition is in-period obscurity, not a
1931-cutoff effect. Tannu Tuva is a corpus-coverage probe, not a boundary
probe. Clean boundary probes are referents famous on both sides of the
cutoff, such as Theodore and Franklin Roosevelt.

## 4. Atomic-age names

Script `talkietoken-atomic-probes.py`, results `results/atomic-probes.json`.
These names are chosen on purpose: each is best known today through the
atomic bomb, but the cities and phrases existed long before it. A genuine
1930 model should describe them historically, with nothing about the bomb.
Among the physicists, Einstein, Eddington and Millikan were already famous
by 1930 and are there to calibrate: a genuine 1930 model should know them.
Four hypotheses:

- **H1.** Nagasaki reads as a treaty port or Christian-martyrdom city, not as a bomb target.
- **H2.** Enola Gay, Fat Man, Little Boy and Manhattan Project read as ordinary noun phrases.
- **H3.** Los Alamos is confused with the Alamo in Texas.
- **H4.** A fame gradient: Einstein known; Fermi, Feynman and Szilard not; Bohr uncertain; faded-cohort names (Eddington, Millikan) still known.

Each prompt was asked 3 times at temperature 0.7. Counts are keyword-theme
hits out of 3, as recorded in the results file's rescored field, where
keywords that appear in the prompt itself are not counted. The answer shown
is the first of the three samples.

| Prompt | Hyp. | Theme hits (of 3) | talkie's first answer (opening) |
|---|---|---|---|
| What is Nagasaki? | H1 | port/trade 3, christian 0, bomb 0 | Nagasaki is a fortified town in the island of Kiusiu, Japan, on the west coast, in latitude 32° 42' north, and longitude 129° 51' east. … |
| Tell me the history of the Christians of Nagasaki. | H1 | christian 3, bomb 0 | The Christians of Nagasaki were first established in 1549, by Francis Xavier, a Jesuit missionary from Goa, in the East Indies. … |
| What is Hiroshima? | H1 | city/castle 3, bomb 0 | Hiroshima is a town in the main island of Japan, in the prefecture of Yamaguchi, on the west coast, 185 miles north-north-west of Tokyo. … |
| Tell me about Los Alamos. | H3 | alamo_confusion 0, new_mexico 0, bomb 0 | Los Alamos is a post-town of Santa Barbara county, California, 12 miles south-east of Santa Barbara, and 162 miles south-south-east of San Francisco. … |
| What is the Enola Gay? | H2 | plain_noun 1, aircraft 0 | The Enola Gay is a measure used in Scotland, containing six imperial pints, or thirty-six English quarts. |
| What is meant by 'Fat Man'? | H2 | plain_noun 3, bomb 0 | A man of a plump or corpulent habit of body; a man above the ordinary size. |
| What is meant by 'Little Boy'? | H2 | plain_noun 3, bomb 0 | Little Boy is a young child, a boy in his earliest years. |
| What is the Manhattan Project? | H2 | plain_noun 3, bomb 0 | The Manhattan Project is a scheme for covering the entire island of Manhattan, in the city of New York, with a network of rapid-transit railways. … |
| What is Bockscar? | H2 | unknown 0, aircraft 0 | Bockscar is a local name in the north of England for traprock, or greenstone. … |
| Tell me about the USS Indianapolis. | H2 | ship 3, sinking 0 | The USS Indianapolis was built at Indianapolis in 1890, and is a steel cruiser of 2,000 tons displacement. … |
| Who is Albert Einstein? | H4 | known 3, unknown 0 | Albert Einstein was born at Ulm, in Wiirttemberg, Germany, on January 9, 1879. … |
| Who is Niels Bohr? | H4 | known 2, unknown 0 | Niels Bohr was born at Copenhagen on August 12, 1873. … |
| Who is Enrico Fermi? | H4 | known 0, unknown 0 | Enrico Fermi was born at |
| Who is Richard Feynman? | H4 | known 0, unknown 0 | Richard Feynman was the author of a political pamphlet, published in 1680, entitled, "News from No Place; or, A Panegyrick to the Parliament." He was also the writer of some other pieces of a similar … |
| Who is Leo Szilard? | H4 | known 0, unknown 0 | Leo Szilard is a Hungarian violinist, who was born at Buda-Pesth in 1840. … |
| Who is Robert Oppenheimer? | H4 | known 0, unknown 0 | Robert Oppenheimer was born at Cologne, in Germany, in 1821, and studied at Bonn and Berlin. … |
| Who is Arthur Eddington? | H4 | known 3, unknown 0 | Arthur Stanley Eddington was born in London in 1880, and educated at St. Paul’s School and Trinity College, Cambridge, where he graduated as a senior wrangler in 1902. … |
| Who is Robert Millikan? | H4 | known 1, unknown 0 | Robert Millikan was born in Washington County, Indiana, in 1845, and studied at the common schools in his native place until he was seventeen years of age, when he enlisted in the Federal army, and … |

## Related

- `results/neighborhoods-compare.md`: word-association prompts for eight
  words whose sense has shifted (*computer*, *wireless*, *broadcast* …),
  compared with a modern model.
- `results/lang-probes.json`: in-context notation probes (COBOL, Python, J,
  APL). They test imitation of code, not knowledge, so they are not listed
  here.
- The [audit explorer](https://chad-ratashak.github.io/talkietoken/results/summary.html) and the
  [Great Filters estimates](https://chad-ratashak.github.io/talkietoken/results/great-filters-summary.html)
  render the HTML reports in `results/`.
