# Behavioral semantic-neighborhood comparison (talkie vs a modern LLM)

Method: ask each model for ten single-word associations of a target word
(temp 0, n=1 — illustrative, not aggregated). Behavioral proxy for the
"same token, different meaning" question; NOT embedding geometry.
talkie = hosted chat API; modern comparand = this assistant (labeled).

| word | talkie (1880–1930) | modern |
|---|---|---|
| gay | lively, merry, cheerful, sportive, frolicsome, blithe, jolly | pride, LGBTQ, homosexual, rainbow, queer, happy, parade |
| broadcast | sown, scattered, dispersed, published, promulgated, diffused | radio, television, news, signal, channel, media, live |
| mouse | rat, hole, cheese, trap, nibble, squeak, grimalkin | computer, click, cursor, keyboard, pointer, scroll, pad |
| web | cobweb, spider, tissue, cambric, veil, network, film | internet, site, online, browser, page, link, www |
| traffic | trade, commerce, dealing, thoroughfare, bustle, business | cars, road, jam, congestion, highway, gridlock, commute |
| wireless | telegraphy, Marconi, operator, station, message, system | wifi, router, bluetooth, broadband, signal, mobile |
| computer | reckoner, calculator, tabulator, scorer, computist, computator | laptop, software, processor, data, machine, screen |
| gas | light, burner, jet, main, pipe, works, fittings, smell | gasoline, fuel, car, petrol, pump, tank, mileage |

Neighborhood overlap (Jaccard) ≈ 0 for all eight: the dominant sense of
each token flipped between the two corpora. "computer" → human reckoner;
"wireless" → Marconi telegraphy; "gas" → coal-gas lighting; "mouse" →
rodent; "gay" → merry. Caveat: n=1, chat-model behavior not base-model
geometry, comparand is a different model family.
