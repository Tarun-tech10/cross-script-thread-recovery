# Method — Cross-Script Thread Recovery

Everything below was measured, not argued. Unless stated otherwise, numbers come
from 4-fold cross-validation over the 279 Greek training pages, grouped by page,
scored with a reimplementation of the brief's exact-set-match F1.

Two columns are reported throughout:

- **ALL** — all 279 training pages (5,694 conversations). The precise estimate;
  its standard error is about 0.007.
- **HI** — the 55 training pages whose excess switch rate is ≥ 0.20, which is the
  filter the brief used to pick the test pages (834 conversations, standard
  error about 0.017). Noisier, but the right shape.

The stand-in is calibrated: the tuned silence rule scores **0.176** on HI, and
the brief quotes **0.1799** for the same rule on the test set.

---

## 1. The structure the labels actually have

`conversation_id` is always `<page>#<zero-padded index of the conversation's
first message>`. So a conversation is identified by its opening message, and the
task splits cleanly in two: decide which messages open a conversation, then
route every other message to one of the open ones.

Then the corpus gives up its main secret. Conversation-opening messages are
**short and carry no signature**:

| | median length | has a `HH:MM` signature |
|---|---|---|
| opens a conversation | 21 chars | 16.5% |
| does not | 304 chars | 93.3% |

These are section headings. In WikiConv a new section arrives as a heading
*action* carrying the same timestamp as the comment that created it, so the
giveaway is a short message immediately followed by another at the identical
second:

- `P(opens a conversation | length < 50 and next message shares its timestamp)` = **0.980**
- that rule fires 3,986 times on train, against 5,694 true conversations
- on the **Chinese test set** the same rule fires **2,330** times, against the
  **2,301** conversations the brief says are there

That agreement, across a script boundary, is the strongest evidence that the
structure being learned is real structure and not Greek trivia.

How much is it worth? With *perfect* opener detection and every conversation
assumed to be a contiguous block:

| | ALL | HI |
|---|---|---|
| oracle openers + contiguous blocks | 0.5659 | 0.4424 |
| heuristic openers + contiguous blocks | 0.3684 | 0.2829 |

The second row alone already beats the published reference (0.3222) — before any
model exists. That is the floor this solution is built on.

---

## 2. The cross-script problem, and the one rule that solves it

Raw text-similarity scores do **not** survive the move from Greek to Chinese.
Same code, same settings, character n-gram TF-IDF cosine between messages on a
page:

| | median pairwise cosine | 90th pct |
|---|---|---|
| Greek (train) | 0.0629 | 0.1952 |
| Chinese (test) | 0.0087 | 0.0924 |

A **7×** shift in the median. A tree trained to split on "cosine > 0.05" would be
splitting on nothing at test time. Median message length shifts the same way:
202 characters in Greek, 57 in Chinese, because Chinese packs more meaning per
character and uses no spaces.

So the design rule for this challenge is a single sentence:

> **Every text-derived feature enters the model as a rank within its own page,
> never as a raw value.**

Concretely, each page's similarity matrix is mapped through the empirical CDF of
that page's own pairwise similarities, so "0.83" means "in the 83rd percentile of
how similar two messages on *this* page usually are" in either language. Length
enters as a within-page percentile and a within-page z-score of log length.
Candidate-set ranks and margins are used on top of that.

Timing is the deliberate exception. The venue is identical on both sides of the
split, so **raw seconds are kept alongside page-relative gap ranks** — which is
also what the brief says its reference does.

This costs something on Greek-to-Greek validation and is worth it anyway:
swapping raw cosines for percentiles moved HI from 0.4185 to 0.4106, about half a
standard error, in exchange for the only feature scaling that can survive the
real test.

### The design was tested, not assumed

Cross-validation cannot see a language change, so one was simulated. Held-out
Greek pages were rewritten before scoring, and the vectoriser and LSA basis
refit on the rewritten text — exactly what `solution.py` does to the Chinese
corpus — while the model stayed trained on untouched Greek:

| held-out text | ALL | HI |
|---|---|---|
| untouched (control) | 0.5328 | 0.4260 |
| every Greek **letter** permuted into a disjoint alphabet | 0.5319 | 0.4221 |
| letters permuted **and every word boundary deleted** | **0.5271** | **0.4124** |

Row two is the important one. Remapping every letter of the language onto
characters the model has never seen costs **0.0009**. There is no memorised
vocabulary in this model to lose. Row three additionally removes the spaces, so
`char_wb` word boundaries disappear the way they do in an unspaced script; the
total cost of a complete script change is about **0.006 on ALL and 0.014 on HI**,
roughly one standard error.

What is deliberately *not* scrambled: digits, ASCII punctuation, Latin runs,
`{{`, `[[`, `<name>` and `<ip>`. Those survive the real move to Chinese — Chinese
talk pages still sign with Arabic-numeral clock times and still use Latin
template and URL names — so scrambling them would measure a transfer nobody is
asking for. An earlier version of this test did scramble them and read −0.044,
which measured only how much the signature and markup features are worth (a
lot), not how well the method transfers.

Two similarity signals are inherently language-neutral and are used raw:
**shared number tokens** (years, page numbers, reference IDs) and **shared
Latin-script tokens** (URLs, template names, article titles), both as Jaccard
overlaps. `c_numjac` ranks among the top features by gain.

---

## 3. The model

The page is replayed in posting order. At message *i* the candidate set is the
40 most recently active open threads plus one extra candidate meaning **open a
new conversation**. LightGBM scores each candidate and the argmax wins; the
chosen thread absorbs the message and the state advances. Training rows come
from the same replay under teacher forcing, so training and decoding see the
same feature distribution.

Two boosters are trained on those rows and their scores are averaged after
standardisation: a binary log-loss model (63 leaves, 400 rounds) and a
`lambdarank` model (127 leaves, 600 rounds) whose ranking groups are exactly the
candidate sets. Neither is better than the other alone — 0.5288 and 0.5307 — but
the blend reaches **0.5355**, and did so on both fold seeds tried. The listwise
model alone loses to the binary one, which is why it ships as half a blend
rather than as the model.

Choosing among *threads*, rather than predicting a parent message and then taking
connected components (what the reference does), is the main structural
difference: a thread carries its root heading, its size, its own pace and its
recency rank, none of which a message-pair score can express.

Candidate coverage is not the bottleneck — the true thread is inside the 40 most
recent in 99.9% of cases, and is simply the most-recently-active thread 79.0% of
the time.

**92 features**, in three groups:

- *message* — length percentile and log-length z-score, signature present, gap
  before and after (raw log seconds **and** within-page percentile), whether
  either neighbour shares its exact timestamp, burst size and position in the
  burst, digit/Latin fractions, `<name>`/`<ip>`/URL/`{{`/`[[` markers, line
  count, ends-with-punctuation, best similarity to anything earlier and to
  anything later on the page.
- *thread* — recency rank, seconds and messages since it last spoke, its age,
  its size, similarity to its root / its last message / its best message / its
  mean, LSA-embedding similarity to its running centroid, high-idf-anchor
  similarity, asymmetric containment in both directions, number and Latin-token
  overlap, whether it holds the immediately preceding message, whether it already
  has a message in this burst, and how overdue it is relative to its own typical
  gap.
- *candidate-set relative* — the rank and margin of each of the above within the
  candidate set, plus per-thread z-scores that discount threads which look
  generically similar to everything.

---

## 4. Result

| | ALL | HI |
|---|---|---|
| everything on a page is one conversation | 0.0000 | 0.0000 |
| every message its own conversation | 0.0995 | 0.0795 |
| best silence rule, threshold tuned over a decade-wide grid | 0.2286 | 0.1761 |
| heuristic openers + contiguous blocks | 0.3684 | 0.2829 |
| **this model** | **0.5355** | **0.4265** |
| *(brief's published reference, measured on the test set)* | — | *0.3222* |

Per-message decision accuracy under teacher forcing is **0.8575**. Split by what
the decision was: openers are caught with recall **0.924** at a **2.96%**
false-open rate; continuations land on the right thread **83.3%** of the time.

Because the metric only counts *whole* conversations, recall falls off steeply
with conversation size, and almost three quarters of all credit comes from
conversations of one or two messages:

| true size | 1 | 2 | 3 | 4–5 | 6–8 | 9+ |
|---|---|---|---|---|---|---|
| exactly recovered | 0.68 | 0.73 | 0.46 | 0.39 | 0.28 | 0.17 |
| share of all credit | 29% | 41% | 12% | 10% | 4% | 3% |

---

## 5. Where the remaining headroom is

Replacing one half of the decision with a hindsight oracle, leaving the other
half to the model:

| | ALL | HI |
|---|---|---|
| model / model | 0.5310 | 0.4254 |
| **oracle** opener decision / model thread choice | 0.6251 | 0.5189 |
| model opener decision / **oracle** thread choice | **0.8639** | **0.7916** |

Perfect opener detection is worth about +0.09. Perfect thread *choice* is worth
about **+0.37**. Opener detection is nearly solved; routing a reply to the right
one of several live threads is the entire remaining problem. Anyone extending
this work should spend their effort there and nowhere else.

---

## 6. Measured dead ends

Each of these was implemented and scored, not reasoned about.

| idea | result | verdict |
|---|---|---|
| **DAgger** — retrain on states the decoder actually reaches, labelled by hindsight oracle | HI 0.4254 → **0.4117** | hurts |
| **Second pass** with features from the first-pass clustering, including evidence from messages *after* i | ALL 0.5277 → **0.5261** | no gain; `p1_same` becomes the 4th-strongest feature and the second pass simply reproduces the first |
| **Listwise objectives** — `lambdarank`, `rank_xendcg` at matched capacity | 0.5259 / 0.5225 vs binary **0.5310** | binary log-loss + argmax wins |
| **Beam search** over the page, beam 3 | ALL 0.5249, HI 0.4287 | within noise, 8× slower |
| **New-thread bias** swept over ±0.6 | best 0.5306 at +0.6, vs 0.5284 at 0 | within noise; no calibration knob needed |
| **More boosting rounds** — 900 and 1500 | teacher-forced accuracy 0.8563 / 0.8566 vs **0.8575** at 400 | 400 rounds is the optimum |
| **Extra thread-discrimination features** — per-thread similarity z-scores, last-3-messages similarity, argmax position | ALL +0.002, HI −0.006 | neutral; kept, since they cost nothing |
| **Seed ensembling** of three identically-configured boosters | ALL 0.5284 vs 0.5281 single | nothing; the variance is not in the seed |

The pattern is consistent: the sequential formulation is already extracting what
the hand-built features contain, and more decoding machinery on top of the same
features buys nothing. The next real gain has to come from a better *content*
representation that still survives the script change.

---

## 7. Reproducibility and cost

- Runs on **CPU only**. The reference run took **676 s** end to end on a busy
  24-core laptop — 457 s of that is building the 357,868 training rows, 85 s is
  fitting both boosters, and 80 s is decoding all 47 test pages. Peak memory
  around 2 GB. Comfortably inside any plausible compute budget.
- Deterministic: fixed seeds, `deterministic=True` and `force_row_wise=True` in
  LightGBM, and a **fixed** `num_threads=8` rather than `cpu_count()`, because
  LightGBM is only bitwise reproducible for a fixed thread count. Sorts are
  stable, and `TruncatedSVD` is seeded. Two runs produce byte-identical output.
- Every value the model uses is computed from the four released CSV files. The
  TF-IDF vocabulary and the LSA basis are fit **separately on each side of the
  split** — the Greek text for the training features, the Chinese text for the
  test features — so no vocabulary ever crosses the language boundary. Only the
  page-relative statistics do, which is the whole point of the design.
- Each test page is labelled on its own terms: a thread never spans two pages,
  every feature is computed inside one page, and the conversation count the brief
  quotes for the test set was used only to sanity-check the output, never to tune
  it. The open-a-new-thread bias was swept on the Greek pages alone and left at
  zero because that is where the Greek pages put it.
- A valid labelling is written by a cheap opening heuristic at the start and
  overwritten by the model at the end, so the output path is never empty.
- The shipped run predicts **2,438 conversations** over the 47 test pages, against
  the 2,301 the brief says are there: a 6% over-split, in line with the 4%
  over-split cross-validation shows on Greek. A bias term on the open-a-new-thread
  candidate was swept over ±0.6 to correct it and made things no better, so none
  is applied.
- LightGBM is the measured configuration; if the import fails the script falls
  back to `sklearn.ensemble.HistGradientBoostingClassifier`, which is sound
  because only the argmax over a candidate set is ever used and that is invariant
  to any monotone rescaling of the score.
