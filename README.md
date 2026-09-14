# Cross-Script Thread Recovery on Wikipedia Talk Pages

## What this is, in plain words

A Wikipedia talk page is a pile of conversations that arrived mixed together.
Someone replies to a two-week-old thread while a new thread is being started
three lines above. A person untangles this without thinking about it.

This program does the untangling automatically. You give it the messages of a
page — just the text and the time each one was posted — and it tells you which
messages belong to the same conversation.

The hard part: it **learns on Greek pages and is graded on Chinese ones**. The
two languages share no words at all, and Chinese is not written with spaces, so
anything the program learned as "a Greek word" is worth exactly nothing on the
test. It also never sees who wrote anything, how deeply a reply was indented, or
what any section was titled.

**No graphics card needed. It runs on any normal laptop in about ten to fifteen minutes.**

---

## What you need before starting

| | |
|---|---|
| Any computer | Windows, Mac or Linux — **no NVIDIA card required** |
| Computer memory (RAM) | 4 GB or more |
| Free disk space | about 500 MB |
| Time | 10-15 minutes, mostly unattended |

---

## Step 1 — Install Python (skip if you already have it)

Download from [python.org/downloads](https://www.python.org/downloads/) and run
the installer.

> **On the first screen, tick the box that says "Add Python to PATH"** before
> clicking Install. If you miss it, none of the commands below will work and you
> will have to reinstall.

---

## Step 2 — Download this code

Green **`< > Code`** button at the top of this page → **Download ZIP**.
Right-click the downloaded file → **Extract All**.

The data is already inside the repo as `CST.zip`, so there is nothing else to
fetch. Your folder should look like this:

```
cross-script-thread-recovery/
    solution.py
    check_submission.py
    requirements.txt
    README.md
    description.md
    CST.zip            <-- the data, already here (6 MB)
```

---

## Step 3 — Open a terminal in that folder

**Windows:** open the extracted folder in File Explorer, click the address bar,
type `cmd` and press Enter.

**Mac/Linux:** right-click the folder → open in Terminal.

---

## Step 4 — Install the four libraries it needs

```
pip install -r requirements.txt
```

Takes a minute or two. If `pip` is not recognised on Windows, use
`python -m pip install -r requirements.txt` instead.

---

## Step 5 — Unpack the data

```
python -c "import zipfile; zipfile.ZipFile('CST.zip').extractall('CST_DATA')"
```

You should now have a `CST_DATA` folder holding five `.csv` files.

---

## Step 6 — Run it

```
python solution.py CST_DATA submission.csv
```

It prints its progress as it goes. Expect something close to this:

```
data dir : CST_DATA
output   : submission.csv
train 21146 msgs / 279 pages | test 9921 msgs / 47 pages
[    1s] wrote submission.csv (heuristic fallback)
[  457s] training rows (357868, 92), positives 21123, groups 21146
    trained binary (400 rounds)
    trained lambdarank (600 rounds)
[  542s] model trained
[  596s] test pages encoded
[  676s] 2438 conversations predicted, median size 2.0
[  676s] wrote submission.csv (model)
[  676s] done
```

Two things worth noticing while it runs:

- It writes a **valid submission within the first few seconds**, using a simple
  rule, and only then overwrites it with the real model output. If the machine
  dies halfway through you still have a usable file.
- The conversation count it reports at the end should land near **2301**, which
  is how many real conversations the test pages contain. The reference run gets
  2438 — about 6% too many, which is the same slight over-splitting
  cross-validation shows on Greek.

---

## Step 7 — Check the file before you send it

```
python check_submission.py submission.csv CST_DATA
```

It must end with `OK - submission is valid.` That means: one row per query, no
duplicates, no missing rows, exactly the two required columns.

---

## What score to expect

The challenge brief quotes these, all measured by its own grader on this exact
split:

| approach | score |
|---|---|
| everything on a page is one conversation | 0.0000 |
| every message is its own conversation | 0.0512 |
| link each message to the one before it | 0.0000 |
| best "new thread after N seconds of silence", N tuned over a decade-wide grid | 0.1799 |
| nearest earlier message by shared vocabulary | 0.0553 |
| **published reference** (trained pair ranker + connected-component decode) | **0.3222** |

This program, measured by 4-fold cross-validation over the 279 Greek training
pages (the only labelled data there is):

| measured on | score |
|---|---|
| all 279 training pages | **0.5355** |
| the 55 training pages that interleave as heavily as the test pages do | **0.4265** |

That second row is the honest comparison. The test pages were deliberately
filtered to be the tangled ones, so the 55 most-interleaved training pages are
the closest stand-in available. Two sanity checks that this stand-in is fair:
the tuned silence rule scores **0.176** on it against **0.1799** quoted on the
test set, and the training pages are the same venue, same conventions, same
editing rhythm.

Expect the real test score to land near the lower of those two numbers, because
the model is also crossing a language boundary.

That boundary was tested rather than hoped about. Held-out Greek pages were
rewritten with **every Greek letter permuted into an alphabet the model has never
seen, and every word boundary deleted**, then scored by a model still trained on
untouched Greek. The whole score of that transformation:

| held-out text | all pages | interleaved pages |
|---|---|---|
| untouched | 0.5328 | 0.4260 |
| every letter swapped for an unseen one | 0.5319 | 0.4221 |
| letters swapped **and** spaces removed | 0.5271 | 0.4124 |

Swapping the entire alphabet costs **0.0009**. The program has no memorised
Greek to lose, which is the whole point of how it was built —
[description.md](description.md) explains the design rule that makes this true.

---

## If something goes wrong

**`could not locate the data files`** — Step 5 did not run, or you ran Step 6
from the wrong folder. Check that `CST_DATA` exists next to `solution.py` and
holds `train_messages.csv`.

**`ModuleNotFoundError: No module named 'lightgbm'`** — Step 4 did not finish.
Re-run it. The program will fall back to a slower scikit-learn model if
LightGBM genuinely cannot be installed, but LightGBM is what the scores above
were measured with.

**It seems to hang around "training rows"** — that step really does take five
or six minutes on a slow machine. It is building 357,868 rows of features. Let
it be.

**The output is not exactly the numbers above** — it should be. The program is
deterministic: same seeds, a fixed thread count, stable sorts. Running it twice
gives byte-identical files. If it does not, tell whoever sent you this link.

---

## Files

| file | what it is |
|---|---|
| `solution.py` | the whole thing — trains and predicts, no other code needed |
| `check_submission.py` | format validator, run it before submitting |
| `description.md` | how the method works and what was tried and rejected |
| `requirements.txt` | the four libraries |
| `submission.csv` | the output of the reference run, so you can compare |
| `CST.zip` | the dataset |
| `DATASET_ATTRIBUTION.txt` | where the data comes from and its licence |
