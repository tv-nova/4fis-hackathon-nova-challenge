# Predict the Prime-Time Audience

Every evening, TV channels compete for the same viewers. Your job is to predict
**how big a slice of the audience channel 3 gets in a given timeslot**, for a period
you have never seen.

The data comes from real measurements, so it is not a textbook dataset.
Some of the data issues were there from the start and some we added deliberately. Finding it and
dealing with it is part of the challenge.

Some features are categorical, some features *can* be categorical, numerical features can have a nice distribution and 
an ugly one.

## The task

Build a model that predicts `share_15_54` of **channel 3** for corresponding rows in
`data/test.csv`.

`share_15_54` is the **audience share in the 15–54 age group**: of all people aged 15–54
watching TV during that timeslot, the percentage watching this channel. It is a number
between 0 and 100.

- **Training data:** 1 Jan 2022 – 31 Oct 2023, labelled, all four channels
- **Test data:** 1 Nov 2023 – 13 Jan 2024, unlabelled, all four channels (yes, that
  includes Christmas)

**Only channel 3 rows are scored.** Rows of channels 4, 9 and 54 are context: their
history in train, and their schedule in test (programme identifiers, 3-month averages),
may help you understand channel 3's competition. You never get competitors' actual
shares for the test period, just as you wouldn't when forecasting for real. Deciding how
to use the other channels is part of the challenge.

### What the data covers

- **Four selected channels:** IDs 3, 4, 9 and 54. Viewers can also watch many other
  channels that are not in the data, so the four shares in a timeslot do **not** add up
  to 100.
- **Evenings only:** timeslots start between 16:00 and 23:55. There is no daytime or
  night data, and you only predict evening slots.

## Files

> **The data is not in this repository.** Download `train.csv` and `test.csv` from
> **https://docsend.com/view/s/48mikrrtrgca8uf4** and put them in `data/`. Both files are
> git-ignored: never commit them, and don't share them outside your team, including in
> the repository you show us.

| File | What it is |
|---|---|
| `data/train.csv` | Labelled history of all four channels, released as-is (download separately) |
| `data/test.csv` | Rows to predict (download separately), all four channels. Same columns as train minus `share_15_54`, plus `row_id`. Only channel 3 rows are scored |
| `template/` | Skeleton of what you hand in: `solution.py` + `requirements.txt` |
| `check_submission.py` | Runs your solution exactly the way the organisers will |

Each row is one **(timeslot, channel)** pair.

| Column | Meaning |
|---|---|
| `row_id` | Test only. Used by the organisers to match predictions to answers |
| `main_ident` | Identifier of the programme on air. One programme covers several timeslots. Look closely at how it is built. |
| `timeslot_datetime_from` | Start of the timeslot (local time) |
| `channel_id` | The channel the row is about: 3, 4, 9 or 54 |
| `share_15_54` | **Target.** Train only |
| `share_15_54_3mo_mean` | A ready-made 3-month average of the share. You may use it, but it is not a free win. |
| `ch{C}__f_{1..12}` | 12 anonymised features describing what is on **each** of the four channels at that moment. They are identical for all rows of the same timeslot. |

So every row tells you about its own channel **and** about what its competitors are
showing at the same time. The `f_*` features are anonymised: some are
numeric, some categorical (`f_10` and `f_11`). Working out what they mean is part of the
fun.

## What you hand in

You submit **code**, not predictions, and we run it. Your submission is a folder with a `solution.py`
defining:

```python
class Model:
    def __init__(self): ...                                    # no arguments; fix your random seeds
    def fit(self, train: pd.DataFrame) -> None: ...           # data/train.csv, raw, as read by pandas
    def predict(self, test: pd.DataFrame) -> np.ndarray: ...  # test.csv without row_id -> one value per row
```

plus a `requirements.txt` listing every package you use. Start from `template/`. Put helper
modules and any external data files you use in the same folder, and load them relative to
`solution.py`, not to the current working directory. We start your code from a different
folder, so `pd.read_csv("weather.csv")` fails on our side:

```python
from pathlib import Path
HERE = Path(__file__).parent
weather = pd.read_csv(HERE / "external" / "weather.csv")   # works wherever we run it from
```

`check_submission.py` also runs your code from a different folder, so it catches this
mistake before you send anything.

`predict()` receives **all** test rows, so you can use the competitor rows as context. It
must return a number for every row, but values for rows of channels 4, 9 and 54 are
ignored; returning 0 for them is fine.

We run every submission the same way:

1. In a clean Linux container with **Python 3.12**, we install `pandas`, `numpy` and your
   `requirements.txt`. Unpinned packages get their newest version, which may not be the
   one you developed with: pandas 3, for example, reads some columns with different types
   than pandas 2. **Pin your versions** (e.g. `pandas==2.2.3`).
2. With **the internet switched off**, we call `Model().fit(train)` and then
   `model.predict(test)`.

| Rule | Why |
|---|---|
| All learning happens inside `fit()`. Pre-trained model weights are not allowed; bundled data files are fine. | We train your model ourselves |
| `predict()` returns one number per input row, in the same order. It must not depend on row order: we shuffle the rows and check. | Robustness |
| CPU only (4 cores, 6 GB RAM), no internet during `fit`/`predict`, at most **4 minutes** for `fit` + `predict` together. `predict()` runs a second time for the shuffle check, and any run still going after 6 minutes in total (loading, `fit` and both `predict` calls) is stopped. Tune hyperparameters offline and hard-code them; don't run a search inside `fit()`. Downloads belong in `requirements.txt` or in bundled files. | Fair and reproducible |
| Read data only from the DataFrames you are given; don't open `data/` yourself | We pass the data in |
| Don't hard-code dates, row counts or column positions. | Robustness tests, see below |

Before sending, run exactly what we run:

```bash
pip install pandas numpy -r my_solution/requirements.txt
python check_submission.py my_solution/
```

If it prints `OK`, we can run you. **If it fails on our side, the submission scores
nothing.** So test it in a fresh virtual environment, not only in your notebook kernel.

### Robustness tests (separate from the automatic score)

After the main run, we put your `Model` through a second, separate set of hidden tests.
They **never change your automatic score**: a crash there does not cost you the
performance part. Their results are a report for the jury's robustness points.

We don't publish what the tests are. Build your model as if it were going to run in
production on real, imperfect data, and test it yourself on your own validation splits.

## How to submit

Email **one zip** per team to **yevhenii.kubov@nova.cz**:

- **Subject:** `[4FIS Hackathon] <team name>`
- **`<team name>.zip`:** your solution folder (`solution.py`, `requirements.txt`, helpers, data
  files). Keep it under 20 MB; if it is bigger, send a download link instead.
- **Slides**, as PDF, at most 10 slides: data problems you found, your approach, what
  drives the share, and where the model fails
- **Optionally**, a link to your repository with notebooks you want to show off

Only the **last email before the deadline** counts.

### Optional dry run

Send your zip before the **dry-run deadline** announced by the organisers, with the
subject `[4FIS Hackathon] <team name> - dry run`. Shortly after, we reply with whether it ran
on our side and its score on **part** of the test period. It is the only score you will see before the end, so use it to catch problems,
not to tune.

## Evaluation

**Metric:** mean absolute error (MAE) of channel 3's share, in share points, over the
whole test set. Lower is better.

For measuring the model performance during the implementation, build your own validation with the given data.
Remember that this is a forecast into the future, so think about which kind of split makes sense.

### Final score

| Part | Weight | How it is scored |
|---|---|---|
| Performance | **50%** | Test MAE, scaled to 0–100 points between a simple non-ML **Baseline** (0 points) and the best team (100 points) |
| Jury | **50%** | Creativity, robustness, insight and optional extras such as prediction confidence. See below. |

The Baseline is a simple heuristic that uses no machine learning. We don't publish its
code or score.

### What the jury rewards (50% weight)

The following points are not evaluated automatically.

- **Creativity & ambition (20%).** Original features; clever use of the competitor
  rows, the other channels' history or the identifier; useful external data; 
  unusual or advanced modelling ideas. Bold experiments that did not
  work count too, if you measured them properly and can explain why they failed. Show
  what you tried in a small table: idea, validation score with vs without it.
- **Robustness (10%).** Did you find and handle the data problems? Is your validation
  honest? Does the model hold up **across the whole evening and in every week**, not just
  on average? We compute MAE for the early evening, prime time and late evening, plus the
  worst-week MAE, for every submission. Does your code run cleanly the first time? How
  does it do in the **robustness tests**?
  Imagine that you are building a model that must be trusted in production
- **Insight (10%).** What drives the share? Where does your model fail, and why? What
  would you tell channel 3's programming team (those who create TV program schedule)?
  Explain the model, not just its score.
- **Optional extras (10%).** Anything that makes the forecast more useful in practice.
  For example:
  - **Prediction confidence:** a range such as "80% chance the share is between 17% and
    26%" for each prediction, with evidence on your own validation data that the range is
    right about 80% of the time, including in unusual weeks like Christmas. A broadcaster
    selling guaranteed audiences to advertisers needs exactly this.
  - **Explanations per prediction:** why the model expects a high or low share for a
    given slot.
  - **A small prediction API or container** around your `Model`.

  Extras are shown in your slides, notebooks or repository; they are not part of the
  automatic score. `predict()` still returns one number per row.

## Rules

- Explore in any language. The code you hand in must be Python following the `Model`
  interface. Any library is fine, and so are AI assistants.
- **External data is allowed** if it is public, free, bundled with your
  code, named in your slides, and does not introduce something from the future (points can be deducted in case of data leaks).
  Examples: public holidays, school breaks,
  weather, daylight, big sports events. These are ideas, not promises.
  You can measure whether they help.
  - **Bundle it as files.** The internet is off while we run your code, so download the
    data during the hackathon and put it in your solution folder. Python packages that
    work offline are fine too (e.g. `holidays`).
  - **Cover the whole timeline:** Jan 2022 – Jan 2024, not only the test period.
- **Not allowed:**
  - any TV audience measurement data (ratings or share tables, published audience
    figures, press articles quoting them), for any period
  - copies of this dataset from anywhere other than the official download link above
  - publishing or sharing the dataset files, during or after the hackathon
  - hard-coded or hand-made predictions
  - code that tries to read anything outside its own folder and the data it is given

## Getting started

1. Download `train.csv` and `test.csv` from https://docsend.com/view/s/48mikrrtrgca8uf4
   into `data/`.
2. Set up the environment and your solution folder:

```bash
pip install pandas numpy
cp -r template my_solution        # then implement fit() and predict()
python check_submission.py my_solution/
```

Load `data/train.csv` and look at it for a long time before modelling.

Good luck, and have fun.
