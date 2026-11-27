"""Check your solution before you hand it in. Organisers run exactly this.

    python check_submission.py path/to/your_solution/ [--out submission.csv]

The folder must contain `solution.py` with a `Model` class (see template/solution.py)
and a `requirements.txt`. The checker:
  - trains a fresh Model on data/train.csv          (Model().fit(train))
  - predicts data/test.csv without row_id            (model.predict(test))
  - checks the output: one finite number per row, in the same order as `test`
  - predicts a shuffled copy of test and checks the predictions follow the rows
  - checks that fit + predict take at most 4 minutes
  - writes the predictions to submission.csv (or --out) and validates them
Your code runs from an empty temporary folder, as on the organisers' side.

Only pandas and numpy are needed to run this file.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "share_15_54"
DEFAULT_DATA = Path(__file__).resolve().parent / "data"
TIME_LIMIT_MIN = 4
SHUFFLE_TOL = 1e-5


class SubmissionError(ValueError):
    """The submission breaks the rules and cannot be scored."""


# --------------------------------------------------------------------------- CSV checks
def validate_submission(sub: pd.DataFrame, expected_row_ids) -> pd.Series:
    """Validate a submission DataFrame. Returns predictions ordered like expected_row_ids."""
    missing_cols = {"row_id", TARGET} - set(sub.columns)
    if missing_cols:
        raise SubmissionError(f"missing column(s) {sorted(missing_cols)}; expected exactly: row_id,{TARGET}")
    extra = set(sub.columns) - {"row_id", TARGET}
    if extra:
        raise SubmissionError(f"unexpected column(s) {sorted(extra)}; expected exactly: row_id,{TARGET}")
    if sub["row_id"].duplicated().any():
        raise SubmissionError(f"duplicated row_id, e.g. {sub.loc[sub.row_id.duplicated(), 'row_id'].head().tolist()}")
    expected = pd.Index(expected_row_ids)
    got = set(sub["row_id"])
    missing, unknown = set(expected) - got, got - set(expected)
    if missing:
        raise SubmissionError(f"{len(missing)} row_id(s) missing, e.g. {sorted(missing)[:5]}")
    if unknown:
        raise SubmissionError(f"{len(unknown)} unknown row_id(s), e.g. {sorted(unknown, key=str)[:5]}")
    pred = pd.to_numeric(sub.set_index("row_id")[TARGET], errors="coerce").reindex(expected)
    bad = ~np.isfinite(pred.to_numpy(dtype=float))
    if bad.any():
        raise SubmissionError(f"{bad.sum()} prediction(s) are empty, NaN, inf or not numeric, "
                              f"e.g. row_id {pred.index[bad][:5].tolist()}")
    return pred.reset_index(drop=True)


def read_submission(path: Path, expected_row_ids) -> pd.Series:
    try:
        sub = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        raise SubmissionError(f"cannot read {path} as CSV: {e}") from e
    return validate_submission(sub, expected_row_ids)


def range_warnings(pred: np.ndarray) -> list[str]:
    out = []
    if (pred < 0).any() or (pred > 100).any():
        out.append(f"{((pred < 0) | (pred > 100)).sum()} prediction(s) outside 0..100 (a share is a percentage)")
    if np.ptp(pred) == 0:
        out.append("all predictions are identical")
    return out


# --------------------------------------------------------------------------- code checks
HINTS = {
    FileNotFoundError: "load bundled files relative to solution.py: Path(__file__).parent / 'file.csv'",
    ModuleNotFoundError: "add the missing package to requirements.txt",
}


def call_team_code(what: str, fn, *args):
    """Run team code; turn any crash into a readable SubmissionError (traceback still printed)."""
    try:
        return fn(*args)
    except SubmissionError:
        raise
    except (Exception, SystemExit) as e:  # SystemExit: team code calling sys.exit()
        traceback.print_exc()
        hint = next((h for t, h in HINTS.items() if isinstance(e, t)), None)
        raise SubmissionError(f"{what} raised {type(e).__name__}: {e}" + (f"  (hint: {hint})" if hint else "")) from e


def load_model_class(solution_dir: Path):
    path = solution_dir / "solution.py"
    if not path.exists():
        raise SubmissionError(f"{path} not found")
    if not (solution_dir / "requirements.txt").exists():
        raise SubmissionError(f"{solution_dir / 'requirements.txt'} not found (list every package you import)")
    sys.path.insert(0, str(solution_dir.resolve()))  # allow helper modules next to solution.py
    sys.dont_write_bytecode = True  # no __pycache__ left in the folder you are about to zip
    spec = importlib.util.spec_from_file_location("team_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Model"):
        raise SubmissionError("solution.py must define a class named Model")
    return module.Model


def as_prediction_array(raw, n: int, what: str) -> np.ndarray:
    try:
        pred = np.asarray(raw, dtype=float).reshape(-1)
    except Exception as e:  # noqa: BLE001
        raise SubmissionError(f"{what} must return numbers (array, list or Series): {e}") from e
    if len(pred) != n:
        raise SubmissionError(f"{what} returned {len(pred)} values for {n} rows")
    if not np.isfinite(pred).all():
        raise SubmissionError(f"{what} returned {(~np.isfinite(pred)).sum()} NaN/inf value(s)")
    return pred


@contextlib.contextmanager
def foreign_working_directory():
    """Run team code from an empty temporary folder, as the organisers do (they start it in /tmp).
    Code that opens bundled files by a relative path like "weather.csv" fails here too,
    instead of only on the organisers' machine. Load files via Path(__file__).parent."""
    old = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="check_submission_")
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)  # ignore files the team code still holds open (Windows)


def run_code(solution_dir: Path, data_dir: Path, out: Path) -> list[str]:
    train = pd.read_csv(data_dir / "train.csv", low_memory=False)
    test_full = pd.read_csv(data_dir / "test.csv", low_memory=False)
    row_ids = test_full["row_id"].to_numpy()
    test = test_full.drop(columns="row_id")
    warnings = []

    Model = call_team_code("importing solution.py", load_model_class, solution_dir)
    t0 = time.perf_counter()
    model = call_team_code("Model()", Model)
    call_team_code("fit()", model.fit, train.copy())
    t_fit = time.perf_counter() - t0
    print(f"  fit        {t_fit:7.1f} s")

    t0 = time.perf_counter()
    pred = as_prediction_array(call_team_code("predict()", model.predict, test.copy()), len(test), "predict()")
    t_pred = time.perf_counter() - t0
    print(f"  predict    {t_pred:7.1f} s")

    perm = np.random.default_rng(0).permutation(len(test))
    shuffled = test.iloc[perm].reset_index(drop=True)
    pred_shuf = as_prediction_array(call_team_code("predict() on shuffled rows", model.predict, shuffled),
                                    len(test), "predict() on shuffled rows")
    diff = np.abs(pred_shuf - pred[perm]).max()
    if diff > SHUFFLE_TOL:
        raise SubmissionError(
            f"predictions change when test rows are shuffled (max diff {diff:.4g}). predict() must return "
            "one value per input row, in the same order, without relying on row order or the index")
    print(f"  shuffle    ok (max diff {diff:.1e})")

    total = t_fit + t_pred
    if total > TIME_LIMIT_MIN * 60:
        raise SubmissionError(f"fit + predict took {total / 60:.1f} min, limit is {TIME_LIMIT_MIN} min")

    sub = pd.DataFrame({"row_id": row_ids, TARGET: pred})
    sub.to_csv(out, index=False)
    read_submission(out, row_ids)  # the written file must itself be valid
    print(f"  wrote      {out}")
    warnings += range_warnings(pred)
    return warnings


# --------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solution_dir", type=Path, help="folder with solution.py and requirements.txt")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"),
                    help="where to write the predictions (default: submission.csv in the current folder)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="folder with train.csv and test.csv")
    args = ap.parse_args()

    try:
        paths = (args.solution_dir.resolve(), args.data_dir.resolve(), args.out.resolve())
        missing = [f for f in ("train.csv", "test.csv") if not (paths[1] / f).exists()]
        if missing:
            raise SubmissionError(f"{', '.join(missing)} not found in {paths[1]}. Download the data "
                                  "(link in README.md) into that folder, or pass --data-dir")
        if not paths[2].parent.is_dir():
            raise SubmissionError(f"--out: folder {paths[2].parent} does not exist")
        with foreign_working_directory():
            warnings = run_code(*paths)
    except SubmissionError as e:
        print(f"FAILED: {e}")
        return 1
    for w in warnings:
        print(f"WARNING: {w}")
    print("OK: ready to submit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
