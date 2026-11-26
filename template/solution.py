"""Your team's final solution.

Keep the class name `Model` and the two method signatures. Everything else is up to you:
add helper modules next to this file, bundle external data files, and so on.

Copy this folder (e.g. to my_solution/), implement fit() and predict(), then check it with:
    python check_submission.py my_solution/
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Folder of this file. Load bundled files from here, e.g. HERE / "external" / "weather.csv".
# We run your code from a different working directory, so plain relative paths would fail.
HERE = Path(__file__).parent


class Model:
    def __init__(self) -> None:
        # No arguments. Fix your random seeds here so a rerun gives the same predictions.
        self.seed = 42

    def fit(self, train: pd.DataFrame) -> None:
        """Learn everything here.

        train: DataFrame (n_rows, 53), data/train.csv as `pd.read_csv(path, low_memory=False)`
               returns it: raw, with duplicates, gaps and all. Any cleaning is your job.
        """
        raise NotImplementedError("train your model here")

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Return one predicted share_15_54 per row of `test`, in the same order.

        test:    DataFrame (n_rows, 52), the train columns minus `share_15_54`, no `row_id`.
                 All four channels; only channel 3 rows are scored. Rows may arrive in any
                 order, so do not rely on the order or on the index.
        returns: float array (n_rows,), no NaN. Values for channels 4, 9, 54 are ignored
                 but must still be numbers (0 is fine).
        """
        raise NotImplementedError("predict here")
