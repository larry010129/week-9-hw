import numpy as np
import pandas as pd


class GroupEstimate(object):
    """Predict a continuous value from the group a categorical row falls in."""

    def __init__(self, estimate: str):
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.columns_ = None
        self.group_estimates_ = None

    def fit(self, X: pd.DataFrame, y):
        """Learn one estimate of y per group of X (X and y are not stored)."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        y = pd.Series(np.asarray(y).ravel())
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if y.isna().any():
            raise ValueError("y must not contain missing values")

        data = X.reset_index(drop=True).copy()
        data["__y__"] = y
        self.columns_ = list(X.columns)
        self.group_estimates_ = (
            data.groupby(self.columns_, observed=True)["__y__"]
            .agg(self.estimate)
        )
        return self

    def predict(self, X_) -> np.ndarray:
        """Return the estimate for each row of X_ (NaN for unseen groups)."""
        if self.group_estimates_ is None:
            raise RuntimeError("Call fit before predict")
        X_ = pd.DataFrame(X_)
        if X_.shape[1] != len(self.columns_):
            raise ValueError(f"X_ needs {len(self.columns_)} columns")
        X_.columns = self.columns_

        index = pd.MultiIndex.from_frame(X_)
        if len(self.columns_) == 1:
            index = pd.Index(X_.iloc[:, 0])
        result = self.group_estimates_.reindex(index).to_numpy(dtype=float)

        missing = int(np.isnan(result).sum())
        if missing:
            print(f"{missing} observation(s) belong to a group "
                  "not seen in fit.")
        return result
