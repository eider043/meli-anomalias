import numpy as np
import pandas as pd
from .config import ITEM_COL, PRICE_COL, DATE_COL, GT_ROLL_WINDOW, GT_MIN_HIST, GT_THRESHOLD_Z

def _rolling_mad(x: pd.Series) -> float:
    med = np.median(x)
    return np.median(np.abs(x - med))

def build_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([ITEM_COL, DATE_COL]).reset_index(drop=True)
    df["label_gt"] = "NORMAL"

    for item, g in df.groupby(ITEM_COL, sort=False):
        idx = g.index
        s = g[PRICE_COL].astype(float)

        roll = s.shift(1).rolling(window=GT_ROLL_WINDOW, min_periods=GT_MIN_HIST)
        med = roll.median()
        mad = roll.apply(_rolling_mad, raw=False) + 1e-9

        z = 0.6745 * (s - med) / mad

        df.loc[idx, "label_gt"] = np.where(
            (roll.count() >= GT_MIN_HIST) & (z.abs() > GT_THRESHOLD_Z),
            "ANOMALO",
            "NORMAL",
        )

    return df
