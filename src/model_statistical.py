import numpy as np
import pandas as pd
from .config import ITEM_COL, PRICE_COL, DATE_COL, ROLL_WINDOW, MIN_HIST, ANOMALY_THRESHOLD_Z

C = 0.6745

def _rolling_mad(x: pd.Series) -> float:
    med = np.median(x)
    return np.median(np.abs(x - med))

def detect_anomalies_stat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([ITEM_COL, DATE_COL]).reset_index(drop=True)

    df["label_stat"] = "NORMAL"
    df["z_score"] = np.nan
    df["stat_med_roll"] = np.nan
    df["stat_mad_roll"] = np.nan
    df["stat_lower"] = np.nan
    df["stat_upper"] = np.nan
    df["stat_count"] = np.nan

    for item, g in df.groupby(ITEM_COL, sort=False):
        idx = g.index
        s = g[PRICE_COL].astype(float)

        # solo pasado
        roll = s.shift(1).rolling(window=ROLL_WINDOW, min_periods=MIN_HIST)

        med = roll.median()
        mad = roll.apply(_rolling_mad, raw=False) + 1e-9
        cnt = roll.count()

        z = C * (s - med) / mad

        # umbrales en términos de precio
        delta = (ANOMALY_THRESHOLD_Z * mad) / C
        lower = med - delta
        upper = med + delta

        # guardar para plot
        df.loc[idx, "z_score"] = z
        df.loc[idx, "stat_med_roll"] = med
        df.loc[idx, "stat_mad_roll"] = mad
        df.loc[idx, "stat_lower"] = lower
        df.loc[idx, "stat_upper"] = upper
        df.loc[idx, "stat_count"] = cnt

        df.loc[idx, "label_stat"] = np.where(
            (cnt >= MIN_HIST) & (z.abs() > ANOMALY_THRESHOLD_Z),
            "ANOMALO",
            "NORMAL",
        )

    return df