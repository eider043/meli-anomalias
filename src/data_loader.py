import pandas as pd
from .config import DATA_PATH, DATE_COL

def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values(DATE_COL)
