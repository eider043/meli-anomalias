import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import ANOMALY_THRESHOLD_Z, ROLL_WINDOW, MIN_HIST  

INPUT = Path("outputs") / "predictions_prefilter.csv"  #  predictions_prefilter.csv   predictions_h2h.csv
df = pd.read_csv(INPUT)
df["ORD_CLOSED_DT"] = pd.to_datetime(df["ORD_CLOSED_DT"])

counts = df["ITEM_ID"].value_counts()
items = counts[counts >= 80].index[:3]

for item in items:
    sub = df[df["ITEM_ID"] == item].sort_values("ORD_CLOSED_DT").copy()

    plt.figure(figsize=(12, 4))
    plt.plot(sub["ORD_CLOSED_DT"], sub["PRICE"], marker="o", linewidth=1, markersize=3, label="Precio")

    # --- UMBRALES MÓVILES (STAT) ---
    # (solo se dibujan donde existan, al inicio suele haber NaN por MIN_HIST)
    plt.plot(sub["ORD_CLOSED_DT"], sub["stat_upper"], linewidth=1, linestyle="--",
             label=f"Umbral superior STAT (|z|>{ANOMALY_THRESHOLD_Z})")
    plt.plot(sub["ORD_CLOSED_DT"], sub["stat_lower"], linewidth=1, linestyle="--",
             label=f"Umbral inferior STAT (|z|>{ANOMALY_THRESHOLD_Z})")

    # Sombrear banda 
    plt.fill_between(
        sub["ORD_CLOSED_DT"],
        sub["stat_lower"],
        sub["stat_upper"],
        alpha=0.12,
        label=f"Banda normal (win={ROLL_WINDOW}, min_hist={MIN_HIST})"
    )

    # -> LLM
    anom_llm = sub[sub["label_llm"] == "ANOMALO"]
    plt.scatter(anom_llm["ORD_CLOSED_DT"], anom_llm["PRICE"], s=40, label="LLM Anomalía")

    # -> STAT
    anom_stat = sub[sub["label_stat"] == "ANOMALO"]
    plt.scatter(anom_stat["ORD_CLOSED_DT"], anom_stat["PRICE"], marker="x", s=60, label="STAT Anomalía")

    plt.title(f"Producto {item} — Umbrales móviles MAD z-score")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/product_{item}.png")
    plt.close()
