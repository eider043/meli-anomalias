from src.data_loader import load_data
from src.model_statistical import detect_anomalies_stat
from src.model_llm import llm_predict
from src.ground_truth import build_ground_truth
from src.config import (
    ITEM_COL, PRICE_COL,
    RUN_MODE,
    H2H_N_TOTAL, H2H_POS_FRAC, H2H_MIN_HIST,
    MAX_LLM_CALLS_PROD,   
    LOG_EVERY_N_CALLS     
)

import pandas as pd
from pathlib import Path

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_llm_predictions(df: pd.DataFrame, only_candidates: bool) -> pd.DataFrame:
    """
    Agrega: label_llm, confidence_llm, reason_llm, latency_llm
    Si only_candidates=True: solo llama LLM cuando label_stat == ANOMALO
    """
    df = df.sort_values([ITEM_COL, "ORD_CLOSED_DT"]).reset_index(drop=True)

    # defaults
    df["label_llm"] = "NORMAL"
    df["confidence_llm"] = 0.0
    df["reason_llm"] = "No evaluado por LLM."
    df["latency_llm"] = 0.0

    cand_set = set(df.index[df["label_stat"].eq("ANOMALO")]) if only_candidates else None

    calls = 0  # <-- contador global de llamadas LLM

    for item, group in df.groupby(ITEM_COL, sort=False):
        group = group.sort_values("ORD_CLOSED_DT")
        prices = group[PRICE_COL].tolist()
        idxs = group.index.tolist()

        for j, idx in enumerate(idxs):
            if only_candidates and (idx not in cand_set):
                df.loc[idx, "reason_llm"] = "No evaluado por LLM (no candidato por stat)."
                continue

            hist_prices = prices[:j]
            if len(hist_prices) < H2H_MIN_HIST:
                df.loc[idx, "label_llm"] = "NORMAL"
                df.loc[idx, "confidence_llm"] = 0.2
                df.loc[idx, "reason_llm"] = f"Historial insuficiente (<{H2H_MIN_HIST})."
                df.loc[idx, "latency_llm"] = 0.0
                continue

            # ---- llamada LLM ----
            label, conf, reason, lat = llm_predict(df.loc[idx, PRICE_COL], hist_prices)
            df.loc[idx, "label_llm"] = label
            df.loc[idx, "confidence_llm"] = conf
            df.loc[idx, "reason_llm"] = reason
            df.loc[idx, "latency_llm"] = lat

            calls += 1
            if calls % LOG_EVERY_N_CALLS == 0:
                print(f"[LLM] calls={calls} | last_item={item} | last_price={df.loc[idx, PRICE_COL]}")

    print(f"[LLM] Done. total_calls={calls}")
    return df


def make_head_to_head_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset estratificado:
      - positivos: label_stat == ANOMALO
      - negativos: label_stat == NORMAL
    """
    df = df.copy()

    pos = df[df["label_stat"].eq("ANOMALO")]
    neg = df[df["label_stat"].eq("NORMAL")]

    n_pos = int(H2H_N_TOTAL * H2H_POS_FRAC)
    n_neg = H2H_N_TOTAL - n_pos

    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))

    sub = pd.concat([
        pos.sample(n_pos, random_state=42) if n_pos > 0 else pos.head(0),
        neg.sample(n_neg, random_state=42) if n_neg > 0 else neg.head(0),
    ], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    return sub


def cap_prefilter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limita la cantidad de candidatos que llegan al LLM en modo prefilter,
    para que el demo sea ejecutable en tiempo razonable.

    Estrategia:
    - ordenar por z_score descendente (más extremos primero) si existe
    - quedarnos con top MAX_LLM_CALLS_PROD candidatos
    - al resto les ponemos label_stat="NORMAL" para que no sean candidatos
    """
    df = df.copy()

    cand_mask = df["label_stat"].eq("ANOMALO")
    cand_df = df[cand_mask]

    if len(cand_df) <= MAX_LLM_CALLS_PROD:
        print(f"[PREFILTER] Candidates within cap: {len(cand_df)} <= {MAX_LLM_CALLS_PROD}")
        return df

    # Ordenar por score 
    if "z_score" in df.columns:
        cand_df = cand_df.reindex(cand_df["z_score"].abs().sort_values(ascending=False).index)
    else:
        cand_df = cand_df.sort_index()

    keep_idx = cand_df.index[:MAX_LLM_CALLS_PROD]
    drop_idx = df.index.difference(keep_idx)

    # Apaga candidatos excedentes
    df.loc[drop_idx, "label_stat"] = "NORMAL"

    print(f"[PREFILTER] Capped candidates to {MAX_LLM_CALLS_PROD} for demo. (original={len(cand_df)})")
    return df


def main():
    df = load_data()

    # 1) Ground Truth proxy
    df = build_ground_truth(df)

    # 2) Baseline estadístico
    df = detect_anomalies_stat(df)

    # ----------- MODO A: Head-to-head -----------
    if RUN_MODE in ("head_to_head", "both"):
        h2h = make_head_to_head_subset(df)
        print(f"[H2H] Subset size={len(h2h)} | Pos(frac stat)= {(h2h['label_stat']=='ANOMALO').mean():.2%}")

        h2h = add_llm_predictions(h2h, only_candidates=False)
        h2h.to_csv(OUT_DIR / "predictions_h2h.csv", index=False)

        llm_calls = (h2h["latency_llm"] > 0).sum()
        print(f"[H2H] LLM calls={llm_calls}/{len(h2h)} ({llm_calls/len(h2h):.2%})")
        print("[H2H] Saved -> outputs/predictions_h2h.csv")

    # ----------- MODO B: Prefiltro producción -----------
    if RUN_MODE in ("prefilter", "both"):
        print(f"[PREFILTER] Total rows={len(df)}")
        cand = (df["label_stat"] == "ANOMALO").sum()
        print(f"[PREFILTER] Candidates(stat)= {cand} ({cand/len(df):.2%})")

        # -> CAP DE CANDIDATOS PARA DEMO
        df_capped = cap_prefilter_candidates(df)

        prod = add_llm_predictions(df_capped, only_candidates=True)
        prod.to_csv(OUT_DIR / "predictions_prefilter.csv", index=False)

        llm_calls = (prod["latency_llm"] > 0).sum()
        avg_lat = prod.loc[prod["latency_llm"] > 0, "latency_llm"].mean() if llm_calls else 0.0
        savings = 1 - (llm_calls / len(prod))

        print(f"[PREFILTER] LLM calls={llm_calls} ({llm_calls/len(prod):.2%}) | avg_latency={avg_lat:.3f}s | savings_vs_full={savings:.2%}")
        print("[PREFILTER] Saved -> outputs/predictions_prefilter.csv")


if __name__ == "__main__":
    main()
