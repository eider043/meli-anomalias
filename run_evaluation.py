import pandas as pd
from src.evaluation import compute_metrics
from src.bootstrap_ab import bootstrap_f1
from pathlib import Path

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def eval_file(path: Path, tag: str):
    df = pd.read_csv(path)

    y_true = df["label_gt"].values
    y_pred_llm = df["label_llm"].values
    y_pred_stat = df["label_stat"].values

    metrics_llm = compute_metrics(y_true, y_pred_llm)
    metrics_stat = compute_metrics(y_true, y_pred_stat)

    boot = bootstrap_f1(y_true, y_pred_llm, y_pred_stat)

    results = pd.DataFrame([
        ["LLM", metrics_llm["f1"], metrics_llm["precision"], metrics_llm["recall"], df["latency_llm"].mean()],
        ["Modelo Estadístico", metrics_stat["f1"], metrics_stat["precision"], metrics_stat["recall"], 0.0],
    ], columns=["Modelo","F1","Precision","Recall","Latencia Promedio"])

    results.to_csv(OUT_DIR / f"metrics_table_{tag}.csv", index=False)

    with open(OUT_DIR / f"bootstrap_{tag}.json", "w", encoding="utf-8") as f:
        import json
        f.write(json.dumps(boot, ensure_ascii=False, indent=2))

    print(f"\n=== MÉTRICAS ({tag}) ===")
    print(results)

    print(f"\n=== BOOTSTRAP A/B ({tag}) ===")
    print(boot)

def main():
    # Head-to-head
    p1 = OUT_DIR / "predictions_h2h.csv"
    if p1.exists():
        eval_file(p1, "h2h")
    else:
        print("No existe outputs/predictions_h2h.csv (ejecuta main.py con RUN_MODE head_to_head/both)")

    # Prefiltro producción
    p2 = OUT_DIR / "predictions_prefilter.csv"
    if p2.exists():
        eval_file(p2, "prefilter")
    else:
        print("No existe outputs/predictions_prefilter.csv (ejecuta main.py con RUN_MODE prefilter/both)")

if __name__ == "__main__":
    main()

print("\n-> run_evaluation.py finalizó OK.")