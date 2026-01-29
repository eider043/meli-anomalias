# Desafío Técnico Mercado Libre — Ejercicio 1 (Anomalías de Precio)

Este repositorio contiene la solución del **Ejercicio 1** del desafío técnico: detección de anomalías en series históricas de precios por `ITEM_ID`, comparando un enfoque **LLM** vs un enfoque **estadístico robusto no supervisado**, incorporando evaluación cuantitativa y prueba A/B mediante bootstrap.

Adicionalmente, se implementa una **arquitectura escalable de prefiltro**, donde el modelo estadístico reduce las llamadas al LLM, simulando un escenario productivo realista.

---

## Dataset

Archivo: `precios_historicos.csv`  
Columnas:
- `ITEM_ID`
- `ORD_CLOSED_DT` (fecha de cierre)
- `PRICE` (precio)

> **Nota:** El dataset no se sube al repositorio.  
> Debe ubicarse localmente y configurarse la ruta en `src/config.py`.

---

## Enfoque

Se construyen dos modelos:

### Modelo estadístico (baseline no supervisado)

- Método: estadística robusta basada en **mediana y MAD (Median Absolute Deviation)** por producto.
- Se calcula un **z-score robusto móvil** sobre el historial reciente de precios.
- Criterio: se etiqueta como **ANOMALO** si `|z_MAD| > umbral`.

Este modelo actúa como:
- Baseline de comparación.
- Prefiltro productivo para reducir llamadas al LLM.

---

### Modelo LLM

- Recibe como entrada:
  - Precio actual.
  - Historial de precios previos del mismo producto.
- Retorna:
  - `label ∈ {ANOMALO, NORMAL}`
  - `confidence ∈ [0,1]`
  - `reason` (justificación breve)
  - `latency` (tiempo de inferencia)

La arquitectura queda preparada para utilizar un LLM real vía API, manteniendo un contrato de salida estructurado.

---

## Ground Truth (para evaluación)

El dataset no contiene etiquetas reales.  
Para evaluación reproducible se define un **ground truth proxy estadístico robusto**:

- Por cada producto:
  - Se calcula z-score MAD global.
  - `ANOMALO` si `|z_MAD| > umbral_gt`.

Este enfoque es estándar en problemas no supervisados y permite medir F1, Precision y Recall.

---

## Arquitectura funcional

```mermaid
flowchart TD
    A[precios_historicos.csv] --> B[Carga y ordenamiento por ITEM_ID]
    B --> C[Ground Truth proxy (MAD global)]
    B --> D[Modelo estadístico MAD móvil]

    D --> H2H[Modo Head-to-Head]
    D --> PF[Modo Prefiltro productivo]

    H2H --> LLM1[LLM evalúa mismas filas]
    PF --> LLM2[LLM evalúa solo candidatos]

    C --> E[Evaluación]
    D --> E
    LLM1 --> E
    LLM2 --> E

    E --> F[Métricas + Bootstrap A/B]
    F --> G[Tablas + Gráficos + Conclusión]
```
---

## Estructura del proyecto

```
meli-anomalias/
│
├── data/
│   └── precios_historicos.csv
│
├── notebooks/
│   └── 01_exploracion.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── ground_truth.py
│   ├── model_statistical.py
│   ├── model_llm.py
│   ├── evaluation.py
│   └── bootstrap_ab.py
│
├── outputs/
│   ├── predictions_h2h.csv
│   ├── predictions_prefilter.csv
│   ├── metrics_table_h2h.csv
│   ├── metrics_table_prefilter.csv
│   ├── bootstrap_h2h.json
│   ├── bootstrap_prefilter.json
│   └── product_*.png
│
├── main.py
├── run_evaluation.py
├── plot_examples.py
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Ejecución

```bash
python main.py
python run_evaluation.py
python plot_examples.py
```

---

## Resultados obtenidos

### Head-to-Head

| Modelo | F1 | Precision | Recall | Latencia Promedio |
|--------|----|-----------|--------|-------------------|
| LLM | 0.520 | 0.364 | 0.910 | 1.16 s |
| Modelo Estadístico | 0.793 | 0.660 | 0.992 | ~0 s |

**Bootstrap A/B:**

- ΔF1 ≈ -0.271  
- IC95% ≈ [-0.313 , -0.220]  
- p-value ≈ 1.0  

Interpretación: El intervalo de confianza es completamente negativo y el p-value indica que el LLM **no supera** al modelo estadístico. La diferencia observada favorece al modelo estadístico de manera consistente.
---

## Conclusión ejecutiva

El modelo estadístico robusto basado en mediana y MAD obtuvo mejor desempeño que el LLM en detección de anomalías de precio. En el experimento head-to-head alcanzó F1=0.793 frente a 0.520 del LLM, con mayor precisión y recall casi perfecto. El análisis bootstrap confirmó que la diferencia de F1 promedio Δ≈-0.27 es estadísticamente significativa (IC95% completamente negativo, p-value≈1.0), evidenciando que el LLM no mejora al baseline. En términos operativos, el modelo estadístico ofrece detección estable, sin latencia y sin costo computacional adicional. El LLM aporta interpretabilidad textual, pero introduce latencia (~1.16 s por predicción) y costo de API sin ganancia en desempeño. Se recomienda usar el modelo estadístico como detector principal y reservar el LLM solo como herramienta de apoyo explicativo o revisión manual de casos críticos.

