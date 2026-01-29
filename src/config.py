DATA_PATH = r"C:\Users\Usuario\Downloads\DPML\meli-anomalias\data\precios_historicos.csv"

DATE_COL = "ORD_CLOSED_DT"
ITEM_COL = "ITEM_ID"
PRICE_COL = "PRICE"

ANOMALY_THRESHOLD_Z = 3.0   # Modelo estadístico
BOOTSTRAP_ITER = 200
RANDOM_SEED = 42

# --- Experimentos ---
RUN_MODE = "prefilter"   # "head_to_head" | "prefilter" | "both"

# --- Head-to-head subset ---
H2H_N_TOTAL = 400          # tamaño total del subset (ajusta según tiempo)
H2H_POS_FRAC = 0.50         # fracción candidatos (stat=ANOMALO) dentro del subset
H2H_MIN_HIST = 5            # mínimo historial para permitir LLM

# --- Prefiltro producción ---
PREFILTER_ONLY_CANDIDATES = True
MAX_HIST_LLM = 30           # ya lo usas en model_llm

MAX_LLM_CALLS_PROD = 200  # demo rápido
LOG_EVERY_N_CALLS = 25

ROLL_WINDOW = 30    # ventana móvil -  últimos 30 precios previos)
MIN_HIST = 10       # mínimo historial para decidir

GT_ROLL_WINDOW = 60
GT_MIN_HIST = 15
GT_THRESHOLD_Z = 3.5


