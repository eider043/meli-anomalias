import os, json, time, re
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")

def safe_json_extract(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except:
        return {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
def llm_predict(current_price: float, hist_prices, max_hist: int = 30):

    hist = list(hist_prices)[-max_hist:]
    hist_str = ", ".join([f"{p:.2f}" for p in hist])

    prompt = f"""
Eres un detector de anomalías de precios.
Responde SOLO con JSON:

{{
 "label":"ANOMALO|NORMAL",
 "confidence":0.0-1.0,
 "reason":"explicación corta"
}}

Historial: [{hist_str}]
Precio actual: {current_price:.2f}
JSON:
"""

    t0 = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    latency = time.time() - t0
    raw = response.choices[0].message.content
    parsed = safe_json_extract(raw)

    label = parsed.get("label","NORMAL")
    conf = float(parsed.get("confidence",0.5))
    reason = parsed.get("reason","")

    label = "ANOMALO" if str(label).upper().startswith("A") else "NORMAL"
    conf = max(0,min(1,conf))

    return label, conf, reason, latency
