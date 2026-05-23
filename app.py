# ============================================================
#  AFND — Arabic Fake News Detector  |  Flask Backend
#  app.py  — matches Colab training script exactly (no margin/unc logic)
# ============================================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import re
import os
from transformers import AutoTokenizer, AutoModel

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

app = Flask(__name__)
CORS(app)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "arabic_fakenews_stable")
MODEL_PATH    = os.path.join(MODEL_DIR, "best_model.pt")
HF_MODEL_NAME = "aubmindlab/bert-base-arabertv02"

MAX_LENGTH     = 192
MIN_WORDS      = 5
MAX_WORDS      = 180
THRESHOLD_REAL = 0.40

DROPOUT  = 0.30
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()


# ════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores  = self.attn(hidden_states).squeeze(-1).float()
        scores  = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).to(hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class FakeNewsDetector(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.pool       = AttentionPooling(hidden)
        self.dropout1   = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(hidden * 2)
        self.fc         = nn.Linear(hidden * 2, hidden)
        self.act        = nn.GELU()
        self.dropout2   = nn.Dropout(dropout / 2)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        try:
            out = self.encoder(**kwargs, token_type_ids=token_type_ids)
        except TypeError:
            out = self.encoder(**kwargs)
        last  = out.last_hidden_state
        cls_v = last[:, 0, :]
        atn_v = self.pool(last, attention_mask)
        x = torch.cat([cls_v, atn_v], dim=1)
        x = self.dropout1(x)
        x = self.norm(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout2(x)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ════════════════════════════════════════════════════════════

_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE    = re.compile(r"\S+@\S+\.\S+")
_SOCIAL_RE   = re.compile(r"[@#]\S+")
_DIACRITIC   = re.compile(r"[\u064B-\u065F\u0670]")
_TATWEEL     = re.compile(r"\u0640")
_ALEF        = re.compile(r"[إأآ]")
_FINAL_YAA   = re.compile(r"ى")
_HAMZA_YAA   = re.compile(r"ئ")
_BOILERPLATE = re.compile(
    r"(جميع\s*الحقوق\s*محفوظة|كل\s*الحقوق\s*محفوظة|تابعونا\s*على"
    r"|اشترك\s*في\s*نشرتنا|اقرأ\s*أيضا|انظر\s*أيضا)", re.IGNORECASE)
_NON_TEXT    = re.compile(r"[^\u0600-\u06FF0-9\s\.\,\!\?\:\;\-\(\)\"\'٪%]")
_SPACE       = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _SOCIAL_RE.sub(" ", text)
    text = _BOILERPLATE.sub(" ", text)
    text = _DIACRITIC.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _ALEF.sub("ا", text)
    text = _FINAL_YAA.sub("ي", text)
    text = _HAMZA_YAA.sub("ي", text)
    text = _NON_TEXT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()

def preprocess_arabic(title: str, body: str) -> tuple:
    title_c  = clean_text(title)
    body_c   = clean_text(body)
    body_c   = " ".join(body_c.split()[:MAX_WORDS])
    combined = f"{title_c} {body_c}".strip()
    return title_c, body_c, combined


# ════════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════════

print(f"\n{'='*55}")
print("  AFND — Loading model…")
print(f"  Device        : {device}")
print(f"  index 0=FAKE, index 1=REAL")
print(f"  Threshold REAL: {THRESHOLD_REAL}")
print(f"  Max words     : {MAX_WORDS}")
print(f"{'='*55}\n")

tok_source = MODEL_DIR if os.path.exists(
    os.path.join(MODEL_DIR, "tokenizer_config.json")) else HF_MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(tok_source)
model     = FakeNewsDetector(HF_MODEL_NAME).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("\n✓ Model ready.\n")


# ════════════════════════════════════════════════════════════
#  PREDICTION — identical to Colab training script
# ════════════════════════════════════════════════════════════

def predict(title: str, body: str = "", threshold_real: float = THRESHOLD_REAL) -> dict:

    model.eval()

    title_c, body_c, combined = preprocess_arabic(title.strip(), body.strip())
    wc = len(combined.split())

    raw_wc = len(f"{title.strip()} {body.strip()}".split())
    if raw_wc < MIN_WORDS:
        return {
            "error": f"النص قصير جداً ({raw_wc} كلمات). الحد الأدنى {MIN_WORDS} كلمات.",
            "word_count": raw_wc
        }

    enc = tokenizer(
        title_c or "[PAD]",
        body_c  or "[PAD]",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    ids   = enc["input_ids"].to(device)
    mask  = enc["attention_mask"].to(device)
    ttype = enc.get("token_type_ids", torch.zeros_like(ids)).to(device)

    with torch.no_grad():
        if USE_FP16:
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=True):
                logits = model(ids, mask, ttype)
        else:
            logits = model(ids, mask, ttype)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # ── Exact match to Colab lines 794-798 ───────────────────
    prob_fake  = float(probs[0])          # index 0 = FAKE
    prob_real  = float(probs[1])          # index 1 = REAL
    pred       = 1 if prob_real >= threshold_real else 0
    confidence = prob_real if pred == 1 else prob_fake

    # ── label: only "real" or "fake" — matches Colab exactly ─
    label = "real" if pred == 1 else "fake"

    # ── Danger level ─────────────────────────────────────────
    if label == "real" and confidence >= 0.85:
        danger_lbl, danger_steps = "منخفض جداً", 1
    elif label == "real":
        danger_lbl, danger_steps = "منخفض", 2
    elif confidence >= 0.82:
        danger_lbl, danger_steps = "مرتفع جداً", 5
    else:
        danger_lbl, danger_steps = "مرتفع", 4

    print(f"[predict] label={label} conf={confidence:.4f} "
          f"real={prob_real:.4f} fake={prob_fake:.4f} "
          f"words={wc}")

    return {
        "prediction":   "Real" if pred == 1 else "Fake",
        "label":        label,
        "confidence":   round(confidence, 4),
        "prob_fake":    round(prob_fake, 4),
        "prob_real":    round(prob_real, 4),
        "word_count":   wc,
        "danger_lbl":   danger_lbl,
        "danger_steps": danger_steps,
        "threshold":    threshold_real,
    }


# ════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict/text", methods=["POST"])
def predict_text():
    data           = request.get_json(force=True)
    raw_text       = data.get("body", "").strip()
    threshold_real = float(data.get("threshold", THRESHOLD_REAL))

    if not raw_text:
        return jsonify({"error": "الرجاء إرسال نص في حقل 'body'."}), 400

    # ── CRITICAL: pass as title="" body=text is WRONG ──────────
    # Colab always calls predict(title=news, body="")
    # so the text goes into segment A of the BERT tokenizer.
    # Passing as body puts it in segment B → different token_type_ids → wrong output.
    result = predict(title=raw_text, body="", threshold_real=threshold_real)
    return jsonify(result)


@app.route("/api/predict/url", methods=["POST"])
def predict_url():
    try:
        import requests as req
        from bs4 import BeautifulSoup
    except ImportError:
        return jsonify({"error": "شغّل: pip install requests beautifulsoup4 lxml"}), 501

    data           = request.get_json(force=True)
    url            = data.get("url", "").strip()
    threshold_real = float(data.get("threshold", THRESHOLD_REAL))

    if not url:
        return jsonify({"error": "الرجاء إرسال رابط في حقل 'url'."}), 400

    try:
        headers  = {"User-Agent": "Mozilla/5.0"}
        response = req.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.find("title")
        title = title.get_text(strip=True) if title else ""
        body  = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

        # Combine title + body into a single string, then pass as title only (body="")
        # This matches Colab exactly: predict(title=text, body="")
        # Passing body separately puts it in BERT segment B → wrong token_type_ids → wrong output
        combined_text = f"{title} {body}".strip()
        combined_text = " ".join(combined_text.split()[:MAX_WORDS])  # respect MAX_WORDS limit

        if len(combined_text.split()) < MIN_WORDS:
            return jsonify({"error": "لم يتم استخراج نص كافٍ من الرابط. جرّب لصق النص مباشرةً."}), 422

        result = predict(title=combined_text, body="", threshold_real=threshold_real)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"فشل تحميل الرابط: {str(e)}"}), 500


@app.route("/api/debug", methods=["POST"])
def debug():
    data     = request.get_json(force=True)
    raw_text = data.get("body", "").strip()
    if not raw_text:
        return jsonify({"error": "send body field"}), 400

    title_c, _, combined = preprocess_arabic(raw_text, "")
    enc = tokenizer(
        title_c or "[PAD]", "[PAD]",
        max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    ids   = enc["input_ids"].to(device)
    mask  = enc["attention_mask"].to(device)
    ttype = enc.get("token_type_ids", torch.zeros_like(ids)).to(device)

    with torch.no_grad():
        logits = model(ids, mask, ttype)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return jsonify({
        "prob_fake":    round(float(probs[0]), 4),
        "prob_real":    round(float(probs[1]), 4),
        "raw_logits":   [round(float(x), 4) for x in logits[0].tolist()],
        "cleaned_text": combined[:300]
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "device":         str(device),
        "label_mapping":  "index 0=FAKE, index 1=REAL",
        "threshold_real": THRESHOLD_REAL,
        "max_words":      MAX_WORDS,
        "fp16":           USE_FP16,
        "eval_mode":      not model.training
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=False, processes=1)