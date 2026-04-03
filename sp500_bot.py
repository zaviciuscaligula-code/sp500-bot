import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import xgboost as xgb
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# Telegram ρυθμίσεις (Θα τα τραβάει αυτόματα από τα Secrets του GitHub)
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Σφάλμα Telegram: {e}")

# 1. ΛΗΨΗ ΔΕΔΟΜΕΝΩΝ
print("⏳ Λήψη εβδομαδιαίων δεδομένων...")
sp_raw = yf.download("^GSPC", period="12y", interval="1wk", auto_adjust=True)
vix_raw = yf.download("^VIX", period="12y", interval="1wk", auto_adjust=True)
tnx_raw = yf.download("^TNX", period="12y", interval="1wk", auto_adjust=True)

df = pd.DataFrame(index=sp_raw.index)
df['SP_Close'] = sp_raw['Close']
df['VIX_Close'] = vix_raw['Close']
df['TNX_Close'] = tnx_raw['Close']
df = df.dropna()

# 2. ΥΠΟΛΟΓΙΣΜΟΣ Z-SCORES
window_z = 20
df['SP_Z'] = (df['SP_Close'] - df['SP_Close'].rolling(window=window_z).mean()) / df['SP_Close'].rolling(window=window_z).std()
df['VIX_Z'] = (df['VIX_Close'] - df['VIX_Close'].rolling(window=window_z).mean()) / df['VIX_Close'].rolling(window=window_z).std()
df['TNX_Z'] = (df['TNX_Close'] - df['TNX_Close'].rolling(window=window_z).mean()) / df['TNX_Close'].rolling(window=window_z).std()
df = df.dropna()

df['SP_Real_1W'] = (df['SP_Z'].shift(-1) > df['SP_Z']).astype(int)

seq_length = 5
sp_z, vix_z, tnx_z = df['SP_Z'].values, df['VIX_Z'].values, df['TNX_Z'].values
y_target = df['SP_Real_1W'].values

X, Y = [], []
for j in range(len(df) - seq_length - 1):
    X.append(np.column_stack((sp_z[j : j + seq_length], vix_z[j : j + seq_length], tnx_z[j : j + seq_length])))
    Y.append(y_target[j + seq_length])

X, Y = np.array(X), np.array(Y)
X_flat = X.reshape(X.shape[0], -1)

# Εκπαίδευση
xgb_m = xgb.XGBClassifier(eval_metric='logloss').fit(X_flat, Y)
test_seq = np.column_stack((df['SP_Z'].iloc[-seq_length:].values, df['VIX_Z'].iloc[-seq_length:].values, df['TNX_Z'].iloc[-seq_length:].values))
X_latest = test_seq.reshape(1, -1)

# Προβλέψεις
xgb_probs = xgb_m.predict_proba(X_latest)[0]
xgb_up_prob = xgb_probs[1] * 100
xgb_label = "🟢 BULLISH" if xgb_up_prob >= 50 else "🔴 BEARISH"

m_sp = ARIMA(df['SP_Z'], order=(2, 1, 0)).fit()
f_sp = m_sp.get_forecast(steps=1)
arima_up_prob = (1 - norm.cdf(df['SP_Z'].iloc[-1], loc=f_sp.predicted_mean.iloc[0], scale=f_sp.se_mean.iloc[0])) * 100
arima_label = "🟢 BULLISH" if arima_up_prob >= 50 else "🔴 BEARISH"

# --- Δικλείδα Ασφαλείας ---
if arima_label == xgb_label:
    if xgb_up_prob >= 70 or arima_up_prob >= 70 or xgb_up_prob <= 30 or arima_up_prob <= 30:
        advice = "🚀 *ΙΣΧΥΡΟ ΣΗΜΑ:* Τα μοντέλα συμφωνούν και δείχνουν μεγάλη βεβαιότητα! Έτοιμοι για δράση."
    else:
        advice = "👌 *ΗΠΙΟ ΣΗΜΑ:* Τα μοντέλα συμφωνούν αλλά χωρίς υπερβολική βεβαιότητα."
else:
    advice = "🛑 *ΑΠΟΧΗ (NO TRADE):* Τα μοντέλα συγκρούονται μετωπικά! Περιμένουμε στο περιθώριο."

# Μήνυμα Telegram
message = (
    "🔮 *ΕΒΔΟΜΑΔΙΑΙΑ ΠΡΟΒΛΕΨΗ S&P 500*\n"
    "===================================\n"
    f"📈 Τρέχουσα Τιμή: `{df['SP_Close'].iloc[-1]:.2f}`\n"
    "-----------------------------------\n"
    f"📊 *ARIMA:* {arima_label} ({arima_up_prob:.1f}%)\n"
    f"🌳 *XGBoost:* {xgb_label} ({xgb_up_prob:.1f}%)\n"
    "-----------------------------------\n"
    f"{advice}"
)

send_telegram_message(message)
print("✅ Η ενημέρωση στάλθηκε στο Telegram!")
