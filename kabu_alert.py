import os
import time
import datetime
import math
import smtplib
import yfinance as yf
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage
import mplfinance as mpf

# ==== 設定セクション ====
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
GMAIL_USER = "k.atsuofxtrade@gmail.com"
GMAIL_PASSWORD = "yyyegokbvfcyufnm"

# 監視対象: 日経平均と米株（例）
TICKERS = ["^N225", "^NDX"]  # 日経平均とNasdaq100 (US100)
# 下落アラート閾値（％）
DROP_THRESHOLD = 1.0
# 状態ファイル保存ディレクトリ
STATE_DIR = "./state"
os.makedirs(STATE_DIR, exist_ok=True)

# メール送信先リスト
RECIPIENT_LIST = [
    "kotera2hjp@gmail.com",
    "k.atsuofxtrade@gmail.com",
    "satosato.k543@gmail.com",
    "yukikimura1124@gmail.com"
]

# 前回価格読み込み/保存

def state_file(ticker: str) -> str:
    safe = ticker.replace("^", "v").replace("/", "_")
    return os.path.join(STATE_DIR, f"{safe}.txt")


def load_last_price(ticker: str) -> float:
    path = state_file(ticker)
    try:
        return float(open(path).read().strip())
    except:
        return None


def save_last_price(ticker: str, price: float):
    open(state_file(ticker), "w").write(f"{price}")

# チャート生成

def generate_chart(ticker: str, period="1d", interval="5m") -> str:
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty:
        return None
    save_path = f"./charts/{ticker.strip('^')}_chart.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mpf.plot(df, type="candle", style="yahoo", title=ticker,
             savefig=dict(fname=save_path, dpi=150))
    return save_path

# メール送信

def send_alert(drops: dict):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"[Alert] {now} 株価下落検知: " + ", ".join(drops.keys())

    for recipient in RECIPIENT_LIST:
        msg = MIMEMultipart("related")
        msg["From"] = GMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = subject

        body = f"<html><body><h3>株価下落検知 ({now})</h3><ul>"
        for tkr, info in drops.items():
            body += f"<li>{tkr}: 前回 {info['last']:.2f} → 現在 {info['now']:.2f} ({info['pct']:.2f}% 下落)</li>"
        body += "</ul></body></html>"
        msg.attach(MIMEText(body, "html"))

        # チャート添付
        for i, tkr in enumerate(drops):
            img_path = generate_chart(tkr)
            if img_path and os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    cid = f"chart{i}"
                    img.add_header('Content-ID', f"<{cid}>")
                    img.add_header('Content-Disposition', 'inline', filename=os.path.basename(img_path))
                    msg.attach(img)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_USER, recipient, msg.as_string())
        time.sleep(1)
    print(f"[INFO] Alerts sent to {len(RECIPIENT_LIST)} recipients.")

# メイン処理

def main():
    drops = {}
    for tkr in TICKERS:
        data = yf.Ticker(tkr).history(period="1d", interval="5m")
        if data.empty:
            continue
        now_price = data["Close"][-1]
        last_price = load_last_price(tkr)
        if last_price:
            pct = (now_price - last_price) / last_price * 100
            if pct <= -DROP_THRESHOLD:
                drops[tkr] = {"last": last_price, "now": now_price, "pct": pct}
        save_last_price(tkr, now_price)

    if drops:
        send_alert(drops)
    else:
        print("[INFO] No significant drops detected.")

if __name__ == "__main__":
    # 30分ごとにチェックをループ
    while True:
        main()
        time.sleep(30 * 60)
