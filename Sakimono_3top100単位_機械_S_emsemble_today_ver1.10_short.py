import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import json
import os
import time
#import sys  # 必要ならアンコメント

# フォルダパスを指定
folder_path = "./分析"

# フォルダが存在しない場合は作成
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# top_stocks_path = "sakimono_top_stocks_全体用固定_0.2_0.2.xlsx"
top_stocks_path = "sakimono_top_stocks_全体用固定_0.2_0.2_20250111_152228.xlsx"
# ここから機械学習モデル
file_path = top_stocks_path

df = pd.read_excel(file_path)
df = df[["業種", "銘柄コード", "企業名"]]

tickers = {row["銘柄コード"]: row["企業名"] for _, row in df.iterrows()}
ticker_symbols = list(tickers.keys())

data = yf.download(ticker_symbols, period="10y", progress=False)["Close"].dropna()

# 本日の日付を取得
today = pd.Timestamp(datetime.now().date())

# 現在の日時を取得
now = datetime.now()

# 今日の日付と午前7時以降の条件をチェックして行を削除
if data.index[-1].date() == now.date() and now.time() >= datetime.strptime("07:00:00", "%H:%M:%S").time():
    print("削除")
    data = data.iloc[:-1]

print("最後のデータ日付", data.index[-1].date())  

# 確認: 最後のデータが本日分ではなくなったことを確認
print(data.tail())
initial_investment = 100000
leverage = 10
stop_loss_threshold = 0.015
price_threshold = 5000
lookback_days = 30
deal_term = 1
interest_rate = 0.028

dates = data.index.unique()
periods_ml = []
start_idx = lookback_days

while start_idx + deal_term < len(dates):
    period = dates[start_idx - lookback_days: start_idx + deal_term]
    periods_ml.append(period)
    start_idx += deal_term

split_idx = int(len(periods_ml) * 0.9)
train_periods = periods_ml[:split_idx]
test_periods = periods_ml[split_idx:]

def create_features_and_labels(periods_input):
    features = []
    labels = []
    dates_list = []
    tickers_list_ = []
    for period_ in periods_input:
        for ticker_ in ticker_symbols:
            try:
                ticker_data = data[ticker_].loc[period_].dropna()
                if len(ticker_data) != lookback_days + deal_term:
                    continue
                x = ticker_data.iloc[:lookback_days].pct_change().dropna().values
                if len(x) != lookback_days - 1:
                    continue
                y = 1 if ticker_data.iloc[-1] > ticker_data.iloc[lookback_days - 1] else 0
                features.append(x)
                labels.append(y)
                dates_list.append(period_[lookback_days - 1])
                tickers_list_.append(ticker_)
            except Exception as e:
                continue
    return np.array(features), np.array(labels), dates_list, tickers_list_

X_train, y_train, _, _ = create_features_and_labels(train_periods)
X_test, y_test, dates_test, tickers_test = create_features_and_labels(test_periods)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def optimize_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train_scaled, y_train)
    return grid_search.best_estimator_

lgb_param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.3],
    'max_depth': [7]
}
best_lgb_model = optimize_model(lgb.LGBMClassifier(random_state=42), lgb_param_grid)

cat_param_grid = {
    'iterations': [500],
    'learning_rate': [0.1],
    'depth': [7]
}
best_cat_model = optimize_model(CatBoostClassifier(random_state=42, verbose=0), cat_param_grid)

xgb_param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.3],
    'max_depth': [11]
}
best_xgb_model = optimize_model(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_param_grid)

lgb_probs = best_lgb_model.predict_proba(X_test_scaled)[:, 1]
cat_probs = best_cat_model.predict_proba(X_test_scaled)[:, 1]
xgb_probs = best_xgb_model.predict_proba(X_test_scaled)[:, 1]

ensemble_probs = (lgb_probs + cat_probs + xgb_probs) / 3

os.makedirs('./temp_model', exist_ok=True)
joblib.dump(best_lgb_model, './temp_model/best_lgb_model.pkl')
joblib.dump(best_cat_model, './temp_model/best_cat_model.pkl')
joblib.dump(best_xgb_model, './temp_model/best_xgb_model.pkl')

optimized_parameters = {
    "LightGBM": best_lgb_model.get_params(),
    "CatBoost": best_cat_model.get_all_params(),
    "XGBoost": best_xgb_model.get_params()
}

with open('optimized_parameters.json', 'w') as f:
    json.dump(optimized_parameters, f, indent=4)

print("Optimized Parameters:")
print(json.dumps(optimized_parameters, indent=4))

investment = initial_investment
simulation_results = []
test_data = pd.DataFrame({
    "Date": dates_test,
    "Ticker": tickers_test,
    "Probability": ensemble_probs
})

unique_dates = test_data["Date"].unique()
winrate = 0
win_cnt = 0
total_cnt = 0

for current_date in unique_dates:
    day_data = test_data[test_data["Date"] == current_date]
    day_data = day_data.sort_values(by="Probability", ascending=False)
    top_stocks_sim = day_data.head(1)
    total_profit_loss = 0
    total_fees = 0

    for idx, row in top_stocks_sim.iterrows():
        ticker_ = row["Ticker"]
        company_name = tickers.get(ticker_, "")
        try:
            date_idx = data.index.get_loc(current_date)
            start_idx_ = date_idx
            end_idx_ = start_idx_ + deal_term
            if end_idx_ >= len(data):
                continue

            start_price = data[ticker_].iloc[start_idx_]
            end_price = data[ticker_].iloc[end_idx_]

            low_prices = data[ticker_].iloc[start_idx_:end_idx_+1]
            stop_loss_price = start_price * (1 - stop_loss_threshold)
            if low_prices.min() <= stop_loss_price:
                exit_price = stop_loss_price
            else:
                exit_price = end_price

            trade_amount = investment 
            position_size = trade_amount

            number_of_shares = position_size // start_price
            if number_of_shares == 0:
                continue

            actual_investment = number_of_shares * start_price
            fee = 0.0132 * actual_investment
            price_difference = exit_price - start_price

            if price_difference > 0:
                win_cnt += 1
            total_cnt += 1

            if total_cnt > 0:
                winrate = win_cnt / total_cnt

            profit_loss = price_difference * number_of_shares * leverage - fee
            investment += profit_loss
            total_profit_loss += profit_loss
            total_fees += fee

            simulation_results.append({
                "Date": current_date.date(),
                "Ticker": ticker_,
                "Company Name": company_name,
                "Start Price": start_price,
                "Stop Loss Price": stop_loss_price,
                "End Price": end_price,
                "Exit Price": exit_price,
                "Price Difference": price_difference,
                "Number of Shares": int(number_of_shares),
                "Profit/Loss (JPY)": round(profit_loss, 2),
                "Win Rate": round(winrate, 4),
                "Fee (JPY)": fee,
                "Total Investment (JPY)": round(investment, 2)
            })
        except Exception as e:
            continue

    print(f"Date: {current_date.date()}, Investment: {round(investment,2)}, Total Profit/Loss: {round(total_profit_loss,2)}, Total Fees: {total_fees}, Win Rate: {round(winrate,4)}")

results_df = pd.DataFrame(simulation_results)
simulation_file_name = f"./分析/simulation_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
results_df.to_excel(simulation_file_name, index=False)
print(f"\nシミュレーション結果が '{simulation_file_name}' に保存されました。")


# `results_df` が提供されたデータと仮定

# Date列をdatetime型に変換
results_df["Date"] = pd.to_datetime(results_df["Date"])

# 曜日列を追加（0=月曜日, ..., 6=日曜日）
results_df["Weekday"] = results_df["Date"].dt.dayofweek

# 曜日ごとに勝率を計算
# 勝ち（Profit/Loss (JPY) > 0）をカウント
weekday_stats = results_df.groupby("Weekday").apply(
    lambda x: pd.Series({
        "勝利数": (x["Profit/Loss (JPY)"] > 0).sum(),
        "取引数": len(x),
        "勝率": (x["Profit/Loss (JPY)"] > 0).mean()  # 勝率計算
    })
).reset_index()

# 曜日を日本語に変換
day_mapping_jp = {
    0: "月曜日",
    1: "火曜日",
    2: "水曜日",
    3: "木曜日",
    4: "金曜日",
    5: "土曜日",
    6: "日曜日",
}
weekday_stats["曜日"] = weekday_stats["Weekday"].map(day_mapping_jp)

# 結果を見やすく整理
weekday_stats = weekday_stats[["曜日", "勝利数", "取引数", "勝率"]]

# 結果を表示
print(weekday_stats)

# エクセルに保存
weekday_winrates_output_file = f"./分析/weekday_winrates_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
weekday_stats.to_excel(weekday_winrates_output_file, index=False)
print(f"\n曜日ごとの勝率が '{weekday_winrates_output_file}' に保存されました。")


# データを読み込む（DataFrameとして提供されていると仮定）
# results_df = pd.read_csv("your_file.csv")  # ファイルから読み込む場合

# Tickerごとに勝率を計算
ticker_stats = results_df.groupby("Ticker").apply(
    lambda x: pd.Series({
        "勝利数": (x["Profit/Loss (JPY)"] > 0).sum(),
        "取引数": len(x),
        "勝率": (x["Profit/Loss (JPY)"] > 0).mean()
    })
).reset_index()

# 結果を見やすく表示
print("\nTickerごとの勝率:")
print(ticker_stats)

# 結果をエクセルに保存
ticker_winrates_output_file = f"./分析/ticker_winrates_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
ticker_stats.to_excel(ticker_winrates_output_file, index=False)
print(f"\nTickerごとの勝率が '{ticker_winrates_output_file}' に保存されました。")


# 予測対象の銘柄と、その銘柄コード→企業名の辞書
file_path = file_path
df = pd.read_excel(file_path)
df = df[["業種", "銘柄コード", "企業名"]]
tickers = {row["銘柄コード"]: row["企業名"] for _, row in df.iterrows()}
ticker_symbols = list(tickers.keys())

def get_daily_close_from_hourly(ticker, period="10y"):
    """
    指定ティッカーの1時間足データを取得し、日足の「終値」Series を返す関数。
    
    Parameters
    ----------
    ticker : str
        ダウンロードするティッカー(例: "ES=F", "SI=F"など)
    period : str
        ダウンロード範囲 (例: "10y", "1y", "6mo" ...)

    Returns
    -------
    df_daily : pd.Series
        インデックスがDatetime、値がその日の日足終値
        欠損日はドロップ済み
    """
    # 1. 1時間足データをダウンロード
    df_hourly = yf.download(ticker, period=period, interval="1h", progress=False)
    if df_hourly.empty:
        print(f"  {ticker}: 1時間足データ取得できず。")
        return pd.Series(dtype=float)

    # 2. 「日足」へリサンプリングして終値を取得
    #    indexがDatetimeIndexになっている前提
    df_hourly = df_hourly.sort_index()  # 念のため時系列ソート
    df_daily = df_hourly['Close'].resample('D').last().dropna()

    return df_daily

def create_features_for_today():
    """
    1) ティッカーごとに1時間足DL → 日足の終値を作成 (get_daily_close_from_hourly)
    2) 最後に(銘柄数, lookback_days-1)の形で特徴量を返す
    3) 併せて「利用できる銘柄のリスト」も返す
    """
    features = []
    tickers_list_today = []

    for ticker in ticker_symbols:
        print(f"\n[INFO] 1時間足データダウンロード → 日足リサンプリング → {ticker}")
        df_daily = get_daily_close_from_hourly(ticker, period="3mo")

        # データがない場合はスキップ
        if df_daily.empty:
            continue

        print(df_daily.tail(5))

        # 今日の日付と午前7時以降の条件をチェックして行を削除
        if len(df_daily) > 0:
            print("チャートデータ：", df_daily.index[-1].date())
            print("現在：", now.date())
            print("現在：", now.time() )
            print("閾時間：", datetime.strptime("07:00:00", "%H:%M:%S").time())
            if df_daily.index[-1].date() == now.date() and now.time() >= datetime.strptime("07:00:00", "%H:%M:%S").time():
                print("  当日分(未確定)を削除")
                df_daily = df_daily.iloc[:-1]

        print(df_daily.tail(5))
        # 欠損除去
        df_daily = df_daily.dropna()

        # 30日分足りないならスキップ
        if len(df_daily) < lookback_days:
            print(f"  {ticker}: 過去{lookback_days}日分のデータがありません。スキップ。")
            continue

        # 過去30日分の終値を取り出し、pct_change() → (29,) のリターンベクトル
        df_lookback = df_daily.iloc[-lookback_days:]
        x = df_lookback.pct_change().dropna().values  # shape->(29,)

        if len(x) != lookback_days - 1:
            print(f"  {ticker}: リターン数が {len(x)} 個しかないためスキップ。")
            continue

        # 2次元化 => shape (1, 29)
        x_2d = x.reshape(1, -1)

        features.append(x_2d)
        tickers_list_today.append(ticker)

    # 全部終わったら、(銘柄数, 29) の形にまとめて返す
    if len(features) == 0:
        print("[INFO] No valid ticker data to create features.")
        X_today = np.empty((0, lookback_days - 1))
    else:
        X_today = np.vstack(features)

    return X_today, tickers_list_today

# --- 今日の予測を実行 ---
X_today, tickers_today = create_features_for_today()
print(f"\n最終的な X_today.shape = {X_today.shape}")
print(f"銘柄数 = {len(tickers_today)}")

if len(X_today) > 0:
    # StandardScaler でスケーリング
    X_today_scaled = scaler.transform(X_today)

    # 各モデルで予測確率
    lgb_probs_today = best_lgb_model.predict_proba(X_today_scaled)[:, 1]
    cat_probs_today = best_cat_model.predict_proba(X_today_scaled)[:, 1]
    xgb_probs_today = best_xgb_model.predict_proba(X_today_scaled)[:, 1]

    # アンサンブル (平均)
    ensemble_probs_today = (lgb_probs_today + cat_probs_today + xgb_probs_today) / 3

    # DataFrame化
    today_data = pd.DataFrame({
        "Ticker": tickers_today,
        "Probability": ensemble_probs_today
    }).sort_values("Probability", ascending=False)

else:
    # 今日の特徴量が作れなかった
    today_data = pd.DataFrame(columns=["Ticker", "Probability"])

# 予測確率の降順にソート
recommendation_data = today_data.sort_values(by="Probability", ascending=False)

# recommendation_data を確認
print("recommendation_data", recommendation_data)

# トップ3・トップ5を抜き出して表示
purchase_recommendations_top3 = []
purchase_recommendations_top5 = []
end_date = datetime.now()
from pandas.tseries.offsets import BusinessDay

# 今日から `deal_term` 営業日後を計算
exit_date = end_date + BusinessDay(deal_term)
print("何日後" ,BusinessDay(deal_term))
# exit_date = end_date + timedelta(days=deal_term)

for top_n, purchase_recommendations in [(3, purchase_recommendations_top3), (5, purchase_recommendations_top5)]:
    # ここで top_n 件分のみ切り出し（すでに Probability 降順）
    top_stocks_rec = recommendation_data.head(top_n)
    for idx, row in top_stocks_rec.iterrows():
        ticker_ = row["Ticker"]
        company_name = tickers.get(ticker_, "")
        # try:
        df_temp = yf.download(ticker_, period="5d", interval="1h", progress=False)["Close"].dropna()

        # データが空かどうかをチェック
        if df_temp.empty:
            print(f"[WARNING] {ticker_}: 現在価格データが空です。スキップします。")
            continue

        # 最新の終値を取得
        current_price = float(df_temp.iloc[-1].item())
        print(f" {ticker_}: 現在価格 = {current_price}")

        # 計算: ストップロス価格と購入枚数
        stop_loss_price = current_price * (1 - stop_loss_threshold)
        number_of_shares = (initial_investment // top_n) // current_price

        if number_of_shares == 0:
            print(f"[WARNING] {ticker_}: 購入枚数が0です。スキップします。")
            continue

        # 推奨リストに追加
        purchase_recommendations.append({
            "Entry Date": end_date.strftime('%Y-%m-%d'),
            "Exit Date": exit_date.strftime('%Y-%m-%d'),
            "Term (Business Days)": deal_term,
            "Ticker": ticker_,
            "Company Name": company_name,
            "Current Price": current_price,
            "Stop Loss Price": stop_loss_price,
            "Shares Bought": int(number_of_shares),
            "Investment per Stock (JPY)": round(number_of_shares * current_price, 2),
            "Predicted Probability (%)": round(row["Probability"] * 100, 2),
        })
        # except Exception as e:
        #     print(f"[ERROR] {ticker_}: {e}")
        #     continue

# 表示用のカラム並び
columns_order = [
    "Entry Date", "Exit Date", "Term (Business Days)", "Ticker", "Company Name",
    "Current Price", "Stop Loss Price", "Shares Bought", "Investment per Stock (JPY)",
    "Predicted Probability (%)"
]

# 上位3件
purchase_df_top3 = pd.DataFrame(purchase_recommendations_top3)
if not purchase_df_top3.empty:
    purchase_df_top3 = purchase_df_top3.sort_values("Predicted Probability (%)", ascending=False)
    purchase_df_top3 = purchase_df_top3[columns_order]

# 上位5件
purchase_df_top5 = pd.DataFrame(purchase_recommendations_top5)
if not purchase_df_top5.empty:
    purchase_df_top5 = purchase_df_top5.sort_values("Predicted Probability (%)", ascending=False)
    purchase_df_top5 = purchase_df_top5[columns_order]

# 結果表示
print("\n上位3件の購入推奨銘柄:")
print(purchase_df_top3)
print("\n上位5件の購入推奨銘柄:")
print(purchase_df_top5)

# ファイル出力
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name_top3 = f"./分析/過去25日間_銘柄選定結果_top3_{timestamp}.xlsx"
file_name_top5 = f"./分析/過去25日間_銘柄選定結果_top5_{timestamp}.xlsx"
purchase_df_top3.to_excel(file_name_top3, index=False)
purchase_df_top5.to_excel(file_name_top5, index=False)
print(f"\n購入リストが保存されました:\n- {file_name_top3}\n- {file_name_top5}")


time.sleep(3)
import smtplib
import os
import time
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Gmail 設定
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
GMAIL_USER = "k.atsuofxtrade@gmail.com"
GMAIL_PASSWORD = "yyyegokbvfcyufnm"

# 送信先リスト
recipient_list = [
    "k.atsuojp429@gmail.com",
    "k.atsuo-jp@outlook.com",
    # "kotera2hjp@gmail.com",
    # "kotera2hjp@outlook.jp",
    # "kotera2hjp@yahoo.co.jp",
    "k.atsuofxtrade@gmail.com"
]


# スクリプトのディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

# ファイルパスをスクリプトのパスに基づいて定義
file_path_top3 = os.path.join(script_dir, file_name_top3)
file_path_top5 = os.path.join(script_dir, file_name_top5)


simulation_file = os.path.join(script_dir, simulation_file_name)
weekday_winrates_output_file_file = os.path.join(script_dir, weekday_winrates_output_file)
ticker_winrates_output_file_file = os.path.join(script_dir, ticker_winrates_output_file)

file_paths = [
    file_path_top3,
    file_path_top5,
    simulation_file,
    weekday_winrates_output_file_file,
    ticker_winrates_output_file_file,
]
file_paths = [path.replace("\\", "/").replace("./", "") for path in file_paths]

print(file_paths)

# 日付取得
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()  # TLS 暗号化を開始
    server.login(GMAIL_USER, GMAIL_PASSWORD)  # ログイン

    rato = investment / initial_investment
    profit = investment - initial_investment

    # HTML形式の表を作成
    weekday_table_html = weekday_stats.to_html(index=False, justify="center", border=1)
    ticker_table_html = ticker_stats.to_html(index=False, justify="center", border=1)
    top5_stocks_html = ""
    if not purchase_df_top5.empty:
        top5_stocks_html = purchase_df_top5.to_html(index=False, justify="center", border=1)

    # シミュレーション結果のテーブルを作成
    simulation_results_html = ""
    if not results_df.empty:
        # 小数点以下1位に丸め、通常表記に変換
        results_df = results_df.applymap(lambda x: f"{x:.1f}" if isinstance(x, (float, int)) else x)
        simulation_results_html = results_df.to_html(index=False, justify="center", border=1, escape=False)

    for recipient in recipient_list:
        # メールの作成
        msg = MIMEMultipart("alternative")
        msg["From"] = GMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = f"先物　実績のある先物購入リストのおすすめ結果 ({current_date}) {int(investment)}円 {rato:.2f}倍"

        # HTML形式のメール本文を作成
        body_html = f"""
        <html>
        <head>
            <style>
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <p>{recipient} 様</p>
            <p>平素よりお世話になっております。</p>
            <p>本日の購入リストのおすすめ結果をお送りいたします。</p>
            <p>
                現在の投資額は {int(investment):,} 円で、初期投資額の {int(initial_investment):,} 円に対し、
                {rato:.2f} 倍となっており、利益は {int(profit):,} 円です。
            </p>
            <p>
                レバレッジ: {leverage} 倍<br>
                取引期間: {deal_term} 営業日<br>
                総合勝率: {winrate * 100:.2f} %
            </p>
            <h3>曜日ごとの勝率:</h3>
            {weekday_table_html}
            <h3>銘柄ごとの勝率:</h3>
            {ticker_table_html}
            <h3>上位5件の推奨銘柄:</h3>
            {top5_stocks_html}
            <h3>シミュレーション結果:</h3>
            {simulation_results_html}
            <p>本リストは、新たに学習を行った結果に基づいて作成されました。</p>
            <p>詳細につきましては、添付ファイルをご確認ください。</p>
            <p>ご不明な点やご質問がございましたら、どうぞお気軽にお問い合わせください。</p>
            <p>今後とも何卒よろしくお願い申し上げます。</p>
            <p>チーム一同</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body_html, "html"))

        # 添付ファイルを追加
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"ファイルが存在しません: {file_path}")
            else:
                print(f"ファイル確認済み: {file_path}, サイズ: {os.path.getsize(file_path) / 1024:.2f} KB")
                with open(file_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    safe_filename = os.path.basename(file_path).encode("ascii", "ignore").decode()
                    part.add_header("Content-Disposition", f"attachment; filename={safe_filename}")
                    msg.attach(part)

        # メール送信
        server.sendmail(GMAIL_USER, recipient, msg.as_string())
        print(f"送信しました: {recipient}")
        time.sleep(3)

    # サーバーを閉じる
    server.quit()
    print("すべてのメールを送信しました。")

except Exception as e:
    print(f"送信エラー: {e}")