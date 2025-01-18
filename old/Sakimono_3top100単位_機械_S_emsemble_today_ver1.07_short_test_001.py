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


# 今日買うべき銘柄を予測する部分だけ修正
lookback_days = 30
stop_loss_threshold = 0.015
initial_investment = 100000
deal_term = 1

# 予測対象の銘柄と、その銘柄コード→企業名の辞書
file_path = "sakimono_top_stocks_全体用固定_0.2_0.2_20250111_152228.xlsx"
df = pd.read_excel(file_path)
df = df[["業種", "銘柄コード", "企業名"]]
tickers = {row["銘柄コード"]: row["企業名"] for _, row in df.iterrows()}
ticker_symbols = list(tickers.keys())

end_date = datetime.now()

def create_features_for_today():
    """
    今日の銘柄を予測するために「銘柄ごとに再ダウンロード」して
    最新データの日付を表示しつつ、特徴量を作成する。
    """
    features = []
    tickers_list_today = []

    for ticker_ in ticker_symbols:
        print(f"\n[INFO] ダウンロード開始 → {ticker_}")
        try:
            # 学習時のように全銘柄一括ではなく、ticker_ を単独でダウンロード
            df_today = yf.download(ticker_, period="10y", progress=False)["Close"]
            if df_today.empty:
                print(f"  {ticker_}: データが取得できませんでした。")
                continue

            # ここでダウンロードした最新日付を表示
            latest_date = df_today.index[-1].strftime('%Y-%m-%d')
            print(f"  {ticker_}: 最新日付 = {latest_date}")
            print(df_today.tail(30))

            # 今日の日付と午前7時以降の条件をチェックして行を削除
            if df_today.index[-1].date() == now.date() and now.time() >= datetime.strptime("07:00:00", "%H:%M:%S").time():
                print("削除")
                df_today = df_today.iloc[:-1]

            # ここでダウンロードした最新日付を表示
            latest_date = df_today.index[-1].strftime('%Y-%m-%d')
            print(f"  {ticker_}: 修正後最新日付 = {latest_date}")
            print(df_today.tail(30))

            # lookback_daysぶん足りない場合はスキップ
            if len(df_today) < lookback_days:
                print(f"  {ticker_}: 過去{lookback_days}日分のデータがありません。スキップ。")
                continue

            # 過去lookback_days日分のリターン変化率ベクトルを作成
            x = df_today.iloc[-lookback_days:].pct_change().dropna().values

            # 必要に応じて次元を調整
            if x.ndim == 1:
                x = x.reshape(1, -1)  # 2次元に変換

            # `x` を特徴量に追加
            features.append(x)
            tickers_list_today.append(ticker_)

        except Exception as e:
            print(f"  {ticker_}: エラーが発生しました: {e}")
            continue

    # 特徴量を 2 次元配列に変換
    features = np.vstack(features) if features else np.empty((0, lookback_days - 1))

    return features, tickers_list_today


# --- 今日の予測を実行 ---
X_today, tickers_today = create_features_for_today()

# もし利用できる銘柄があれば確率を計算
if len(X_today) > 0:
    X_today_scaled = scaler.transform(X_today)

    # 各モデルから予測確率を取得
    lgb_probs_today = best_lgb_model.predict_proba(X_today_scaled)[:, 1]
    cat_probs_today = best_cat_model.predict_proba(X_today_scaled)[:, 1]
    xgb_probs_today = best_xgb_model.predict_proba(X_today_scaled)[:, 1]

    # アンサンブル
    ensemble_probs_today = (lgb_probs_today + cat_probs_today + xgb_probs_today) / 3

    # 結果をDataFrame化
    today_data = pd.DataFrame({
        "Ticker": tickers_today,
        "Probability": ensemble_probs_today
    })

else:
    # 今日の特徴量が作れなかった
    today_data = pd.DataFrame(columns=["Ticker", "Probability"])

# 予測確率の降順にソート
recommendation_data = today_data.sort_values(by="Probability", ascending=False)

# トップ3・トップ5を抜き出して表示
purchase_recommendations_top3 = []
purchase_recommendations_top5 = []
exit_date = end_date + timedelta(days=deal_term)

for top_n, purchase_recommendations in [(3, purchase_recommendations_top3), (5, purchase_recommendations_top5)]:
    top_stocks_rec = recommendation_data.head(top_n)
    for idx, row in top_stocks_rec.iterrows():
        ticker_ = row["Ticker"]
        company_name = tickers.get(ticker_, "")
        try:
            # 直近の終値(ダウンロード済みのdf_todayを再度読むでも可)
            # 今回は簡易的に再DLする
            df_temp = yf.download(ticker_, period="1d", progress=False)["Close"].dropna()
            if len(df_temp) == 0:
                continue
            current_price = df_temp.iloc[-1]

            stop_loss_price = current_price * (1 - stop_loss_threshold)
            number_of_shares = (initial_investment // top_n) // current_price
            if number_of_shares == 0:
                continue

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
        except Exception as e:
            print(f"  {ticker_}: {e}")
            continue

# 表示用のカラム並び
columns_order = [
    "Entry Date", "Exit Date", "Term (Business Days)", "Ticker", "Company Name",
    "Current Price", "Stop Loss Price", "Shares Bought", "Investment per Stock (JPY)",
    "Predicted Probability (%)"
]
purchase_df_top3 = pd.DataFrame(purchase_recommendations_top3)[columns_order] if purchase_recommendations_top3 else pd.DataFrame(columns=columns_order)
purchase_df_top5 = pd.DataFrame(purchase_recommendations_top5)[columns_order] if purchase_recommendations_top5 else pd.DataFrame(columns=columns_order)

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

print("\n購入リストが以下のファイルに保存されました。")
print(f" - {file_name_top3}")
print(f" - {file_name_top5}")

# 必要に応じてメール送信などの処理をそのまま継承
# -----------------------------
try:
    recipient_list = [
        "k.atsuojp429@gmail.com",
        "k.atsuo-jp@outlook.com",
        "kotera2hjp@gmail.com",
        "kotera2hjp@outlook.jp",
        "kotera2hjp@yahoo.co.jp"
    ]
    current_dir = os.getcwd()
    file_path_top3 = os.path.join(current_dir, file_name_top3)
    file_path_top5 = os.path.join(current_dir, file_name_top5)
    outlook = win32com.client.Dispatch("Outlook.Application")
    for recipient in recipient_list:
        mail = outlook.CreateItem(0)
        current_date = datetime.now().strftime("%Y-%m-%d")
        mail.To = recipient
        mail.Subject = f"先物ショート　先物購入リストのおすすめ結果 ({current_date})"
        mail.Body = (
            f"{recipient} 様\n\n"
            "本日の購入リストのおすすめ結果をお送りします。\n\n"
            "添付ファイルをご確認ください。\n\n"
            "よろしくお願いいたします。\n\n"
            "チーム一同"
        )
        mail.Attachments.Add(file_path_top3)
        mail.Attachments.Add(file_path_top5)
        mail.Send()
        time.sleep(1)

except Exception as e:
    print(f"メール送信エラー: {e}")