
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

# ティッカーリスト（業種ごとに分類）
futures_tickers = {
    "Indices": [
        "ES=F", "NQ=F", "YM=F", "RTY=F", "VIX=F", "DAX=F", "FTSE=F",
        "NK=F", "HSI=F", "KOSPI=F", "S&P500", "NASDAQ100"
    ],
    "Commodities": [
        "CL=F", "BZ=F", "NG=F", "HO=F", "RB=F",
        "GC=F", "SI=F", "PL=F", "PA=F",
        "HG=F", "ALI=F", "ZC=F", "ZW=F", "ZS=F",
        "CC=F", "KC=F", "LB=F", "CT=F", "OJ=F"
    ],
    "Currencies": [
        "6E=F", "6J=F", "6A=F", "6C=F", "6B=F", "6N=F", "6S=F",
        "DX=F"
    ],
    "Interest Rates": [
        "ZB=F", "ZN=F", "ZF=F", "ZT=F",
        "GE=F", "ED=F"
    ],
    "Energy": [
        "CL=F", "NG=F", "HO=F", "RB=F", "BZ=F",
        "QL=F", "QA=F"
    ],
    "Metals": [
        "GC=F", "SI=F", "PL=F", "PA=F", "HG=F"
    ],
    "Agriculture": [
        "ZC=F", "ZW=F", "ZS=F", "ZM=F", "ZL=F",
        "CC=F", "KC=F", "CT=F", "LB=F", "OJ=F"
    ],
    "Softs": [
        "SB=F", "JO=F", "CC=F", "KC=F"
    ],
    "Global Indices": [
        "NK=F", "HSI=F", "DAX=F", "FTSE=F", "CAC=F"
    ]
}

# データ取得
print("先物データ取得中...")
futures_data = {}
error_log = []

for category, tickers_list in futures_tickers.items():
    try:
        data = yf.download(tickers_list, period="10y", group_by="ticker", progress=False)
        if data is not None and not data.empty:
            futures_data[category] = data
        else:
            print(f"{category}: データが取得できませんでした。")
    except Exception as e:
        error_log.append(f"{category}: {e}")

# 1年間で20%以上成長の銘柄をフィルタ
print("過去1年間の20%以上成長銘柄をフィルタリング中...")
filtered_tickers = {}
error_log = []

for genre, tickers_list in futures_tickers.items():
    filtered_genre_tickers = []
    for ticker in tickers_list:
        try:
            data = yf.download(ticker, period="1y", progress=False)
            if not data.empty and "Close" in data.columns:
                close_prices = data['Close'].dropna()
                if len(close_prices) > 1:
                    first_price = close_prices.iloc[0]
                    last_price = close_prices.iloc[-1]
                    growth = (last_price / first_price) - 1
                    if float(growth) >= 0.2:  # 20%以上成長
                        filtered_genre_tickers.append(ticker)
            else:
                error_log.append(f"データが不足: {ticker}")
        except Exception as e:
            error_log.append(f"{ticker}: {e}")
    print(genre, filtered_genre_tickers)
    filtered_tickers[genre] = filtered_genre_tickers

# フィルタリング結果を保存
filtered_tickers_path = "filtered_tickers_with_names.xlsx"
filtered_tickers_data = []

for genre, tickers_list in filtered_tickers.items():
    for ticker in tickers_list:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            company_name = info.get("shortName", "不明") if info else "不明"
            filtered_tickers_data.append({"業種": genre, "銘柄コード": ticker, "企業名": company_name})
        except Exception as e:
            error_log.append(f"{ticker}: 企業名取得エラー - {e}")

df_filtered_tickers = pd.DataFrame(filtered_tickers_data)
df_filtered_tickers.to_excel(filtered_tickers_path, index=False, sheet_name="Filtered Tickers")
print(f"\nフィルタリング後のティッカーリストが保存されました: {filtered_tickers_path}")

# フィルタリング済みのティッカーリストで2年分のデータ取得
print("\nフィルタリング済みティッカーリストでデータを取得中...")
full_data = {}
for genre, tickers_list in filtered_tickers.items():
    if tickers_list:
        try:
            data_downloaded = yf.download(tickers_list, period="2y", group_by="ticker", progress=False)
            if data_downloaded is not None and not data_downloaded.empty:
                full_data[genre] = data_downloaded
            else:
                print(f"{genre}: データが取得できませんでした。")
        except Exception as e:
            print(f"{genre}: データ取得中にエラーが発生しました - {e}")

# 期間と重み
periods = {
    "1週間前": {"days": 5, "weight": 1.3},
    "2週間前": {"days": 10, "weight": 1.15},
    "1か月前": {"days": 21, "weight": 1.0},
    "6週間前": {"days": 30, "weight": 0.9},
    "9週間前": {"days": 45, "weight": 0.8},
    "12週間前": {"days": 60, "weight": 0.7},
}

# スコア計算
genre_scores = []
for genre, data_genre in full_data.items():
    total_score = 0
    for period_name, period_info in periods.items():
        returns = []
        # ティッカー抽出
        if isinstance(data_genre.columns, pd.MultiIndex):
            tickers_in_genre = data_genre.columns.levels[0]
        else:
            # 単一ティッカーの場合の対応
            tickers_in_genre = [genre] if isinstance(data_genre.columns, pd.Index) else []
        for ticker in tickers_in_genre:
            try:
                close_prices = data_genre[ticker]['Close'].dropna()
                if len(close_prices) >= period_info["days"]:
                    recent_return = (close_prices.iloc[-1] / close_prices.iloc[-period_info["days"]] - 1)
                    returns.append(recent_return)
            except KeyError:
                continue
        if returns:
            avg_return = sum(returns) / len(returns)
            total_score += avg_return * period_info["weight"]
    genre_scores.append({"業種": genre, "スコア": total_score})

# スコア結果を保存
scores_path = "genre_scores.xlsx"
df_scores = pd.DataFrame(genre_scores)
df_scores.to_excel(scores_path, index=False, sheet_name="Genre Scores")
print(f"\n業種ごとのスコアが保存されました: {scores_path}")

# スコアの閾値
threshold = 0.2
valid_genres = [genre for genre in genre_scores if genre["スコア"] >= threshold]
if not valid_genres:
    print("スコアが閾値以上の業種がありません。プログラムを終了します。")
    # sys.exit() # 必要に応じて使用
    exit()

print("\nスコアが閾値以上の業種:")
all_top_stocks = []
for genre in valid_genres:
    print(f"{genre['業種']}: スコア {genre['スコア']:.2f}")
    selected_data = full_data.get(genre["業種"])
    if selected_data is None or selected_data.empty:
        continue
    stock_performance = []
    # filtered_tickers[genre["業種"]]を走査
    for ticker in filtered_tickers[genre["業種"]]:
        total_score = 0
        try:
            close_prices = selected_data[ticker]['Close'].dropna()
            for period_name, period_info in periods.items():
                if len(close_prices) >= period_info["days"]:
                    recent_return = (close_prices.iloc[-1] / close_prices.iloc[-period_info["days"]] - 1)
                    total_score += recent_return * period_info["weight"]
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            company_name = info.get("shortName", "不明") if info else "不明"
            stock_performance.append({"業種": genre["業種"], "銘柄コード": ticker, "企業名": company_name, "スコア": total_score})
        except KeyError:
            continue
    top_stocks = sorted(stock_performance, key=lambda x: x["スコア"], reverse=True)[:5]
    all_top_stocks.extend(top_stocks)

# 上位銘柄結果をDataFrame化
df_top_stocks = pd.DataFrame(all_top_stocks, columns=["業種", "銘柄コード", "企業名", "スコア"])

# タイムスタンプを取得
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 新しいファイルパスを作成
top_stocks_path = f"./分析/sakimono_top_stocks_全体用固定_0.2_0.2_{timestamp}.xlsx"
df_top_stocks.to_excel(top_stocks_path, index=False, sheet_name="Top Stocks")
print(f"\nスコアが閾値以上の業種に属するトップ株が保存されました: {top_stocks_path}")

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
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt
import mplfinance as mpf


# Gmail 設定
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
GMAIL_USER = "k.atsuofxtrade@gmail.com"
GMAIL_PASSWORD = "yyyegokbvfcyufnm"

# 送信先リスト
recipient_list = [
    # "k.atsuojp429@gmail.com",
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

import matplotlib.pyplot as plt

def generate_advanced_chart(ticker, save_path, period="5d", interval="15m"):
    """
    高度なチャートを生成する関数
      - ロウソク足チャート
      - 移動平均線 (SMA, EMA)
      - RSI (Relative Strength Index)
    ※15分足など細かい足の場合は、取得期間（period）も短めに設定する必要があります。

    Parameters
    ----------
    ticker : str
        銘柄のティッカーシンボル (例: "AAPL" や "CC=F" など)
    save_path : str
        チャートを保存するパス
    period : str
        データ取得期間 (例: "5d", "1mo", "3mo", "1y")
    interval : str
        データ間隔 (例: "15m", "1d", "1h")
    """
    try:
        # データ取得
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            print(f"[WARNING] {ticker}: データが取得できませんでした。")
            return None

        # カラムが MultiIndex になっている場合はフラット化する
        if isinstance(df.columns, pd.MultiIndex):  # MultiIndex の場合のみ適用
            df.columns = df.columns.droplevel('Ticker')  # 'Ticker' の階層を削除

        # print(df)

        # 必須カラムのチェック
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"[ERROR] {ticker}: missing columns: {missing_cols}")
            return None

        # 各カラムを数値型に変換
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # NaN を含む行は削除
        df.dropna(subset=required_columns, inplace=True)
        if df.empty:
            print(f"[WARNING] {ticker}: 数値データが存在しません。")
            return None

        # インディケーター計算
        df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()      # 10本移動平均
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()               # 20本指数移動平均

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # addplot に渡すデータは 1 次元のリストまたは Series である必要がある
        sma10 = df["SMA_10"].tolist()
        ema20 = df["EMA_20"].tolist()
        rsi   = df["RSI"].tolist()

        add_plots = [
            mpf.make_addplot(sma10, color="blue", linestyle="--", width=1, label="SMA 10"),
            mpf.make_addplot(ema20, color="red", linestyle="-", width=1, label="EMA 20"),
            mpf.make_addplot(rsi,   panel=1, color="green", ylabel="RSI (14)")
        ]

        # チャート描画＆保存
        mpf.plot(
            df,
            type="candle",
            style="yahoo",
            title=f"{ticker} Advanced Chart ({interval})",
            ylabel="Price",
            ylabel_lower="Volume",
            volume=True,
            addplot=add_plots,
            figscale=1.2,  # サイズ調整
            figratio=(12, 8),  # 横縦比
            tight_layout=True,  # レイアウト自動調整
            savefig=dict(fname=save_path, dpi=300, bbox_inches="tight", pad_inches=0.1),
        )
        print(f"[INFO] {ticker}: チャートを保存しました -> {save_path}")
        return save_path

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None

def generate_charts_for_top_stocks(top_stocks, save_dir="./charts", period="5d", interval="15m"):
    """
    上位銘柄のチャートを生成し、ファイルパスをリストで返す関数

    Parameters
    ----------
    top_stocks : list
        銘柄のティッカーシンボルのリスト
    save_dir : str
        チャート画像を保存するディレクトリ
    period : str
        データ取得期間 (例: "5d", "1mo", "3mo", "1y")
    interval : str
        データ間隔 (例: "15m", "1d", "1h")
    
    Returns
    -------
    list
        チャート画像ファイルのパスのリスト
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    chart_files = []
    for ticker in top_stocks:
        save_path = os.path.join(save_dir, f"{ticker}_advanced_chart.png")
        chart_file = generate_advanced_chart(ticker, save_path, period=period, interval=interval)
        if chart_file:
            chart_files.append(chart_file)
    return chart_files

# 日付取得
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

chart_files = generate_charts_for_top_stocks(
    purchase_df_top5["Ticker"].tolist(),
    save_dir="./charts",
    period="5d",    # ※15分足の場合、取得期間を短くすることが望ましい
    interval="1h"
)

try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()  # TLS 暗号化を開始
    server.login(GMAIL_USER, GMAIL_PASSWORD)  # ログイン

    raito = investment / initial_investment
    profit = investment - initial_investment

    # HTML形式の表を作成
    weekday_table_html = weekday_stats.to_html(index=False, justify="center", border=1)
    ticker_table_html = ticker_stats.to_html(index=False, justify="center", border=1)
    top5_stocks_html = purchase_df_top5.to_html(index=False, justify="center", border=1) if not purchase_df_top5.empty else ""
    results_df["Win Rate"] = (results_df["Win Rate"] * 100).round(2)
    results_df = results_df.round(2)
    simulation_results_html = results_df.to_html(index=False, justify="center", float_format="{:.2f}".format, border=1, escape=False) if not results_df.empty else ""

    for recipient in recipient_list:
        # メールの作成
        msg = MIMEMultipart("related")
        msg["From"] = GMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = f"先物　新しく思考した先物購入リストのおすすめ結果 ({current_date}) {int(investment)}円 {raito:.2f}倍"

        # HTML本文を作成
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
                img {{
                    max-width: 100%;
                    height: auto;
                    float: left;
                }}
            </style>
        </head>
        <body>
            <p>{recipient} 様</p>
            <p>平素よりお世話になっております。</p>
            <p>本日の購入リストのおすすめ結果をお送りいたします。</p>
            <p>現在の投資額: {int(investment):,} 円</p>
            <p>初期投資額: {int(initial_investment):,} 円</p>
            <p>レバレッジ: {leverage} 倍</p>
            <p>取引期間: {deal_term} 営業日</p>
            <p>総合勝率: {winrate * 100:.2f} %</p>

            <h3>上位5件の推奨銘柄:</h3>
            {top5_stocks_html}

            <h3>曜日ごとの勝率:</h3>
            {weekday_table_html}

            <h3>銘柄ごとの勝率:</h3>
            {ticker_table_html}

            <h3>上位5社の株価チャート:</h3>
        """

        # 画像をHTMLに埋め込む
        for i, img_path in enumerate(chart_files):
            img_id = f"chart{i+1}"
            body_html += f'<p><img src="cid:{img_id}" alt="Stock Chart {i+1}"></p>'

        body_html += f"""
            <h3>シミュレーション結果:</h3>
            {simulation_results_html}
            <p>詳細につきましては、添付ファイルをご確認ください。</p>
            <p>ご不明な点がございましたら、お気軽にお問い合わせください。</p>
        </body>
        </html>
        """

        msg.attach(MIMEText(body_html, "html"))

        # 画像をメールに添付
        for i, img_path in enumerate(chart_files):
            with open(img_path, "rb") as img_file:
                img = MIMEImage(img_file.read())
                img.add_header("Content-ID", f"<chart{i+1}>")
                img.add_header("Content-Disposition", "inline", filename=os.path.basename(img_path))
                msg.attach(img)

        # 添付ファイルを追加
        for file_path in file_paths:
            if os.path.exists(file_path):
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