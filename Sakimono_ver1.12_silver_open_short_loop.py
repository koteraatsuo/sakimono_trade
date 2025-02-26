import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib
from datetime import datetime, timedelta, time as dtime
import json
import os
import time

# ---------------------------
# 初期設定・データ取得
# ---------------------------
folder_path = "./分析"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Excelファイルからティッカーリストを読み込み

top_stocks_path = "silver.xlsx"



file_path = top_stocks_path
df = pd.read_excel(file_path)
df = df[["業種", "銘柄コード", "企業名"]]
tickers = {row["銘柄コード"]: row["企業名"] for _, row in df.iterrows()}
ticker_symbols = list(tickers.keys())
ticker_symbols = df["銘柄コード"].dropna().astype(str).str.strip().tolist()
print(ticker_symbols)
# Yahoo Finance の仕様変更に対応するため、auto_adjust=False と group_by='ticker' を指定
data = yf.download(ticker_symbols, period="10y", progress=False, auto_adjust=False, group_by='ticker')

# 本日の日付・現在時刻を取得
today = pd.Timestamp(datetime.now().date())
now = datetime.now()

# 今日の未確定データ（本日分）が含まれている場合は削除
if data.index[-1].date() == now.date() and now.time() >= datetime.strptime("07:00:00", "%H:%M:%S").time():
    print("削除")
    data = data.iloc[:-1]
print("最後のデータ日付", data.index[-1].date())

# ---------------------------
# 各ティッカーのDataFrameに分離
# ---------------------------
# data.columns は MultiIndex になっているので、第1レベルにティッカー名が入ります
ticker_data_dict = {}
available_tickers = set(data.columns.get_level_values(0).unique())
for ticker in ticker_symbols:
    if ticker in available_tickers:
        ticker_data_dict[ticker] = data.xs(ticker, axis=1, level=0)
    else:
        print(f"Ticker {ticker} は取得されていません。")

# サンプル用に利用可能なティッカーの1つを選択
try:
    sample_ticker = next(iter(ticker_data_dict))
except StopIteration:
    raise ValueError("利用可能なティッカーがありません。ティッカーリストや取得結果を確認してください。")

# ---------------------------
# シミュレーション・機械学習用パラメータ
# ---------------------------
best_num = 0
best_score = 0
record_investments = []
winrates = []

# サンプルとして、利用可能なティッカーの1つのデータのインデックスを利用して日付群を作成
dates = ticker_data_dict[sample_ticker].index.unique()

# ---------------------------
# シミュレーションループ
# ---------------------------
# for num in np.arange(0.001, 0.100, 0.002):

# for num in np.arange(1, 10, 1):
#     print(data.tail())
#     initial_investment = 100000
#     leverage = 5
#     stop_loss_threshold = 0.007
#     price_threshold = 5000
#     lookback_days = 30
#     deal_term = 7
#     interest_rate = 0.028
# for num in np.arange(1, 10, 1):
# # for num in np.arange(0.005, 0.100, 0.001):
#     print(data.tail())
#     initial_investment = 100000
#     leverage = num
#     stop_loss_threshold = 0.012
#     price_threshold = 5000
#     lookback_days = int(6 * 4.1)
#     deal_term = int(6)
#     interest_rate = 0.028

for num in np.arange(0.005, 0.200, 0.005):
# for num in np.arange(5, 100, 5):
# for num in np.arange(1, 20, 1):
    print(data.tail())
    initial_investment = 100000
    leverage = 7
    stop_loss_threshold = 0.03
    price_threshold = 5000
    lookback_days = int(9)
    deal_term = int(3)
    position_num = 1
    interest_rate = 0.028
    Probability_threshold = 0.5


    # initial_investment = 100000
    # leverage = num
    # stop_loss_threshold = 0.007
    # price_threshold = 5000
    # lookback_days = int(30)
    # deal_term = int(7)
    # interest_rate = 0.028

    # 過去lookback_days + deal_termの日付期間を作成
    periods_ml = []
    start_idx = lookback_days
    while start_idx + deal_term < len(dates):
        period = dates[start_idx - lookback_days: start_idx + deal_term]
        periods_ml.append(period)
        start_idx += deal_term

    split_idx = int(len(periods_ml) * 0.9)
    train_periods = periods_ml[:split_idx]
    test_periods = periods_ml[split_idx:]

    # ---------------------------
    # 特徴量・ラベル生成関数
    # ---------------------------
    def create_features_and_labels(periods_input):
        features = []
        labels = []
        dates_list = []
        tickers_list_ = []
        for period_ in periods_input:
            for ticker_ in ticker_symbols:
                if ticker_ not in ticker_data_dict:
                    continue
                try:
                    # Close価格の時系列を取得
                    series = ticker_data_dict[ticker_]['Open']
                    ticker_data = series.loc[period_].dropna()
                    if len(ticker_data) < lookback_days + deal_term:
                        continue
                    # 最新の (lookback_days+deal_term) 日分だけを使用
                    ticker_data = ticker_data.iloc[-(lookback_days+deal_term):]
                    x = ticker_data.iloc[:lookback_days].pct_change().dropna().values
                    if len(x) != lookback_days - 1:
                        continue
                    y = 1 if ticker_data.iloc[-1] > ticker_data.iloc[lookback_days - 1] else 0
                    features.append(x)
                    labels.append(y)
                    dates_list.append(period_[lookback_days - 1])
                    tickers_list_.append(ticker_)
                except Exception as e:
                    print(f"Error processing {ticker_} for period {period_}: {e}")
                    continue
        return np.array(features), np.array(labels), dates_list, tickers_list_

    X_train, y_train, _, _ = create_features_and_labels(train_periods)
    X_test, y_test, dates_test, tickers_test = create_features_and_labels(test_periods)
    if X_train.size == 0:
        print("Warning: X_train が空です。パラメータやデータ期間を見直してください。")
        continue

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
    best_lgb_model = optimize_model(lgb.LGBMClassifier(random_state=42, verbose=-1), lgb_param_grid)
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
    test_data_df = pd.DataFrame({
        "Date": dates_test,
        "Ticker": tickers_test,
        "Probability": ensemble_probs
    })
    unique_dates = test_data_df["Date"].unique()
    winrate = 0
    win_cnt = 0
    total_cnt = 0

    # ---------------------------
    # シミュレーション処理
    # ---------------------------
    for current_date in unique_dates:
        day_data = test_data_df[test_data_df["Date"] == current_date]
        day_data = day_data.sort_values(by="Probability", ascending=False)
        print(float(day_data.head(1)["Probability"].iloc[0]))
        if float(day_data.head(1)["Probability"].iloc[0]) < Probability_threshold:
            continue 
        top_stocks_sim = day_data.head(1)
        total_profit_loss = 0
        total_fees = 0
        for idx, row in top_stocks_sim.iterrows():
            ticker_ = row["Ticker"]
            company_name = tickers.get(ticker_, "")
            try:
                df_ticker = ticker_data_dict[ticker_]
                date_idx = df_ticker.index.get_loc(current_date)
                start_idx_ = date_idx
                end_idx_ = start_idx_ + deal_term
                if end_idx_ >= len(df_ticker):
                    continue
                start_price = df_ticker['Open'].iloc[start_idx_]
                end_price = df_ticker['Open'].iloc[end_idx_]
                # エントリー価格
                start_price = df_ticker['Open'].iloc[start_idx_]
                stop_loss_price = start_price * (1 - stop_loss_threshold)
                exit_price = end_price  # 仮にエグジットがなかった場合は最終日の始値（end_price）

                # エントリー日の日付
                entry_date = df_ticker.index[start_idx_].date()

                # 保有期間内の日付を順次チェック
                for i in range(start_idx_, end_idx_):
                    current_date = df_ticker.index[i].date()
                    # エントリー日の場合：intradayのLowでチェック
                    if current_date == entry_date:
                        if df_ticker['Low'].iloc[i] <= stop_loss_price:
                            exit_price = stop_loss_price
                            break  # エグジットしたのでループ終了
                    else:
                        # 翌日以降の場合は、その日の始値でチェック
                        current_open = df_ticker['Open'].iloc[i]
                        if current_open <= stop_loss_price:
                            exit_price = current_open
                            break
                        if df_ticker['Low'].iloc[i] <= stop_loss_price:
                            exit_price = stop_loss_price
                            break  # エグジットしたのでループ終了                        
                trade_amount = investment
                position_size = trade_amount
                number_of_shares = position_size // start_price
                if number_of_shares == 0:
                    continue
                actual_investment = number_of_shares * start_price
                fee = 0.0025 * leverage * actual_investment
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
                    "Date": current_date,
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
                print(f"Simulation error for ticker {ticker_} on {current_date}: {e}")
                continue
        print(f"Date: {current_date}, Investment: {round(investment,2)}, Total Profit/Loss: {round(total_profit_loss,2)}, Total Fees: {total_fees}, Win Rate: {round(winrate,4)}")

    results_df = pd.DataFrame(simulation_results)
    simulation_file_name = f"./分析/simulation_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    results_df.to_excel(simulation_file_name, index=False)
    print(f"\nシミュレーション結果が '{simulation_file_name}' に保存されました。")

    if best_score < investment:
        best_num = num
        best_score = investment
    record_investments.append(investment)
    winrates.append(winrate)
    
    
    print("best_score, best_num, num", best_score, best_num, num)
    print("winrates", winrates)
    ranks = []
    for inv in record_investments:
        rank = 1 + sum(1 for x in record_investments if x > inv)
        ranks.append(rank)
    print("record_investments:", record_investments)
    print("ranks:", ranks)
