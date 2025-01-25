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
best_num = 0
best_score = 0
for num in [1,2,3,4,5]:

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
    deal_term = num
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


    if best_score < investment:
        best_num = num
        best_score = investment    
    
    print(" 1best_score, best_num, num", best_score, best_num, num)
