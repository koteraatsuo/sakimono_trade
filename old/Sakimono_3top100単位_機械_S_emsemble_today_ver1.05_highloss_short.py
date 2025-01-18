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

# # ティッカーリスト（業種ごとに分類）
# futures_tickers = {
#     "Indices": [
#         "ES=F", "NQ=F", "YM=F", "RTY=F", "VIX=F", "DAX=F", "FTSE=F",
#         "NK=F", "HSI=F", "KOSPI=F", "S&P500", "NASDAQ100"
#     ],
#     "Commodities": [
#         "CL=F", "BZ=F", "NG=F", "HO=F", "RB=F",
#         "GC=F", "SI=F", "PL=F", "PA=F",
#         "HG=F", "ALI=F", "ZC=F", "ZW=F", "ZS=F",
#         "CC=F", "KC=F", "LB=F", "CT=F", "OJ=F"
#     ],
#     "Currencies": [
#         "6E=F", "6J=F", "6A=F", "6C=F", "6B=F", "6N=F", "6S=F",
#         "DX=F"
#     ],
#     "Interest Rates": [
#         "ZB=F", "ZN=F", "ZF=F", "ZT=F",
#         "GE=F", "ED=F"
#     ],
#     "Energy": [
#         "CL=F", "NG=F", "HO=F", "RB=F", "BZ=F",
#         "QL=F", "QA=F"
#     ],
#     "Metals": [
#         "GC=F", "SI=F", "PL=F", "PA=F", "HG=F"
#     ],
#     "Agriculture": [
#         "ZC=F", "ZW=F", "ZS=F", "ZM=F", "ZL=F",
#         "CC=F", "KC=F", "CT=F", "LB=F", "OJ=F"
#     ],
#     "Softs": [
#         "SB=F", "JO=F", "CC=F", "KC=F"
#     ],
#     "Global Indices": [
#         "NK=F", "HSI=F", "DAX=F", "FTSE=F", "CAC=F"
#     ]
# }

# # データ取得
# print("先物データ取得中...")
# futures_data = {}
# error_log = []

# for category, tickers_list in futures_tickers.items():
#     try:
#         data = yf.download(tickers_list, period="10y", group_by="ticker", progress=False)
#         if data is not None and not data.empty:
#             futures_data[category] = data
#         else:
#             print(f"{category}: データが取得できませんでした。")
#     except Exception as e:
#         error_log.append(f"{category}: {e}")

# # 1年間で20%以上成長の銘柄をフィルタ
# print("過去1年間の20%以上成長銘柄をフィルタリング中...")
# filtered_tickers = {}
# error_log = []

# for genre, tickers_list in futures_tickers.items():
#     filtered_genre_tickers = []
#     for ticker in tickers_list:
#         try:
#             data = yf.download(ticker, period="1y", progress=False)
#             if not data.empty and "Close" in data.columns:
#                 close_prices = data['Close'].dropna()
#                 if len(close_prices) > 1:
#                     first_price = close_prices.iloc[0]
#                     last_price = close_prices.iloc[-1]
#                     growth = (last_price / first_price) - 1
#                     if float(growth) >= 0.2:  # 20%以上成長
#                         filtered_genre_tickers.append(ticker)
#             else:
#                 error_log.append(f"データが不足: {ticker}")
#         except Exception as e:
#             error_log.append(f"{ticker}: {e}")
#     print(genre, filtered_genre_tickers)
#     filtered_tickers[genre] = filtered_genre_tickers

# # フィルタリング結果を保存
# filtered_tickers_path = "filtered_tickers_with_names.xlsx"
# filtered_tickers_data = []

# for genre, tickers_list in filtered_tickers.items():
#     for ticker in tickers_list:
#         try:
#             ticker_obj = yf.Ticker(ticker)
#             info = ticker_obj.info
#             company_name = info.get("shortName", "不明") if info else "不明"
#             filtered_tickers_data.append({"業種": genre, "銘柄コード": ticker, "企業名": company_name})
#         except Exception as e:
#             error_log.append(f"{ticker}: 企業名取得エラー - {e}")

# df_filtered_tickers = pd.DataFrame(filtered_tickers_data)
# df_filtered_tickers.to_excel(filtered_tickers_path, index=False, sheet_name="Filtered Tickers")
# print(f"\nフィルタリング後のティッカーリストが保存されました: {filtered_tickers_path}")

# # フィルタリング済みのティッカーリストで2年分のデータ取得
# print("\nフィルタリング済みティッカーリストでデータを取得中...")
# full_data = {}
# for genre, tickers_list in filtered_tickers.items():
#     if tickers_list:
#         try:
#             data_downloaded = yf.download(tickers_list, period="2y", group_by="ticker", progress=False)
#             if data_downloaded is not None and not data_downloaded.empty:
#                 full_data[genre] = data_downloaded
#             else:
#                 print(f"{genre}: データが取得できませんでした。")
#         except Exception as e:
#             print(f"{genre}: データ取得中にエラーが発生しました - {e}")

# # 期間と重み
# periods = {
#     "1週間前": {"days": 5, "weight": 1.3},
#     "2週間前": {"days": 10, "weight": 1.15},
#     "1か月前": {"days": 21, "weight": 1.0},
#     "6週間前": {"days": 30, "weight": 0.9},
#     "9週間前": {"days": 45, "weight": 0.8},
#     "12週間前": {"days": 60, "weight": 0.7},
# }

# # スコア計算
# genre_scores = []
# for genre, data_genre in full_data.items():
#     total_score = 0
#     for period_name, period_info in periods.items():
#         returns = []
#         # ティッカー抽出
#         if isinstance(data_genre.columns, pd.MultiIndex):
#             tickers_in_genre = data_genre.columns.levels[0]
#         else:
#             # 単一ティッカーの場合の対応
#             tickers_in_genre = [genre] if isinstance(data_genre.columns, pd.Index) else []
#         for ticker in tickers_in_genre:
#             try:
#                 close_prices = data_genre[ticker]['Close'].dropna()
#                 if len(close_prices) >= period_info["days"]:
#                     recent_return = (close_prices.iloc[-1] / close_prices.iloc[-period_info["days"]] - 1)
#                     returns.append(recent_return)
#             except KeyError:
#                 continue
#         if returns:
#             avg_return = sum(returns) / len(returns)
#             total_score += avg_return * period_info["weight"]
#     genre_scores.append({"業種": genre, "スコア": total_score})

# # スコア結果を保存
# scores_path = "genre_scores.xlsx"
# df_scores = pd.DataFrame(genre_scores)
# df_scores.to_excel(scores_path, index=False, sheet_name="Genre Scores")
# print(f"\n業種ごとのスコアが保存されました: {scores_path}")

# # スコアの閾値
# threshold = 0.2
# valid_genres = [genre for genre in genre_scores if genre["スコア"] >= threshold]
# if not valid_genres:
#     print("スコアが閾値以上の業種がありません。プログラムを終了します。")
#     # sys.exit() # 必要に応じて使用
#     exit()

# print("\nスコアが閾値以上の業種:")
# all_top_stocks = []
# for genre in valid_genres:
#     print(f"{genre['業種']}: スコア {genre['スコア']:.2f}")
#     selected_data = full_data.get(genre["業種"])
#     if selected_data is None or selected_data.empty:
#         continue
#     stock_performance = []
#     # filtered_tickers[genre["業種"]]を走査
#     for ticker in filtered_tickers[genre["業種"]]:
#         total_score = 0
#         try:
#             close_prices = selected_data[ticker]['Close'].dropna()
#             for period_name, period_info in periods.items():
#                 if len(close_prices) >= period_info["days"]:
#                     recent_return = (close_prices.iloc[-1] / close_prices.iloc[-period_info["days"]] - 1)
#                     total_score += recent_return * period_info["weight"]
#             ticker_obj = yf.Ticker(ticker)
#             info = ticker_obj.info
#             company_name = info.get("shortName", "不明") if info else "不明"
#             stock_performance.append({"業種": genre["業種"], "銘柄コード": ticker, "企業名": company_name, "スコア": total_score})
#         except KeyError:
#             continue
#     top_stocks = sorted(stock_performance, key=lambda x: x["スコア"], reverse=True)[:5]
#     all_top_stocks.extend(top_stocks)

# # 上位銘柄結果をDataFrame化
# df_top_stocks = pd.DataFrame(all_top_stocks, columns=["業種", "銘柄コード", "企業名", "スコア"])
# top_stocks_path = "sakimono_top_stocks_全体用固定_0.2_0.2.xlsx"
# df_top_stocks.to_excel(top_stocks_path, index=False, sheet_name="Top Stocks")
# print(f"\nスコアが閾値以上の業種に属するトップ株が保存されました: {top_stocks_path}")


top_stocks_path = "sakimono_top_stocks_全体用固定_0.2_0.2.xlsx"
# ここから機械学習モデル
file_path = top_stocks_path

df = pd.read_excel(file_path)
df = df[["業種", "銘柄コード", "企業名"]]

tickers = {row["銘柄コード"]: row["企業名"] for _, row in df.iterrows()}
ticker_symbols = list(tickers.keys())

data = yf.download(ticker_symbols, period="10y", progress=False)["Close"].dropna()

# 本日の日付を取得
today = pd.Timestamp(datetime.now().date())

# データの最後の日付が本日と一致する場合、その行を削除
if data.index[-1].date() == today.date():
    print("削除")
    data = data.iloc[:-1]

# 確認: 最後のデータが本日分ではなくなったことを確認
print(data.tail())
initial_investment = 100000
leverage = 10
stop_loss_threshold = 0.03
price_threshold = 5000
lookback_days = 30
deal_term = 5
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
    top_stocks_sim = day_data.head(3)
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
            deal_type ="normal"
            if low_prices.min() <= stop_loss_price:
                exit_price = stop_loss_price
                deal_type ="損切り"
            else:
                exit_price = end_price

            trade_amount = investment / 3
            position_size = trade_amount

            number_of_shares = position_size // start_price
            if number_of_shares == 0:
                continue

            actual_investment = number_of_shares * start_price
            fee = 0.00132 * actual_investment
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
                "deal": deal_type,
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

# 今日買うべき銘柄
end_date = datetime.now()

def create_features_for_today():
    features = []
    tickers_list_today = []
    for ticker_ in ticker_symbols:
        try:
            ticker_data = data[ticker_].dropna()
            if len(ticker_data) < lookback_days:
                continue
            x = ticker_data.iloc[-lookback_days:].pct_change().dropna().values
            if len(x) != lookback_days - 1:
                continue
            features.append(x)
            tickers_list_today.append(ticker_)
        except:
            continue
    return np.array(features), tickers_list_today

X_today, tickers_today = create_features_for_today()
if len(X_today) > 0:
    X_today_scaled = scaler.transform(X_today)
    lgb_probs_today = best_lgb_model.predict_proba(X_today_scaled)[:, 1]
    cat_probs_today = best_cat_model.predict_proba(X_today_scaled)[:, 1]
    xgb_probs_today = best_xgb_model.predict_proba(X_today_scaled)[:, 1]
    ensemble_probs_today = (lgb_probs_today + cat_probs_today + xgb_probs_today) / 3

    today_data = pd.DataFrame({
        "Ticker": tickers_today,
        "Probability": ensemble_probs_today
    })
else:
    today_data = pd.DataFrame(columns=["Ticker", "Probability"])

recommendation_data = today_data.sort_values(by="Probability", ascending=False)

purchase_recommendations_top3 = []
purchase_recommendations_top5 = []

# エグジット日付は単純計算
exit_date = end_date + timedelta(days=deal_term)

for top_n, purchase_recommendations in [(3, purchase_recommendations_top3), (5, purchase_recommendations_top5)]:
    top_stocks_rec = recommendation_data.head(top_n)
    for idx, row in top_stocks_rec.iterrows():
        ticker_ = row["Ticker"]
        company_name = tickers.get(ticker_, "")
        try:
            start_price = data[ticker_].iloc[-1]
            trade_amount = initial_investment / top_n
            position_size = trade_amount
            stop_loss_price = (1 - stop_loss_threshold) * start_price
            stop_loss_amount = stop_loss_threshold * start_price

            number_of_shares = position_size // start_price
            if number_of_shares == 0:
                continue
            actual_investment = number_of_shares * start_price

            purchase_recommendations.append({
                "Entry Date": end_date.strftime('%Y-%m-%d'),
                "Exit Date": exit_date.strftime('%Y-%m-%d'),
                "Term (Business Days)": deal_term,
                "Ticker": ticker_,
                "Company Name": company_name,
                "Current Price": start_price,
                "Stop Loss Amount": stop_loss_amount * int(number_of_shares),
                "Stop Loss Price": stop_loss_price,
                "Shares Bought": int(number_of_shares),
                "Investment per Stock (JPY)": round(actual_investment, 2),
                "Predicted Probability (%)": round(row["Probability"] * 100, 2)
            })
        except Exception as e:
            continue

columns_order = ["Entry Date", "Exit Date", "Term (Business Days)", "Ticker", "Company Name",
                 "Current Price", "Shares Bought", "Stop Loss Amount", "Stop Loss Price", "Investment per Stock (JPY)",
                 "Predicted Probability (%)"]

purchase_df_top3 = pd.DataFrame(purchase_recommendations_top3)[columns_order] if purchase_recommendations_top3 else pd.DataFrame(columns=columns_order)
purchase_df_top5 = pd.DataFrame(purchase_recommendations_top5)[columns_order] if purchase_recommendations_top5 else pd.DataFrame(columns=columns_order)

print("\n上位3件の購入推奨銘柄:")
print(purchase_df_top3)

print("\n上位5件の購入推奨銘柄:")
print(purchase_df_top5)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name_top3 = f"./分析/過去25日間_銘柄選定結果_top3_{timestamp}.xlsx"
file_name_top5 = f"./分析/過去25日間_銘柄選定結果_top5_{timestamp}.xlsx"
purchase_df_top3.to_excel(file_name_top3, index=False)
purchase_df_top5.to_excel(file_name_top5, index=False)
print("\n購入リストが以下のファイルに保存されました。")
print(f" - {file_name_top3}")
print(f" - {file_name_top5}")


# メール送信コードは環境依存のためコメントアウトします

import os
import win32com.client
import time
if True:
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
    file_path_simulation_file_name= os.path.join(current_dir, simulation_file_name)
    try:
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
            mail.Attachments.Add(file_path_simulation_file_name)
            mail.Send()
            time.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}")