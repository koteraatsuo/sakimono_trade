import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib
import yfinance as yf
from datetime import datetime, timedelta, time as dtime
import json
from dotenv import load_dotenv
import os
import math
import logging
import time
import requests
from bs4 import BeautifulSoup
import re
import wikipedia
import urllib.parse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt
import mplfinance as mpf
from zoneinfo import ZoneInfo
from pandas.tseries.offsets import BusinessDay

# .envファイルを読み込む
load_dotenv()

# フォルダパスを指定
folder_path = "./分析"
os.makedirs(folder_path, exist_ok=True)

# -----------------------------------------------------------------------------
# 特徴量生成などに関する関数群 (変更なし)
# -----------------------------------------------------------------------------
def process_ha_and_streaks(df_rates):
    if not isinstance(df_rates, pd.DataFrame): df_rates = pd.DataFrame(df_rates)
    df_rates.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True, errors='ignore')
    ha_open_values, ha_close_values = [], []
    for i in range(len(df_rates)):
        if i == 0:
            ha_open = (df_rates['Open'].iloc[i] + df_rates['Close'].iloc[i]) / 2
            ha_close = (df_rates['Open'].iloc[i] + df_rates['High'].iloc[i] + df_rates['Low'].iloc[i] + df_rates['Close'].iloc[i]) / 4
        else:
            ha_open = (ha_open_values[i-1] + ha_close_values[i-1]) / 2
            ha_close = (df_rates['Open'].iloc[i] + df_rates['High'].iloc[i] + df_rates['Low'].iloc[i] + df_rates['Close'].iloc[i]) / 4
        ha_open_values.append(ha_open); ha_close_values.append(ha_close)
    df_rates['ha_open'] = ha_open_values; df_rates['ha_close'] = ha_close_values
    df_rates['ha_high'] = df_rates[['ha_open', 'ha_close', 'High']].max(axis=1)
    df_rates['ha_low'] = df_rates[['ha_open', 'ha_close', 'Low']].min(axis=1)
    df_rates['ha_color'] = '陽線'; df_rates.loc[df_rates['ha_open'] > df_rates['ha_close'], 'ha_color'] = '陰線'
    return df_rates

def calculate_rsi(df, period=14):
    df_copy = df.copy(); df_copy['delta'] = df_copy['Open'].diff()
    df_copy['gain'] = np.where(df_copy['delta'] > 0, df_copy['delta'], 0)
    df_copy['loss'] = np.where(df_copy['delta'] < 0, -df_copy['delta'], 0)
    df_copy['avg_gain'] = df_copy['gain'].rolling(window=period, min_periods=1).mean()
    df_copy['avg_loss'] = df_copy['loss'].rolling(window=period, min_periods=1).mean()
    df_copy['RS'] = df_copy['avg_gain'] / df_copy['avg_loss']
    df_copy['RSI'] = 100 - (100 / (1 + df_copy['RS'])); return df_copy[['RSI']]

def calculate_ichimoku(df, conversion_period=3, base_period=27, span_b_period=52, displacement=26):
    high_prices, low_prices = df['High'], df['Low']
    df['tenkan_sen'] = (high_prices.rolling(window=int(conversion_period)).max() + low_prices.rolling(window=int(conversion_period)).min()) / 2
    df['kijun_sen'] = (high_prices.rolling(window=int(base_period)).max() + low_prices.rolling(window=int(base_period)).min()) / 2
    df['tenkan_sen_slope'] = df['tenkan_sen'].diff(); df['kijun_sen_slope'] = df['kijun_sen'].diff()
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(int(displacement))
    df['senkou_span_b'] = ((high_prices.rolling(window=int(span_b_period)).max() + low_prices.rolling(window=int(span_b_period)).min()) / 2).shift(int(displacement))
    return df

def rolling_fft_features(df, target_col='Open', window=60, freq_list=[1,2,3,5,10]):
    df = df.copy(); all_amp, all_phase = {f:[] for f in freq_list}, {f:[] for f in freq_list}
    for i in range(len(df)):
        if i < window:
            for f in freq_list: all_amp[f].append(np.nan); all_phase[f].append(np.nan)
            continue
        data_window = df[target_col].iloc[i-window:i].values; fft_res = np.fft.fft(data_window)
        for f in freq_list:
            amplitude, phase = (np.abs(fft_res[f]), np.angle(fft_res[f])) if f < len(fft_res) else (np.nan, np.nan)
            all_amp[f].append(amplitude); all_phase[f].append(phase)
    for f in freq_list:
        df[f'fft_amp_{f}'] = all_amp[f]; df[f'fft_phase_{f}'] = all_phase[f]
        df[f'fft_amp_{f}_slope'] = df[f'fft_amp_{f}'].diff(); df[f'fft_phase_{f}_slope'] = df[f'fft_phase_{f}'].diff()
    return df

def quinn_fernandes_extrapolation(data, calc_bars=60, harmonic_period=20, freq_tolerance=0.01, bars_to_render=500, n_harmonics=7):
    if len(data) < calc_bars: raise ValueError(f'データが {calc_bars} 本より少ないので計算できません。')
    past_data = data[-calc_bars:]; n = calc_bars; i = np.arange(n)
    m = np.mean(past_data); model = np.zeros(n); residue = past_data - m; params = []
    for _ in range(n_harmonics):
        fft_res = np.fft.fft(residue); freqs = np.fft.fftfreq(n, d=1.0); pos_mask = freqs > 0
        fft_res_pos, freqs_pos = fft_res[pos_mask], freqs[pos_mask]
        idx = np.argmax(np.abs(fft_res_pos)); w_est = 2.0 * np.pi * freqs_pos[idx]
        if abs(w_est) < freq_tolerance: w_est = freq_tolerance
        X = np.column_stack([np.cos(w_est * i), np.sin(w_est * i)])
        try: coeffs, _, _, _ = np.linalg.lstsq(X, residue, rcond=None)
        except np.linalg.LinAlgError: coeffs = np.array([np.nan, np.nan])
        a, b = coeffs; params.append((w_est, a, b))
        model += a * np.cos(w_est * i) + b * np.sin(w_est * i); residue = past_data - (m + model)
    reconst_past = m + model
    future_i = np.arange(n, n + bars_to_render); reconst_future = np.full(bars_to_render, m)
    for w, a, b in params: reconst_future += a * np.cos(w * future_i) + b * np.sin(w * future_i)
    reconst = np.concatenate([reconst_past, reconst_future])
    future_idx = np.arange(len(data) - calc_bars, len(data) - calc_bars + len(reconst))
    return reconst, future_idx

def rolling_quinn_features_no_future(df, calc_bars=59, harmonic_period=10, freq_tolerance=0.01, n_harmonics=15):
    results, indices = [], []; selected_indices = [1, 3, 7, 11, 17, 23]
    for i in range(calc_bars, len(df)):
        current_close = df['Open'].iloc[i]; window_data = df['Open'].iloc[i - calc_bars : i].values
        reconst, _ = quinn_fernandes_extrapolation(data=window_data, calc_bars=calc_bars, harmonic_period=harmonic_period, freq_tolerance=freq_tolerance, bars_to_render=0, n_harmonics=n_harmonics)
        diff_vector = (reconst - current_close) * 10.0
        feature_dict = {f"qf_diff_{j}": diff_vector[j] for j in selected_indices if j < calc_bars}
        results.append(feature_dict); indices.append(df.index[i])
    return pd.DataFrame(results, index=indices)

def prepare_features_2(master_df, num_1=59, num_2=10, conversion_period=2, base_period=7, span_b_period=45, displacement=22, calc_bars=60, bars_to_render=10, harmonic_period=10):
    df = master_df.copy()
    df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True, errors='ignore')
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns: raise ValueError(f"必要なカラム '{col}' がDataFrameに存在しません。")
    if len(df) < calc_bars + 20: raise ValueError(f"データが少なすぎます。")
    df['price_diff_1'] = df['Open'].diff(1) * 10.00
    df = calculate_ichimoku(df, conversion_period, base_period, span_b_period, displacement)
    for diff in [1,2,4,8,12,15]:
        df[f'tenkan_diff{diff}'] = df['tenkan_sen'].diff(diff) * 10.00
        df[f'kijun_diff{diff}'] = df['kijun_sen'].diff(diff) * 10.00
        df[f'senkou_a_diff{diff}'] = df['senkou_span_a'].diff(diff) * 10.00
        df[f'senkou_b_diff{diff}'] = df['senkou_span_b'].diff(diff) * 10.00
    df = rolling_fft_features(df, target_col='Open', window=60, freq_list=[1,2,3,5,10])
    qf_df = rolling_quinn_features_no_future(df, calc_bars=int(60), harmonic_period=20)
    df_merged = df.merge(qf_df, how='inner', left_index=True, right_index=True)
    df_merged = df_merged.drop(columns=['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', "Adj Close", "Volume", 'High', 'Low', 'Close'], errors='ignore')
    df_merged.dropna(inplace=True)
    if df_merged.empty: raise ValueError("特徴量が空です。")
    return df_merged

def create_combined_features(df_price, lookback_days=25, deal_term=5):
    try:
        df_extra = prepare_features_2(df_price.copy(),
            num_1=59, num_2=10, conversion_period=2, base_period=7, span_b_period=45,
            displacement=22, calc_bars=60, bars_to_render=10, harmonic_period=10)
    except ValueError as e: return np.array([]), np.array([]), [], []
    n_extra = df_extra.shape[1]
    X_combined_list, y_list, dates_list, indices_list = [], [], [], []
    for i in range(len(df_price) - lookback_days - deal_term):
        window = df_price['Open'].iloc[i : i + lookback_days]
        if len(window) < lookback_days: continue
        returns = window.pct_change().dropna().values
        if returns.shape[0] != lookback_days - 1: continue
        idx_close = i + lookback_days - 1
        label = 1 if df_price['Open'].iat[idx_close + deal_term] > df_price['Open'].iat[idx_close] else 0
        dt = df_price.index[idx_close]
        extra = df_extra.loc[dt].values if dt in df_extra.index else np.full(n_extra, np.nan)
        combined = np.hstack([returns, extra])
        if np.isnan(combined).any(): continue
        X_combined_list.append(combined); y_list.append(label); dates_list.append(dt); indices_list.append(idx_close)
    if not X_combined_list: return np.array([]), np.array([]), [], []
    return np.vstack(X_combined_list), np.array(y_list, dtype=int), dates_list, indices_list

logger = logging.getLogger(__name__); COMPANY_LIST_FILE = "data_j.xls"; _COMPANY_NAME_MAP = None
def _load_company_name_map():
    global _COMPANY_NAME_MAP
    if _COMPANY_NAME_MAP is None:
        if not os.path.exists(COMPANY_LIST_FILE): _COMPANY_NAME_MAP = {}
        else:
            df = pd.read_excel(COMPANY_LIST_FILE, dtype=str)
            df["コード"] = df["コード"].str.strip().str.zfill(4)
            _COMPANY_NAME_MAP = dict(zip(df["コード"], df["銘柄名"].str.strip()))
    return _COMPANY_NAME_MAP
def get_company_name(ticker): return _load_company_name_map().get(ticker.replace(".T","").zfill(4), "不明")


import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# ==============================================================================
# 1. 最新のETFリストファイルを読み込む (CSV/Excel自動判別)
# ==============================================================================
folder = "./分析"
if not os.path.isdir(folder):
    exit(f"分析フォルダが見つかりません: {folder}")

try:
    # ./分析 フォルダから対象となるファイルを探す
    candidate_files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if (f == "top_30_etfs_by_growth.csv" or  # 元のCSVファイル
           (f.startswith("etf_top30_") and f.endswith(".xlsx"))) # 新しいExcelファイル
    ]
    
    if not candidate_files:
        raise FileNotFoundError(f"分析フォルダ内に 'top_30_etfs_by_growth.csv' または 'etf_top30_*.xlsx' ファイルが見つかりません。")

    # 更新日時が最も新しいファイルを選択
    latest_file = max(candidate_files, key=os.path.getctime)
    print(f"読み込みファイル: {latest_file}")

    # --- 修正箇所：ファイルの拡張子によって読み込み方法を切り替える ---
    if latest_file.endswith(".csv"):
        df = pd.read_csv(latest_file)
        # CSVファイルのカラム名をExcel形式に合わせる
        if 'ティッカー' in df.columns:
            df.rename(columns={'ティッカー': '銘柄コード'}, inplace=True)
            
    elif latest_file.endswith(".xlsx"):
        df = pd.read_excel(latest_file)
        
    else:
        raise ValueError(f"サポートされていないファイル形式です: {os.path.basename(latest_file)}")

    # ティッカーリストと、後で銘柄情報を復元するための辞書を作成
    all_ticker_list = list(df["銘柄コード"].unique())
    ticker_info_map = df.set_index('銘柄コード').to_dict('index')

except (FileNotFoundError, ValueError) as e:
    exit(str(e))
except Exception as e:
    exit(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")


# ==============================================================================
# 2. 読み込んだETFリストを対象に、過去1年間の上昇率で再スクリーニング
# ==============================================================================
print(f"\n{len(all_ticker_list)}銘柄のETFを対象に、過去1年の上昇率（スコア）を計算中...")

try:
    data_all = yf.download(
        all_ticker_list,
        period="1y",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True
    )

    all_data = []
    for ticker in all_ticker_list:
        if ticker in data_all.columns.get_level_values(0):
            prices = data_all[ticker]["Close"].dropna()
            
            if len(prices) >= 2 and prices.iloc[0] > 0:
                growth = (prices.iloc[-1] / prices.iloc[0]) - 1
                
                if growth > 0:
                    all_data.append({"銘柄コード": ticker, "スコア": float(growth)})
    
    if not all_data:
        raise ValueError("スコア計算対象の銘柄（1年間の上昇率がプラスのETF）がありませんでした。")

    top_ranked = pd.DataFrame(all_data).sort_values(by="スコア", ascending=False)

    top_ranked["銘柄名"] = top_ranked["銘柄コード"].apply(lambda x: ticker_info_map[x].get('銘柄名', 'N/A'))
    top_ranked["カテゴリ"] = top_ranked["銘柄コード"].apply(lambda x: ticker_info_map[x].get('カテゴリ', 'N/A'))
    
    top_ranked["スコア (%)"] = (top_ranked["スコア"] * 100).round(2)
    top_ranked = top_ranked[["銘柄コード", "銘柄名", "カテゴリ", "スコア (%)"]]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(folder, f"etf_ranked_top_{len(top_ranked)}_{timestamp}.xlsx")
    
    top_ranked.to_excel(output_path, index=False)
    print(f"\n再計算されたETFランキングが保存されました: {output_path}")

except Exception as e:
    print(f"\n銘柄の再計算中にエラーが発生しました: {e}")
    exit("処理を中断します。")
# ==============================================================================
# 2. 学習・予測対象データの取得 (10年分)
# ==============================================================================
df_tickers = pd.read_excel(output_path)
ticker_symbols = list(df_tickers["銘柄コード"])
print(f"\n学習・予測対象の銘柄（{len(ticker_symbols)}件）の10年分データを取得中...")
data = yf.download(ticker_symbols, period="10y", progress=False, auto_adjust=False, group_by='ticker')
ticker_data_dict = {ticker: data.xs(ticker, axis=1, level=0).copy() for ticker in ticker_symbols if ticker in data.columns.get_level_values(0)}

# =================================================================================
# 3. モデルの準備（条件に応じて学習または読み込み）
# =================================================================================
print("\nモデルの準備（必要に応じて学習・再学習）を開始します...")
OUTPUT_DIR = "saved_models_ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)

lgb_models_dict, xgb_models_dict, cat_models_dict, scalers_dict = {}, {}, {}, {}
lookback_days = 25; deal_term = 5
ONE_MONTH_AGO = datetime.now() - timedelta(days=30)

for ticker in ticker_symbols:
    lgb_path = os.path.join(OUTPUT_DIR, f"{ticker}_lgb_model.pkl")
    xgb_path = os.path.join(OUTPUT_DIR, f"{ticker}_xgb_model.pkl")
    cat_path = os.path.join(OUTPUT_DIR, f"{ticker}_cat_model.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, f"{ticker}_scaler.pkl")
    
    all_files_exist = all(os.path.exists(p) for p in [lgb_path, xgb_path, cat_path, scaler_path])
    
    should_train = False
    if not all_files_exist:
        should_train = True
        print(f"--- {ticker}: モデルファイルが不足しているため、新規に学習します。 ---")
    else:
        last_modified_date = datetime.fromtimestamp(os.path.getmtime(scaler_path))
        if last_modified_date < ONE_MONTH_AGO:
            should_train = True
            print(f"--- {ticker}: モデルが1ヶ月以上古いため({last_modified_date.strftime('%Y-%m-%d')})、再学習します。 ---")

    if should_train:
        df_price = ticker_data_dict.get(ticker)
        if df_price is None or df_price.empty or len(df_price) < 200:
            print(f"--- {ticker}: 学習データが不足しているためスキップ。 ---")
            continue
            
        try:
            print(f"  > 特徴量生成中...")
            X, y, _, _ = create_combined_features(df_price, lookback_days=lookback_days, deal_term=deal_term)
            
            if len(X) < 50:
                print(f"--- {ticker}: 特徴量生成後のサンプル数が不足({len(X)}件)のためスキップ。 ---")
                continue

            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            print(f"  > LightGBM 学習中...")
            lgb_model = lgb.LGBMClassifier(random_state=42).fit(X_scaled, y)
            print(f"  > XGBoost 学習中...")
            xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X_scaled, y)
            print(f"  > CatBoost 学習中...")
            cat_model = CatBoostClassifier(random_state=42, verbose=0, iterations=500).fit(X_scaled, y)

            joblib.dump(lgb_model, lgb_path); joblib.dump(xgb_model, xgb_path)
            joblib.dump(cat_model, cat_path); joblib.dump(scaler, scaler_path)
            print(f"--- {ticker}: 3モデルとスケーラーを保存しました ---")

            lgb_models_dict[ticker] = lgb_model; xgb_models_dict[ticker] = xgb_model
            cat_models_dict[ticker] = cat_model; scalers_dict[ticker] = scaler

        except Exception as e:
            print(f"--- {ticker} の学習中にエラーが発生: {e} ---")
            continue
    else:
        print(f"--- {ticker}: 既存の最新モデルを読み込みます。 ---")
        try:
            lgb_models_dict[ticker] = joblib.load(lgb_path)
            xgb_models_dict[ticker] = joblib.load(xgb_path)
            cat_models_dict[ticker] = joblib.load(cat_path)
            scalers_dict[ticker] = joblib.load(scaler_path)
        except Exception as e:
            print(f"--- {ticker} のモデル読み込み中にエラー: {e} ---")

print("\n全ティッカーのモデル準備が完了しました。")
print(f"ロード済みモデル数: LightGBM({len(lgb_models_dict)}), XGBoost({len(xgb_models_dict)}), CatBoost({len(cat_models_dict)})")

# ==============================================================================
# 5. シミュレーションの実行 (アンサンブル予測)
# ==============================================================================
print("\nアンサンブルモデルによるシミュレーションを開始します...")
simulation_results = []
initial_investment = 2000000; leverage = 1; stop_loss_threshold = 0.3

for ticker in ticker_symbols:
    data_exists = ticker in ticker_data_dict and not ticker_data_dict[ticker].empty
    models_exist = all(ticker in d for d in [lgb_models_dict, xgb_models_dict, cat_models_dict, scalers_dict])

    if not (data_exists and models_exist):
        continue

    df_price = ticker_data_dict[ticker]
    lgb_model, xgb_model, cat_model = lgb_models_dict[ticker], xgb_models_dict[ticker], cat_models_dict[ticker]
    scaler = scalers_dict[ticker]
    
    X_sim, _, _, indices_sim = create_combined_features(df_price, lookback_days=lookback_days, deal_term=deal_term)
    if len(X_sim) == 0: continue
    
    X_sim_scaled = scaler.transform(X_sim)
    pred_probs = (lgb_model.predict_proba(X_sim_scaled)[:, 1] + xgb_model.predict_proba(X_sim_scaled)[:, 1] + cat_model.predict_proba(X_sim_scaled)[:, 1]) / 3.0
    
    sim_count = max(1, int(len(X_sim) * 0.1))
    next_trade_available_idx = 0 
    for sample_idx in range(len(X_sim) - sim_count, len(X_sim)):
        if sample_idx >= next_trade_available_idx:
            start_idx = indices_sim[sample_idx]
            end_idx = start_idx + deal_term
            if end_idx >= len(df_price) or df_price['Open'].iloc[start_idx] == 0: continue
            start_price = df_price['Open'].iloc[start_idx]
            exit_price = df_price['Open'].iloc[end_idx]
            stop_loss_price = start_price * (1 - stop_loss_threshold)
            for i in range(start_idx, end_idx):
                if df_price['Low'].iloc[i] <= stop_loss_price: exit_price = stop_loss_price; break
            number_of_shares = initial_investment // start_price
            if number_of_shares == 0: continue
            profit_loss = (exit_price - start_price) * number_of_shares * leverage
            simulation_results.append({
                "Ticker": ticker, "Company Name": get_company_name(ticker), "Simulation Date": df_price.index[start_idx].date(),
                "Start Price": start_price, "Stop Loss Price": stop_loss_price, "End Price": df_price['Open'].iloc[end_idx],
                "Exit Price": exit_price, "Price Difference": exit_price - start_price, "Profit/Loss (JPY)": round(profit_loss, 2),
                "Predicted Probability": round(pred_probs[sample_idx], 3)
            })
            next_trade_available_idx = sample_idx + deal_term

df_simulation = pd.DataFrame(simulation_results)
simulation_file_name = f"./分析/simulation_results_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
if not df_simulation.empty:
    df_simulation.to_excel(simulation_file_name, index=False)
    print(f"\nシミュレーション結果が '{simulation_file_name}' に保存されました。")
else:
    simulation_file_name = ""
    print("\nシミュレーション結果はありませんでした。")

# ==============================================================================
# 6. 再投資シミュレーションと結果分析
# ==============================================================================
current_portfolio = initial_investment; win_rate = 0.0; df_portfolio_progression = pd.DataFrame()
weekday_winrates_output_file, ticker_winrates_output_file = "", ""

if not df_simulation.empty:
    simulation_dates = sorted(df_simulation["Simulation Date"].unique())
    portfolio_progression, current_holdings = [], []
    total_trades, winning_trades = 0, 0
    MAX_HOLDINGS = 3
    for sim_date in simulation_dates:
        holdings_after_sell = []
        for holding in current_holdings:
            if sim_date >= (holding["enter_date"] + pd.DateOffset(days=deal_term)).date():
                total_trades += 1; row = holding["row_data"]
                percentage_return = ((row["Exit Price"] - row["Start Price"]) / row["Start Price"]) * leverage if row["Start Price"] > 0 else 0
                if percentage_return > 0: winning_trades += 1
                current_portfolio *= (1 + (percentage_return / MAX_HOLDINGS))
                portfolio_progression.append({"Date": sim_date, "Event": "SELL", "Ticker": row["Ticker"], "Trade Return (%)": percentage_return * 100, "Updated Portfolio": current_portfolio, "Win Rate (%)": (winning_trades / total_trades * 100)})
            else: holdings_after_sell.append(holding)
        current_holdings = holdings_after_sell
        available_slots = MAX_HOLDINGS - len(current_holdings)
        if available_slots > 0:
            df_deals = df_simulation[df_simulation["Simulation Date"] == sim_date]
            df_new_buys = df_deals[~df_deals["Ticker"].isin([h["ticker"] for h in current_holdings])].sort_values(by="Predicted Probability", ascending=False).head(available_slots)
            for _, row in df_new_buys.iterrows():
                current_holdings.append({"enter_date": sim_date, "ticker": row["Ticker"], "row_data": row})
                portfolio_progression.append({"Date": sim_date, "Event": "BUY", "Ticker": row["Ticker"], "Trade Return (%)": None, "Updated Portfolio": current_portfolio, "Win Rate (%)": (winning_trades / total_trades * 100) if total_trades > 0 else 0})
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    df_portfolio_progression = pd.DataFrame(portfolio_progression)
    trade_log_file_name = f"./分析/trade_log_max3_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"; df_portfolio_progression.to_excel(trade_log_file_name, index=False)
    
    df_simulation["Date"] = pd.to_datetime(df_simulation["Simulation Date"]); df_simulation["Weekday"] = df_simulation["Date"].dt.dayofweek
    weekday_stats = df_simulation.groupby("Weekday").apply(lambda x: pd.Series({"勝利数": (x["Profit/Loss (JPY)"] > 0).sum(), "取引数": len(x), "勝率": (x["Profit/Loss (JPY)"] > 0).mean()})).reset_index()
    weekday_stats["曜日"] = weekday_stats["Weekday"].map({0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"})
    weekday_winrates_output_file = f"./分析/weekday_winrates_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"; weekday_stats[["曜日", "勝利数", "取引数", "勝率"]].to_excel(weekday_winrates_output_file, index=False)
    
    ticker_stats = df_simulation.groupby("Ticker").apply(lambda x: pd.Series({"勝利数": (x["Profit/Loss (JPY)"] > 0).sum(), "取引数": len(x), "勝率": (x["Profit/Loss (JPY)"] > 0).mean()})).reset_index()
    ticker_stats["Company Name"] = ticker_stats["Ticker"].apply(get_company_name)
    ticker_winrates_output_file = f"./分析/ticker_winrates_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"; ticker_stats[["Ticker", "Company Name", "勝利数", "取引数", "勝率"]].to_excel(ticker_winrates_output_file, index=False)

# ==============================================================================
# ▼▼▼ 7. 今日の予測と推奨銘柄のリストアップ（修正済みのセクション） ▼▼▼
# ==============================================================================


def get_daily_ohlc_from_hourly(ticker, period="10y"):
    """
    指定ティッカーの1時間足データを取得し、日足の「始値」「高値」「安値」「終値」を持つ DataFrame を返す関数。
    処理前後にログを出力します。
    """
    # 1. 1時間足データをダウンロード
    df_hourly = yf.download(ticker, period=period, interval="1h", progress=False)
    if df_hourly.empty:
        print(f"  {ticker}: 1時間足データ取得できず。")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"], dtype=float)

    # リサンプリング前の直近25本を表示
    print("=" * 50)
    print(f"[{ticker}] リサンプリング前（直近 25 本の 1 時間足）")
    print(df_hourly.tail(25))
    print("=" * 50)

    # 2. 日足のOHLCをリサンプリング
    df_hourly = df_hourly.sort_index()  # 時系列ソート
    daily_open  = df_hourly['Open'].resample('D').first()
    daily_high  = df_hourly['High'].resample('D').max()
    daily_low   = df_hourly['Low'].resample('D').min()
    daily_close = df_hourly['Close'].resample('D').last()

    df_daily = pd.concat(
        [daily_open, daily_high, daily_low, daily_close],
        axis=1,
        keys=['Open', 'High', 'Low', 'Close']
    ).dropna()

    # リサンプリング後の直近5日を表示
    print(f"[{ticker}] リサンプリング後（日足 OHLC）")
    print(df_daily.tail(1005))
    print("=" * 50)

    return df_daily


from zoneinfo import ZoneInfo

def get_market_threshold_time():
    """
    市場の取引開始時刻（閾値）を自動で返す関数です。
    ここでは、例として米国東部時間のマーケットを想定し、
    夏時間（DST）が適用されている場合は06:00、そうでなければ07:00を返します。
    """
    # 対象のマーケットの現地タイムゾーンを指定
    market_tz = ZoneInfo("America/New_York")
    now_market = datetime.now(market_tz)
    
    # DSTが適用されているかどうかで閾値を設定
    if now_market.dst() != timedelta(0):
        # 夏時間の場合
        threshold = dtime(22, 31, 0)
    else:
        # 冬時間の場合
        threshold = dtime(23, 31, 0)
    
    return threshold

# def create_features_for_today():
#     """
#     1) ティッカーごとに1時間足DL → 日足の終値を作成 (get_daily_close_from_hourly)
#     2) 最後に(銘柄数, lookback_days-1)の形で特徴量を返す
#     3) 併せて「利用できる銘柄のリスト」も返す
#     """
#     features = []
#     tickers_list_today = []

#     for ticker in ticker_symbols:
#         print(f"\n[INFO] 1時間足データダウンロード → 日足リサンプリング → {ticker}")
#         df_daily = get_daily_close_from_hourly(ticker, period="3mo")

#         # データがない場合はスキップ
#         if df_daily.empty:
#             continue
        
#         print("当日分(未確定)削除前")
#         print("="*50)
#         print(df_daily.tail(3))
   

#         # 今日の日付と午前7時以降の条件をチェックして行を削除
#         if len(df_daily) > 0:
#             print("チャートデータ：", df_daily.index[-1].date())
#             print("現在：", now.date())
#             print("現在：", now.time() )
#             print("閾時間：", get_market_threshold_time())
#             # 最新データの取引日を求めるため、7時間戻す
#             threshold = get_market_threshold_time()

#             # 現在時刻が07:00以降かつ、最新行の実際の取引日が今日の場合は、最後の行を除外する
#             if now.time() >= threshold and df_daily.index[-1].date() == now.date():
#                 print("="*50)
#                 print("="*50)
#                 print("="*50)
#                 print("  当日分(未確定)を削除")
#                 df_daily = df_daily.iloc[:-1]
#                 print("="*50)
#                 print("="*50)
#                 print("="*50)

#         print("当日分(未確定)削除後")
#         print(df_daily.tail(3))
        
#         # 欠損除去
#         df_daily = df_daily.dropna()

#         # 30日分足りないならスキップ
#         if len(df_daily) < lookback_days:
#             print(f"  {ticker}: 過去{lookback_days}日分のデータがありません。スキップ。")
#             continue

#         # 過去30日分の終値を取り出し、pct_change() → (29,) のリターンベクトル
#         df_lookback = df_daily.iloc[-lookback_days:]
#         x = df_lookback.pct_change().dropna().values  # shape->(29,)

#         if len(x) != lookback_days - 1:
#             print(f"  {ticker}: リターン数が {len(x)} 個しかないためスキップ。")
#             continue

#         # 2次元化 => shape (1, 29)
#         x_2d = x.reshape(1, -1)

#         features.append(x_2d)
#         tickers_list_today.append(ticker)

#     # 全部終わったら、(銘柄数, 29) の形にまとめて返す
#     if len(features) == 0:
#         print("[INFO] No valid ticker data to create features.")
#         X_today = np.empty((0, lookback_days - 1))
#     else:
#         X_today = np.vstack(features)

#     return X_today, tickers_list_today


import requests
from bs4 import BeautifulSoup

def get_japanese_name(ticker: str) -> str:
    """
    yfinance のティッカー（例: '7203.T'）から
    Yahoo!ファイナンス日本版のページタイトルを解析し、
    日本語の会社名を返します。
    """
    url = f"https://finance.yahoo.co.jp/quote/{ticker}"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    title = soup.title.string  # 例: "トヨタ自動車(7203.T) : 株価｜株式 - Yahoo!ファイナンス"
    # 「(ティッカー」で分割して前半を会社名とみなす
    name = title.split("(")[0].strip()
    return name


# def get_us_company_name_by_scrape(ticker: str) -> str:
#     """
#     Yahoo! Finance（米国版）のページタイトルを解析して
#     英語の会社名を返す。
#     URL例: https://finance.yahoo.com/quote/AAPL
#     """
#     url = f"https://finance.yahoo.com/quote/{ticker}"
#     resp = requests.get(url, timeout=5)
#     resp.raise_for_status()
#     soup = BeautifulSoup(resp.text, "lxml")
#     title = soup.title.string
#     # タイトル例: "Apple Inc. (AAPL) Stock Price, News, Quote & History - Yahoo Finance"
#     # 「 (ティッカー」で分割して前半を会社名とみなす
#     name = title.split(f" ({ticker})")[0].strip()
#     return name



def create_features_for_today_combined(ticker_symbols,
                                       lookback_days=25,
                                       deal_term=5):
    """
    1) ティッカーごとに1時間足→日足リサンプリング
    2) 過去 lookback_days 日のリターン (pct_change→(lookback_days-1,))
    3) prepare_features_2 で df_extra を計算 (全日分)
    4) 最終日に対応する df_extra.loc[last_date] を取り出して結合
    5) (銘柄数, (lookback_days-1)+n_extra) の X_today と
       利用可能ティッカーリストを返す
    """
    X_rows = []
    tickers_ok = []

    for ticker in ticker_symbols:
        print(f"[INFO] Processing {ticker}...")
        df_daily = get_daily_ohlc_from_hourly(ticker, period="2y")
        if df_daily.empty:
            continue
        # MultiIndex の第０レベルを落とす
        df_daily.columns = df_daily.columns.droplevel('Ticker')

        # 確認
        print(df_daily.columns)
        df_extra = prepare_features_2(df_daily,
                num_1=59,
                num_2=10,
                conversion_period=2,
                base_period=7,
                span_b_period=45,
                displacement=22,
                calc_bars=60,
                bars_to_render=10,
                harmonic_period=10
            )

        #―― 未確定当日分の削除ロジックはお手持ちのものをそのまま――
        # 省略...
        df_daily = df_daily.dropna()
        if len(df_daily) < lookback_days:
            continue
        # print(df_daily.colums)
        # (1) 過去 lookback_days 日の終値でリターンベクトル
        window = df_daily['Open'].iloc[-lookback_days:]
        returns = window.pct_change().dropna().values
        if len(returns) != lookback_days - 1:
            continue
        # returns: shape=(lookback_days-1,)

        # (2) prepare_features_2 で追加特徴量 DataFrame を一度だけ計算
        #     ※内部で 'Close'→'close' リネームを済ませておいてください

        n_extra = df_extra.shape[1]

        # (3) 最終日インデックスを取得
        last_dt = df_daily.index[-1]
        if last_dt in df_extra.index:
            extra = df_extra.loc[last_dt].values
        else:
            extra = np.zeros(n_extra, dtype=float)

        # (4) ドッキング
        combined = np.hstack([returns, extra])  # shape=(lookback_days-1 + n_extra,)

        X_rows.append(combined)
        tickers_ok.append(ticker)

    # (5) テーブル化
    if X_rows:
        X_today = np.vstack(X_rows)
    else:
        X_today = np.empty((0, lookback_days - 1 + n_extra))

    return X_today, tickers_ok

# --- 今日の特徴量作成 ---
# X_today, tickers_today = create_features_for_today()
X_today, tickers_today = create_features_for_today_combined(ticker_symbols,
                                       lookback_days=25,
                                       deal_term=5)
print(f"\n最終的な X_today.shape = {X_today.shape}")
print(f"銘柄数 = {len(tickers_today)}")

# --- 各ティッカーのモデルを用いて本日の予測確率を算出 --




# --- 各ティッカーのモデルを用いて本日の予測確率を算出 ---
predictions = []
if len(X_today) > 0:
    for ticker, feature_vector in zip(tickers_today, X_today):
        if all(ticker in d for d in [lgb_models_dict, xgb_models_dict, cat_models_dict, scalers_dict]):
            scaler = scalers_dict[ticker]
            lgb_model, xgb_model, cat_model = lgb_models_dict[ticker], xgb_models_dict[ticker], cat_models_dict[ticker]
            x_scaled = scaler.transform(feature_vector.reshape(1, -1))
            prob_lgb = lgb_model.predict_proba(x_scaled)[0, 1]
            prob_xgb = xgb_model.predict_proba(x_scaled)[0, 1]
            prob_cat = cat_model.predict_proba(x_scaled)[0, 1]
            prob = (prob_lgb + prob_xgb + prob_cat) / 3.0
            predictions.append({"Ticker": ticker, "Probability": prob})





today_data = pd.DataFrame(predictions).sort_values("Probability", ascending=False)

# 予測確率の降順にソート（必要なら重複処理も可能）
recommendation_data = today_data.sort_values(by="Probability", ascending=False)
print("本日の各ティッカー別予測結果:")
print(recommendation_data)

# --- 購入推奨銘柄の算出 ---
# 以下は例として、上位3件、5件、6件の推奨銘柄リストを作成する例です。
purchase_recommendations_top3 = []
purchase_recommendations_top5 = []
purchase_recommendations_top6 = []

# エグジット日付の計算
end_date = datetime.now()
from pandas.tseries.offsets import BusinessDay
exit_date = end_date + BusinessDay(deal_term)
print("Exit Date (deal_term 営業日後):", exit_date)

for top_n, purchase_recommendations in [(3, purchase_recommendations_top3), (15, purchase_recommendations_top5), (6, purchase_recommendations_top6)]:
    top_stocks_rec = recommendation_data.head(top_n)
    for idx, row in top_stocks_rec.iterrows():
        ticker_ = row["Ticker"]
        company_name = get_company_name(ticker_)
        try:
            # data[ticker_]は各ティッカーの日足終値Series（もしくはDataFrame）の辞書
            # 終値をスカラーに変換する
            start_price = float(data[ticker_]["Open"].iloc[-1])
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
            print(f"{ticker_}: エラーが発生しました。", e)
            continue

columns_order = ["Entry Date", "Exit Date", "Term (Business Days)", "Ticker", "Company Name",
                 "Current Price", "Shares Bought", "Stop Loss Amount", "Stop Loss Price", "Investment per Stock (JPY)",
                 "Predicted Probability (%)"]

purchase_df_top3 = pd.DataFrame(purchase_recommendations_top3)[columns_order] if purchase_recommendations_top3 else pd.DataFrame(columns=columns_order)
purchase_df_top5 = pd.DataFrame(purchase_recommendations_top5)[columns_order] if purchase_recommendations_top5 else pd.DataFrame(columns=columns_order)
purchase_df_top6 = pd.DataFrame(purchase_recommendations_top6)[columns_order] if purchase_recommendations_top6 else pd.DataFrame(columns=columns_order)

print("\n上位3件の購入推奨銘柄:")
print(purchase_df_top3)

print("\n上位5件の購入推奨銘柄:")
print(purchase_df_top5)

print("\n上位6件の購入推奨銘柄:")
print(purchase_df_top6)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name_top3 = f"./分析/過去25日間_銘柄選定結果_15_top3_{timestamp}.xlsx"
file_name_top5 = f"./分析/過去25日間_銘柄選定結果_15_top5_{timestamp}.xlsx"
file_name_top6 = f"./分析/過去25日間_銘柄選定結果_15_top6_{timestamp}.xlsx"
purchase_df_top3.to_excel(file_name_top3, index=False)
purchase_df_top5.to_excel(file_name_top5, index=False)
purchase_df_top6.to_excel(file_name_top6, index=False)
print("\n購入リストが以下のファイルに保存されました。")
print(f" - {file_name_top3}")
print(f" - {file_name_top5}")
print(f" - {file_name_top6}")




time.sleep(3)
import smtplib
from dotenv import load_dotenv
import os

# 1. まず.envファイルを読み込む
load_dotenv()
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
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD")




# ==== 設定セクション ====
EMAIL_BASE_DIR = "./mail_adress"
os.makedirs(EMAIL_BASE_DIR, exist_ok=True)

# 受信先を外部 CSV から読み込む
RECIPIENT_CSV = os.path.join(EMAIL_BASE_DIR, "recipients.csv")
def load_recipients_from_csv(path):
    """CSV の 'email' 列からコメントや空行を除いて読み込む"""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, dtype=str)
    # 空白／NaN 行は除去
    df = df.dropna(subset=['email'])
    # 前後空白とコメント行（先頭'#'）を排除
    emails = [
        e.strip() for e in df['email'].tolist()
        if e.strip() and not e.strip().startswith('#')
    ]

    return emails

# 読み込んだリストをグローバル変数に
recipient_list = load_recipients_from_csv(RECIPIENT_CSV)

# 送信先リスト
recipient_list = [
    # "k.atsuojp429@gmail.com",
    # "k.atsuo-jp@outlook.com",
    "kotera2hjp@gmail.com",
    # "kotera2hjp@outlook.jp",
    # "kotera2hjp@yahoo.co.jp",
    "k.atsuofxtrade@gmail.com",
    "satosato.k543@gmail.com",
    "clubtrdr@gmail.com"
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



import re
import wikipedia
import urllib.parse

# TradingViewリンク生成関数
def to_tradingview_link(ticker: str) -> str:
    if ticker.endswith(".T"):
        code = ticker.replace(".T", "")
        url = (
            f"https://www.tradingview.com/chart/"
            f"?symbol=TSE:{code}"
            f"&interval=D"
            f"&chartType=1"
        )
    else:
        url = (
            f"https://www.tradingview.com/chart/"
            f"?symbol={ticker}"
            f"&interval=D"
            f"&chartType=1"
        )
    return f'<a href="{url}" target="_blank">{ticker}</a>'

# Wikipediaリンク生成関数（日本語版をimport wikipediaで検索）
def to_wikipedia_link(company_name: str) -> str:
    try:
        wikipedia.set_lang("ja")
        results = wikipedia.search(company_name, results=1)
        if results:
            slug = results[0].replace(" ", "_")
            url = f"https://ja.wikipedia.org/wiki/{slug}"
            return f'<a href="{url}" target="_blank">Wiki</a>'
    except Exception:
        pass
    return ""

# Homepageリンク生成関数
def fetch_homepage(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        site = info.get("website", "").rstrip("/")
        if site:
            return f'<a href="{site}" target="_blank">Home</a>'
    except Exception:
        pass
    return ""


def to_x_link(company_name: str) -> str:
    # company_nameが文字列でない場合（NaNなど）を考慮し、空の文字列に変換する
    if not isinstance(company_name, str):
        company_name = ""
        
    query = urllib.parse.quote_plus(company_name)
    url   = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
    return f'<a href="{url}" target="_blank">Xで最新を見る</a>'

# --- purchase_df_top5 の拡張 ---
raw_tickers   = purchase_df_top5["Ticker"].str.replace(r"<.*?>", "", regex=True)
company_names = purchase_df_top5["Company Name"].tolist()

# 既存のリンク生成部分
tv_links   = []
wiki_links = []
home_links = []
x_links    = []

for ticker, cname in zip(raw_tickers, company_names):
    tv_links.append(to_tradingview_link(ticker))
    wiki_links.append(to_wikipedia_link(cname))
    home_links.append(fetch_homepage(ticker))
    x_links.append(to_x_link(cname))

purchase_df5 = purchase_df_top5.copy()
purchase_df5["Ticker"]   = tv_links
purchase_df5["Wiki"]     = wiki_links
purchase_df5["Homepage"] = home_links
purchase_df5["Xリンク"]  = x_links

top5_stocks_html = purchase_df5.to_html(
    index=False,
    justify="center",
    border=1,
    escape=False
)

# 日付取得
current_date = datetime.datetime.now().strftime("%Y-%m-%d")


# 送信前に数値をクリーニング
if math.isnan(current_portfolio):
    current_portfolio = 0
if math.isnan(initial_investment):
    initial_investment = 0




try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()  # TLS 暗号化を開始
    server.login(GMAIL_USER, GMAIL_PASSWORD)  # ログイン

    raito = current_portfolio / initial_investment
    profit = current_portfolio - initial_investment

    # HTML形式の表を作成

    # DataFrameのTicker列をリンク付きに変換
    # purchase_df_top5['Ticker'] = purchase_df_top5['Ticker'].apply(to_tradingview_link)
    purchase_df_top3['Ticker'] = purchase_df_top3['Ticker'].apply(to_tradingview_link)
    purchase_df_top6['Ticker'] = purchase_df_top6['Ticker'].apply(to_tradingview_link)


    # # HTMLに変換するときは escape=False を指定
    # top5_stocks_html = purchase_df_top5.to_html(
    #     index=False,
    #     justify="center",
    #     border=1,
    #     escape=False
    # )


    # 2) ticker_stats を作成した直後にリンク化
    ticker_stats['Ticker'] = ticker_stats['Ticker'].apply(to_tradingview_link)

    # 3) HTML に変換するときは escape=False を指定
    ticker_table_html = ticker_stats.to_html(
        index=False,
        justify="center",
        border=1,
        escape=False
    )


    weekday_table_html = weekday_stats.to_html(index=False, justify="center", border=1)
    # ticker_table_html = ticker_stats.to_html(index=False, justify="center", border=1)
    # top5_stocks_html = purchase_df_top5.to_html(index=False, justify="center", border=1) if not purchase_df_top5.empty else ""
    df_portfolio_progression["Win Rate (%)"] = (df_portfolio_progression["Win Rate (%)"]).round(2)
    df_portfolio_progression = df_portfolio_progression.round(2)
    simulation_results_html = df_portfolio_progression.to_html(index=False, justify="center", float_format="{:.2f}".format, border=1, escape=False) if not df_portfolio_progression.empty else ""

    for recipient in recipient_list:
        # メールの作成
        msg = MIMEMultipart("related")
        msg["From"] = GMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = f"日本v3 ETF上位(TOP50件)） アンサンブル推奨 ({current_date}) {int(current_portfolio)}円 {raito:.2f}倍"

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
            <p>本日の購入リストのフィクションおすすめ結果をお送りいたします。</p>
            <p>現在の投資額: {int(current_portfolio):,} 円</p>
            <p>初期投資額: {int(initial_investment):,} 円</p>
            <p>レバレッジ: {leverage} 倍</p>
            <p>取引期間: {deal_term} 営業日</p>
            <p>総合勝率: {win_rate :.2f} %</p>

            <h3>上位15件の推奨銘柄:</h3>
            {top5_stocks_html}

            <h3>曜日ごとの勝率:</h3>
            {weekday_table_html}

            <h3>銘柄ごとの勝率:</h3>
            {ticker_table_html}

            <h3>上位5社の株価チャート:</h3>
        """


        body_html += f"""
            <h3>シミュレーション結果:</h3>
            {simulation_results_html}
            <p>詳細につきましては、添付ファイルをご確認ください。</p>
            <p>ご不明な点がございましたら、お気軽にお問い合わせください。</p>
        </body>
        </html>
        """

        msg.attach(MIMEText(body_html, "html"))



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









