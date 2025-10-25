import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- 修正箇所 ---
# 各要素を「ティッカー  # 銘柄名」という形式の一つの文字列に修正しました。
futures_tickers = {
    "主要な仮想通貨": [
        "BTC-USD  # Bitcoin",
        "ETH-USD  # Ethereum",
        "SOL-USD  # Solana",
        "XRP-USD  # XRP",
        "ADA-USD  # Cardano",
        "AVAX-USD # Avalanche",
        "DOGE-USD # Dogecoin",
        "DOT-USD  # Polkadot",
        "LINK-USD # Chainlink",
        "MATIC-USD# Polygon",
    ],
    "レイヤー1・レイヤー2": [
        "NEAR-USD # NEAR Protocol",
        "ATOM-USD # Cosmos",
        "OP-USD   # Optimism",
        "ARB-USD  # Arbitrum",
        "SUI-USD  # Sui",
        "SEI-USD  # Sei",
    ],
    "DeFi (分散型金融)": [
        "UNI-USD  # Uniswap",
        "AAVE-USD # Aave",
        "LDO-USD  # Lido DAO",
        "MKR-USD  # Maker",
        "SNX-USD  # Synthetix",
        "COMP-USD # Compound",
    ],
    "ミームコイン": [
        "SHIB-USD # Shiba Inu",
        "PEPE-USD # Pepe",
        "WIF-USD  # dogwifhat",
        "BONK-USD # Bonk",
    ],
    "NFT・メタバース・ゲーム": [
        "IMX-USD  # Immutable",
        "SAND-USD # The Sandbox",
        "MANA-USD # Decentraland",
        "AXS-USD  # Axie Infinity",
        "GALA-USD # Gala",
        "ENJ-USD  # Enjin Coin",
    ],
    "ステーブルコイン": [
        "USDT-USD # Tether",
        "USDC-USD # USD Coin",
        "DAI-USD  # Dai",
    ],
    "その他": [
        "BCH-USD  # Bitcoin Cash",
        "LTC-USD  # Litecoin",
        "ETC-USD  # Ethereum Classic",
        "XLM-USD  # Stellar",
        "FIL-USD  # Filecoin",
        "GRT-USD  # The Graph",
    ]
}

# --- ここからが実行コード ---

# 保存先フォルダの指定（存在しなければ作成）
save_folder = "./分析"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 1. すべてのティッカーを一つのリストにまとめる
all_tickers = []
ticker_to_name = {}
for category, tickers in futures_tickers.items():
    for ticker_with_name in tickers:
        # "  # " (スペース2つ、#、スペース1つ)で分割
        parts = ticker_with_name.split("  # ")
        if len(parts) == 2:
            ticker = parts[0].strip()
            name = parts[1].strip()
            all_tickers.append(ticker)
            ticker_to_name[ticker] = name

# 重複を除外
ticker_symbols = sorted(list(set(all_tickers)))

# --- 追加：ティッカーリストが空でないかチェック ---
if not ticker_symbols:
    print("エラー: 処理対象のティッカーが見つかりませんでした。futures_tickersのデータ形式を確認してください。")
    exit() # プログラムを終了

print(f"合計 {len(ticker_symbols)} 銘柄の株価データをダウンロードします。処理には数分かかる場合があります...")

# 2. yfinanceで株価データを一括ダウンロード
try:
    data = yf.download(
        ticker_symbols,
        period="10y",
        progress=True,
        auto_adjust=False,
        group_by='ticker'
    )
    print("データのダウンロードが完了しました。")

    # 3. 各銘柄の上昇率を計算
    results = []
    
    for ticker in ticker_symbols:
        try:
            ticker_data = data[ticker] if len(ticker_symbols) > 1 else data
            
            adj_close = ticker_data['Adj Close'].dropna()

            if len(adj_close) > 1:
                start_price = adj_close.iloc[0]
                end_price = adj_close.iloc[-1]
                
                start_date = adj_close.index[0].strftime('%Y-%m-%d')
                end_date = adj_close.index[-1].strftime('%Y-%m-%d')

                if start_price > 0:
                    growth_rate = (end_price - start_price) / start_price
                    results.append({
                        'ティッカー': ticker,
                        '銘柄名': ticker_to_name.get(ticker, 'N/A'),
                        '上昇率 (%)': growth_rate * 100,
                        'カテゴリ': next((cat for cat, tkrs in futures_tickers.items() if any(t.startswith(ticker) for t in tkrs)), 'N/A'),
                        '開始日': start_date,
                        '終了日': end_date,
                        '開始価格': start_price,
                        '終了価格': end_price
                    })
        except (KeyError, IndexError) as e:
            print(f"ティッカー {ticker} のデータ処理中にエラーが発生しました: {e} スキップします。")

    # 4. 上昇率でソートし、上位30件を取得
    if results:
        results_df = pd.DataFrame(results)
        top_30_df = results_df.sort_values(by='上昇率 (%)', ascending=False).head(50).reset_index(drop=True)

        # 5. 結果を表示
        print("\n--- 過去10年間（または上場以来）の上昇率トップ30 ETF ---")
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.width', 120)
        print(top_30_df[['ティッカー', '銘柄名', '上昇率 (%)', 'カテゴリ']])
        
        # 6. 結果をCSVファイルに保存
        file_path = os.path.join(save_folder, "top_30_crypto_by_growth.csv")
        top_30_to_save = top_30_df.copy()
        top_30_to_save['上昇率 (%)'] = top_30_to_save['上昇率 (%)'].round(2)
        top_30_to_save['開始価格'] = top_30_to_save['開始価格'].round(2)
        top_30_to_save['終了価格'] = top_30_to_save['終了価格'].round(2)
        
        top_30_to_save.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"\n分析結果を {file_path} に保存しました。")

    else:
        print("上昇率を計算できる銘柄がありませんでした。")

except Exception as e:
    print(f"データダウンロード中に予期せぬエラーが発生しました: {e}")
    print("ネットワーク接続を確認するか、しばらくしてから再試行してください。")


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


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
# ★ NGBoost 変更点 1: NGBoost関連ライブラリをインポート
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from sklearn.tree import DecisionTreeRegressor
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

    top_ranked = pd.DataFrame(all_data).sort_values(by="スコア", ascending=False).head(15)

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
# 3. モデルの準備（クラスベース設計による最終修正版）
# =================================================================================
print("\nモデルの準備（必要に応じて学習・再学習）を開始します...")
OUTPUT_DIR = "saved_models_ensemble_1day"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5モデル分のモデル辞書を準備
lgb_models_dict, xgb_models_dict, cat_models_dict, tabnet_models_dict, ngb_models_dict, scalers_dict = {}, {}, {}, {}, {}, {}
lookback_days = 25
deal_term = 1
ONE_MONTH_AGO = datetime.now() - timedelta(days=30)
TRAIN_RATIO = 0.9

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 修正点: 状態管理のためのヘルパークラスを定義 ★
# これにより、nonlocalの問題を完全に回避します。
class ModelUpdater:
    def __init__(self, ticker):
        self.ticker = ticker
        self.X_train_cache = None
        self.y_train_cache = None
        self.X_train_scaled_cache = None
        self.is_data_prepared = False
        
        # 各モデルのパスをインスタンス変数として保持
        self.paths = {
            'lgb': os.path.join(OUTPUT_DIR, f"{ticker}_lgb_model.pkl"),
            'xgb': os.path.join(OUTPUT_DIR, f"{ticker}_xgb_model.pkl"),
            'cat': os.path.join(OUTPUT_DIR, f"{ticker}_cat_model.pkl"),
            'ngb': os.path.join(OUTPUT_DIR, f"{ticker}_ngb_model.pkl"),
            'tabnet': os.path.join(OUTPUT_DIR, f"{ticker}_tabnet_model.zip"),
            'scaler': os.path.join(OUTPUT_DIR, f"{ticker}_scaler.pkl")
        }

    def check_if_update_needed(self, model_key):
        path = self.paths[model_key]
        if not os.path.exists(path): return True
        if datetime.fromtimestamp(os.path.getmtime(path)) < ONE_MONTH_AGO: return True
        return False

    def prepare_data_once(self):
        if self.is_data_prepared:
            return True
        
        print(f"  > (初回実行) {self.ticker} の訓練データの特徴量を生成しています...")
        df_price_full = ticker_data_dict.get(self.ticker)
        if df_price_full is None or df_price_full.empty or len(df_price_full) < 200:
            print(f"  > データ不足のためスキップ。")
            return False
        
        split_index = int(len(df_price_full) * TRAIN_RATIO)
        df_price_train = df_price_full.iloc[:split_index]
        
        X_train, y_train, _, _ = create_combined_features(df_price_train, lookback_days=lookback_days, deal_term=deal_term)
        if len(X_train) < 50:
            print(f"  > 訓練サンプル数不足({len(X_train)}件)のためスキップ。")
            return False

        self.X_train_cache = X_train.astype(np.float32)
        self.y_train_cache = y_train.astype(np.int64)
        self.is_data_prepared = True
        return True

    def run_updates(self):
        # --- スケーラーの更新 ---
        if self.check_if_update_needed('scaler'):
            print(f"  > スケーラーを更新します...")
            if not self.prepare_data_once(): return False
            scaler = StandardScaler().fit(self.X_train_cache)
            joblib.dump(scaler, self.paths['scaler'])
            print(f"    - 新しいスケーラーを保存しました。")
        
        # --- 各モデルの更新 ---
        # スケーリング済みデータが必要になった場合に一度だけ作成
        def get_scaled_data():
            if self.X_train_scaled_cache is None:
                scaler = joblib.load(self.paths['scaler'])
                self.X_train_scaled_cache = scaler.transform(self.X_train_cache)
            return self.X_train_scaled_cache

        if self.check_if_update_needed('lgb'):
            print(f"  > LightGBMを更新します...")
            if not self.prepare_data_once(): return False
            lgb_model = lgb.LGBMClassifier(random_state=42).fit(get_scaled_data(), self.y_train_cache)
            joblib.dump(lgb_model, self.paths['lgb'])

        if self.check_if_update_needed('xgb'):
            print(f"  > XGBoostを更新します...")
            if not self.prepare_data_once(): return False
            xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(get_scaled_data(), self.y_train_cache)
            joblib.dump(xgb_model, self.paths['xgb'])

        if self.check_if_update_needed('cat'):
            print(f"  > CatBoostを更新します...")
            if not self.prepare_data_once(): return False
            cat_model = CatBoostClassifier(random_state=42, verbose=0, iterations=500).fit(get_scaled_data(), self.y_train_cache)
            joblib.dump(cat_model, self.paths['cat'])
            
        if self.check_if_update_needed('ngb'):
            print(f"  > NGBoostを更新します...")
            if not self.prepare_data_once(): return False
            base_learner = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)
            ngb_model = NGBClassifier(Dist=Bernoulli, Base=base_learner, n_estimators=500, learning_rate=0.05, verbose=False, random_state=42)
            ngb_model.fit(get_scaled_data(), self.y_train_cache)
            joblib.dump(ngb_model, self.paths['ngb'])

        if self.check_if_update_needed('tabnet'):
            print(f"  > TabNetを更新します...")
            if not self.prepare_data_once(): return False
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            tabnet_model = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                                            scheduler_params={"step_size":10, "gamma":0.9}, scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                            mask_type='sparsemax', device_name=device, verbose=0)
            tabnet_model.fit(X_train=get_scaled_data(), y_train=self.y_train_cache, eval_set=[(get_scaled_data(), self.y_train_cache)],
                             patience=50, max_epochs=1000, batch_size=1024)
            tabnet_model.save_model(self.paths['tabnet'].replace(".zip", ""))

        return True

    def load_all_models(self):
        lgb_models_dict[self.ticker] = joblib.load(self.paths['lgb'])
        xgb_models_dict[self.ticker] = joblib.load(self.paths['xgb'])
        cat_models_dict[self.ticker] = joblib.load(self.paths['cat'])
        ngb_models_dict[self.ticker] = joblib.load(self.paths['ngb'])
        tabnet_model = TabNetClassifier(); tabnet_model.load_model(self.paths['tabnet'])
        tabnet_models_dict[self.ticker] = tabnet_model
        scalers_dict[self.ticker] = joblib.load(self.paths['scaler'])
        print(f"  > {self.ticker}: 全モデルのロード完了。")

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# メインの for ループ
for ticker in ticker_symbols:
    print(f"\n--- {ticker} のモデルを処理しています... ---")
    try:
        updater = ModelUpdater(ticker) # tickerごとにインスタンスを作成
        if updater.run_updates():      # 更新処理を実行
            updater.load_all_models()  # 最終的に全てのモデルをロード
    except Exception as e:
        print(f"--- {ticker} の処理中に致命的なエラーが発生: {e} ---")
        continue

print("\n全ティッカーのモデル準備が完了しました。")
print(f"ロード済みモデル数: LightGBM({len(lgb_models_dict)}), XGBoost({len(xgb_models_dict)}), CatBoost({len(cat_models_dict)}), TabNet({len(tabnet_models_dict)}), NGBoost({len(ngb_models_dict)})")

# ==============================================================================
# 5. シミュレーションの実行 (アンサンブル予測)
# ==============================================================================
print("\nアンサンブルモデルによるシミュレーションを開始します...")
simulation_results = []
initial_investment = 2000000; leverage = 1; stop_loss_threshold = 0.3

for ticker in ticker_symbols:
    # ★ NGBoost 変更点 9: NGBoostのモデルも存在するかチェック
    data_exists = ticker in ticker_data_dict and not ticker_data_dict[ticker].empty
    models_exist = all(ticker in d for d in [lgb_models_dict, xgb_models_dict, cat_models_dict, tabnet_models_dict, ngb_models_dict, scalers_dict])

    if not (data_exists and models_exist):
        continue

    df_price = ticker_data_dict[ticker]
    # ★ NGBoost 変更点 10: ngb_modelも辞書から取り出す
    lgb_model, xgb_model, cat_model, tabnet_model, ngb_model = lgb_models_dict[ticker], xgb_models_dict[ticker], cat_models_dict[ticker], tabnet_models_dict[ticker], ngb_models_dict[ticker]
    scaler = scalers_dict[ticker]
    
    X_sim, _, _, indices_sim = create_combined_features(df_price, lookback_days=lookback_days, deal_term=deal_term)
    if len(X_sim) == 0: continue
    
    X_sim_scaled = scaler.transform(X_sim)
    
    # ★ NGBoost 変更点 11: NGBoostの予測確率を加えて5モデルの平均を取る
    pred_probs_lgb = lgb_model.predict_proba(X_sim_scaled)[:, 1]
    pred_probs_xgb = xgb_model.predict_proba(X_sim_scaled)[:, 1]
    pred_probs_cat = cat_model.predict_proba(X_sim_scaled)[:, 1]
    pred_probs_tabnet = tabnet_model.predict_proba(X_sim_scaled.astype(np.float32))[:, 1]
    pred_probs_ngb = ngb_model.predict_proba(X_sim_scaled)[:, 1]

    pred_probs = (pred_probs_lgb + pred_probs_xgb + pred_probs_cat + pred_probs_tabnet + pred_probs_ngb) / 5.0
    
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
simulation_file_name = f"./分析/simulation_results_ensemble_5models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
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
    trade_log_file_name = f"./分析/trade_log_max3_ensemble_5models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"; df_portfolio_progression.to_excel(trade_log_file_name, index=False)
    
    df_simulation["Date"] = pd.to_datetime(df_simulation["Simulation Date"]); df_simulation["Weekday"] = df_simulation["Date"].dt.dayofweek
    weekday_stats = df_simulation.groupby("Weekday").apply(lambda x: pd.Series({"勝利数": (x["Profit/Loss (JPY)"] > 0).sum(), "取引数": len(x), "勝率": (x["Profit/Loss (JPY)"] > 0).mean()})).reset_index()
    weekday_stats["曜日"] = weekday_stats["Weekday"].map({0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"})
    weekday_winrates_output_file = f"./分析/weekday_winrates_ensemble_5models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"; weekday_stats[["曜日", "勝利数", "取引数", "勝率"]].to_excel(weekday_winrates_output_file, index=False)
    
    ticker_stats = df_simulation.groupby("Ticker").apply(lambda x: pd.Series({"勝利数": (x["Profit/Loss (JPY)"] > 0).sum(), "取引数": len(x), "勝率": (x["Profit/Loss (JPY)"] > 0).mean()})).reset_index()
    ticker_stats["Company Name"] = ticker_stats["Ticker"].apply(get_company_name)
    ticker_winrates_output_file = f"./分析/ticker_winrates_ensemble_5models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"; ticker_stats[["Ticker", "Company Name", "勝利数", "取引数", "勝率"]].to_excel(ticker_winrates_output_file, index=False)


# ==============================================================================
# ▼▼▼ 7. 今日の予測と推奨銘柄のリストアップ（修正済みのセクション） ▼▼▼
# ==============================================================================
def get_daily_ohlc_from_hourly(ticker, period="10y"):
    df_hourly = yf.download(ticker, period=period, interval="1h", progress=False)
    if df_hourly.empty:
        print(f"  {ticker}: 1時間足データ取得できず。")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"], dtype=float)
    print("=" * 50); print(f"[{ticker}] リサンプリング前（直近 25 本の 1 時間足）"); print(df_hourly.tail(25)); print("=" * 50)
    df_hourly = df_hourly.sort_index()
    daily_open = df_hourly['Open'].resample('D').first(); daily_high = df_hourly['High'].resample('D').max()
    daily_low = df_hourly['Low'].resample('D').min(); daily_close = df_hourly['Close'].resample('D').last()
    df_daily = pd.concat([daily_open, daily_high, daily_low, daily_close], axis=1, keys=['Open', 'High', 'Low', 'Close']).dropna()
    print(f"[{ticker}] リサンプリング後（日足 OHLC）"); print(df_daily.tail(1005)); print("=" * 50)
    return df_daily

def get_market_threshold_time():
    market_tz = ZoneInfo("America/New_York"); now_market = datetime.now(market_tz)
    return dtime(22, 31, 0) if now_market.dst() != timedelta(0) else dtime(23, 31, 0)

def create_features_for_today_combined(ticker_symbols, lookback_days=25, deal_term=5):
    X_rows, tickers_ok = [], []
    for ticker in ticker_symbols:
        print(f"[INFO] Processing {ticker}...")
        df_daily = get_daily_ohlc_from_hourly(ticker, period="2y")
        if df_daily.empty: continue
        if isinstance(df_daily.columns, pd.MultiIndex) and 'Ticker' in df_daily.columns.names:
            df_daily.columns = df_daily.columns.droplevel('Ticker')
        df_extra = prepare_features_2(df_daily, num_1=59, num_2=10, conversion_period=2, base_period=7, span_b_period=45, displacement=22, calc_bars=60, bars_to_render=10, harmonic_period=10)
        df_daily = df_daily.dropna()
        if len(df_daily) < lookback_days: continue
        window = df_daily['Open'].iloc[-lookback_days:]; returns = window.pct_change().dropna().values
        if len(returns) != lookback_days - 1: continue
        n_extra = df_extra.shape[1]
        last_dt = df_daily.index[-1]
        extra = df_extra.loc[last_dt].values if last_dt in df_extra.index else np.zeros(n_extra, dtype=float)
        combined = np.hstack([returns, extra])
        X_rows.append(combined); tickers_ok.append(ticker)
    if X_rows: X_today = np.vstack(X_rows)
    else: X_today = np.empty((0, lookback_days - 1 + n_extra if 'n_extra' in locals() else lookback_days - 1))
    return X_today, tickers_ok

# --- 今日の特徴量作成 ---
X_today, tickers_today = create_features_for_today_combined(ticker_symbols, lookback_days=25, deal_term=5)
print(f"\n最終的な X_today.shape = {X_today.shape}"); print(f"銘柄数 = {len(tickers_today)}")

# --- 各ティッカーのモデルを用いて本日の予測確率を算出 ---
predictions = []
if len(X_today) > 0:
    for ticker, feature_vector in zip(tickers_today, X_today):
        # ★ NGBoost 変更点 12: NGBoostのモデル辞書もチェック
        if all(ticker in d for d in [lgb_models_dict, xgb_models_dict, cat_models_dict, tabnet_models_dict, ngb_models_dict, scalers_dict]):
            scaler = scalers_dict[ticker]
            # ★ NGBoost 変更点 13: ngb_modelも辞書から取り出す
            lgb_model, xgb_model, cat_model, tabnet_model, ngb_model = lgb_models_dict[ticker], xgb_models_dict[ticker], cat_models_dict[ticker], tabnet_models_dict[ticker], ngb_models_dict[ticker]
            
            x_today_vec = feature_vector.reshape(1, -1)
            x_scaled = scaler.transform(x_today_vec)
            
            prob_lgb = lgb_model.predict_proba(x_scaled)[0, 1]
            prob_xgb = xgb_model.predict_proba(x_scaled)[0, 1]
            prob_cat = cat_model.predict_proba(x_scaled)[0, 1]
            prob_tabnet = tabnet_model.predict_proba(x_scaled.astype(np.float32))[0, 1]
            # ★ NGBoost 変更点 14: NGBoostで予測
            prob_ngb = ngb_model.predict_proba(x_scaled)[0, 1]
            
            # ★ NGBoost 変更点 15: 5モデルの平均を計算
            prob = (prob_lgb + prob_xgb + prob_cat + prob_tabnet + prob_ngb) / 5.0
            predictions.append({"Ticker": ticker, "Probability": prob})

# (以降の推奨銘柄リストアップ、メール送信部分は変更なし)
today_data = pd.DataFrame(predictions).sort_values("Probability", ascending=False)
recommendation_data = today_data.sort_values(by="Probability", ascending=False)
print("本日の各ティッカー別予測結果:"); print(recommendation_data)
purchase_recommendations_top3, purchase_recommendations_top5, purchase_recommendations_top6 = [], [], []
end_date = datetime.now(); exit_date = end_date + BusinessDay(deal_term)
print("Exit Date (deal_term 営業日後):", exit_date)
for top_n, purchase_recommendations in [(3, purchase_recommendations_top3), (15, purchase_recommendations_top5), (6, purchase_recommendations_top6)]:
    top_stocks_rec = recommendation_data.head(top_n)
    for idx, row in top_stocks_rec.iterrows():
        ticker_ = row["Ticker"]; company_name = get_company_name(ticker_)
        try:
            start_price = float(data[ticker_]["Open"].iloc[-1]); trade_amount = initial_investment / top_n
            position_size = trade_amount; stop_loss_price = (1 - stop_loss_threshold) * start_price
            stop_loss_amount = stop_loss_threshold * start_price; number_of_shares = position_size // start_price
            if number_of_shares == 0: continue
            actual_investment = number_of_shares * start_price
            purchase_recommendations.append({"Entry Date": end_date.strftime('%Y-%m-%d'), "Exit Date": exit_date.strftime('%Y-%m-%d'), "Term (Business Days)": deal_term, "Ticker": ticker_, "Company Name": company_name, "Current Price": start_price, "Stop Loss Amount": stop_loss_amount * int(number_of_shares), "Stop Loss Price": stop_loss_price, "Shares Bought": int(number_of_shares), "Investment per Stock (JPY)": round(actual_investment, 2), "Predicted Probability (%)": round(row["Probability"] * 100, 2)})
        except Exception as e: print(f"{ticker_}: エラーが発生しました。", e); continue
columns_order = ["Entry Date", "Exit Date", "Term (Business Days)", "Ticker", "Company Name", "Current Price", "Shares Bought", "Stop Loss Amount", "Stop Loss Price", "Investment per Stock (JPY)", "Predicted Probability (%)"]
purchase_df_top3 = pd.DataFrame(purchase_recommendations_top3)[columns_order] if purchase_recommendations_top3 else pd.DataFrame(columns=columns_order)
purchase_df_top5 = pd.DataFrame(purchase_recommendations_top5)[columns_order] if purchase_recommendations_top5 else pd.DataFrame(columns=columns_order)
purchase_df_top6 = pd.DataFrame(purchase_recommendations_top6)[columns_order] if purchase_recommendations_top6 else pd.DataFrame(columns=columns_order)
print("\n上位3件の購入推奨銘柄:"); print(purchase_df_top3); print("\n上位15件の購入推奨銘柄:"); print(purchase_df_top5); print("\n上位6件の購入推奨銘柄:"); print(purchase_df_top6)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name_top3 = f"./分析/5models_top3_{timestamp}.xlsx"; file_name_top5 = f"./分析/5models_top5_{timestamp}.xlsx"; file_name_top6 = f"./分析/5models_top6_{timestamp}.xlsx"
purchase_df_top3.to_excel(file_name_top3, index=False); purchase_df_top5.to_excel(file_name_top5, index=False); purchase_df_top6.to_excel(file_name_top6, index=False)
print("\n購入リストが以下のファイルに保存されました。"); print(f" - {file_name_top3}"); print(f" - {file_name_top5}"); print(f" - {file_name_top6}")


