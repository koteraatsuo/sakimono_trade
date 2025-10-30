import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- 修正箇所 ---
# 各要素を「ティッカー  # 銘柄名」という形式の一つの文字列に修正しました。
futures_tickers = {
    "メジャー通貨ペア": [
        "EURUSD=X  # ユーロ／ 米ドル",
        "JPY=X      # 米ドル／ 円",
        "GBPUSD=X  # イギリスポンド／ 米ドル",
        "CHF=X      # 米ドル／ スイスフラン",
        "AUDUSD=X  # オーストラリアドル／ 米ドル",
        "CAD=X      # 米ドル／ カナダドル",
        "NZDUSD=X  # ニュージーランドドル／ 米ドル",
    ],
    "クロス円": [
        "EURJPY=X  # ユーロ／ 円",
        "GBPJPY=X  # イギリスポンド／ 円",
        "AUDJPY=X  # オーストラリアドル／ 円",
        "NZDJPY=X  # ニュージーランドドル／ 円",
        "CADJPY=X  # カナダドル／ 円",
        "CHFJPY=X  # スイスフラン／ 円",
        "CNHJPY=X  # 人民元／ 日本円",
    ],
    "その他のクロス通貨": [
        "EURGBP=X  # ユーロ／ イギリスポンド",
        "EURAUD=X  # ユーロ／ オーストラリアドル",
        "EURNZD=X  # ユーロ／ ニュージーランドドル",
        "EURCAD=X  # ユーロ／ カナダドル",
        "EURCHF=X  # ユーロ／ スイスフラン",
        "GBPAUD=X  # イギリスポンド／ オーストラリアドル",
        "GBPNZD=X  # イギリスポンド／ ニュージーランドドル",
        "GBPCAD=X  # イギリスポンド／ カナダドル",
        "GBPCHF=X  # イギリスポンド／ スイスフラン",
        "AUDNZD=X  # オーストラリアドル／ ニュージーランドドル",
        "AUDCAD=X  # オーストラリアドル／ カナダドル",
        "AUDCHF=X  # オーストラリアドル／ スイスフラン",
        "NZDCAD=X  # ニュージーランドドル／ カナダドル",
        "NZDCHF=X  # ニュージーランドドル／ スイスフラン",
        "CADCHF=X  # カナダドル／ スイスフラン",
    ],
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
        file_path = os.path.join(save_folder, "top_30_fx_gaitame_by_growth.csv")
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

# =============================================================================
# ▼▼▼ 1. トップレベル：すべてのインポート文 ▼▼▼
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
import datetime
import time
import sys

# =============================================================================
# ▼▼▼ 2. トップレベル：すべてのグローバル変数・定数 ▼▼▼
# =============================================================================
OUTPUT_DIR = "saved_models_ensemble_1day"
COMPANY_LIST_FILE = "data_j.xls"
_COMPANY_NAME_MAP = None

# =============================================================================
# ▼▼▼ 3. トップレベル：すべての関数・クラス定義 ▼▼▼
# =============================================================================

# -----------------------------------------------------------------------------
# 特徴量生成などに関する関数群
# -----------------------------------------------------------------------------
def process_ha_and_streaks(df_rates):
    if not isinstance(df_rates, pd.DataFrame):
        df_rates = pd.DataFrame(df_rates)
    
    ha_open_values = []
    ha_close_values = []
    
    for i in range(len(df_rates)):
        if i == 0:
            ha_open = (df_rates['Open'].iloc[i] + df_rates['close'].iloc[i]) / 2
            ha_close = (df_rates['Open'].iloc[i] + df_rates['high'].iloc[i] +
                        df_rates['low'].iloc[i] + df_rates['close'].iloc[i]) / 4
        else:
            ha_open = (ha_open_values[i-1] + ha_close_values[i-1]) / 2
            ha_close = (df_rates['open'].iloc[i] + df_rates['high'].iloc[i] +
                        df_rates['low'].iloc[i] + df_rates['close'].iloc[i]) / 4
        ha_open_values.append(ha_open)
        ha_close_values.append(ha_close)
    
    df_rates['ha_open'] = ha_open_values
    df_rates['ha_close'] = ha_close_values
    df_rates['ha_high'] = df_rates[['ha_open', 'ha_close', 'high']].max(axis=1)
    df_rates['ha_low'] = df_rates[['ha_open', 'ha_close', 'low']].min(axis=1)
    df_rates['ha_color'] = '陽線'
    df_rates.loc[df_rates['ha_open'] > df_rates['ha_close'], 'ha_color'] = '陰線'
    return df_rates

# ② RSI の計算関数
def calculate_rsi(df, period=14):
    df_copy = df.copy()
    df_copy['delta'] = df_copy['Open'].diff()
    df_copy['gain'] = np.where(df_copy['delta'] > 0, df_copy['delta'], 0)
    df_copy['loss'] = np.where(df_copy['delta'] < 0, -df_copy['delta'], 0)
    df_copy['avg_gain'] = df_copy['gain'].rolling(window=period, min_periods=1).mean()
    df_copy['avg_loss'] = df_copy['loss'].rolling(window=period, min_periods=1).mean()
    df_copy['RS'] = df_copy['avg_gain'] / df_copy['avg_loss']
    df_copy['RSI'] = 100 - (100 / (1 + df_copy['RS']))
    return df_copy[['RSI']]

# ③ 一目均衡表の計算関数（パラメータを外部から渡せるように変更）
def calculate_ichimoku(df, conversion_period=3, base_period=27, span_b_period=52, displacement=26):
    high_prices = df['High']
    low_prices = df['Low']
    close_prices = df['Open']

    # 転換線 (Tenkan-sen)
    period9_high = high_prices.rolling(window=int(conversion_period)).max()
    period9_low = low_prices.rolling(window=int(conversion_period)).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2

    # 基準線 (Kijun-sen)
    period26_high = high_prices.rolling(window=int(base_period)).max()
    period26_low = low_prices.rolling(window=int(base_period)).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    # 転換線・基準線の傾き
    df['tenkan_sen_slope'] = df['tenkan_sen'].diff()
    df['kijun_sen_slope'] = df['kijun_sen'].diff()

    # 先行スパンA
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(int(displacement))

    # 先行スパンB
    period52_high = high_prices.rolling(window=int(span_b_period)).max()
    period52_low = low_prices.rolling(window=int(span_b_period)).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(int(displacement))

    # 遅行スパン
    # df['chikou_span'] = close_prices.shift(-int(displacement))
    return df


def rolling_fft_features(df, target_col='Open', window=60, freq_list=[1,2,3,5,10]):
    """
    df内の target_col に対して、過去 window バーを用いたフーリエ変換を行い、
    freq_list で指定した周波数成分の振幅・位相を特徴量として付与する。
    
    【出力】
    - df に以下の列を追加
      fft_amp_{freq}, fft_phase_{freq}  (freq_list内の周波数ごと)
    """
    df = df.copy()
    
    # 結果を格納するリスト（計算途中でappendし、最後にDataFrameに結合）
    all_amp = {f:[] for f in freq_list}
    all_phase = {f:[] for f in freq_list}
    
    # ローリングでウィンドウを移動しながらFFT
    for i in range(len(df)):
        # window分たまってない部分はNaNにする
        if i < window:
            for f in freq_list:
                all_amp[f].append(np.nan)
                all_phase[f].append(np.nan)
            continue
        
        # 過去windowバー分のデータ
        data_window = df[target_col].iloc[i-window:i].values
        fft_res = np.fft.fft(data_window)
        
        # 各周波数成分を取得
        # freq_listの値fは、配列インデックスfreq_idxとして利用
        # ※基本的には freq_idx = f だが、もし「周期(サイクル)」指定なら計算式が異なるので注意
        for f in freq_list:
            freq_idx = f  # 例：インデックスをそのままfとする
            if freq_idx < len(fft_res):
                amplitude = np.abs(fft_res[freq_idx])
                phase = np.angle(fft_res[freq_idx])
            else:
                amplitude = np.nan
                phase = np.nan
            
            all_amp[f].append(amplitude)
            all_phase[f].append(phase)
    
    # DataFrameに列を追加
    for f in freq_list:
        df[f'fft_amp_{f}'] = all_amp[f]
        df[f'fft_phase_{f}'] = all_phase[f]

    for f in [1, 2, 3, 5, 10]:
        df[f'fft_amp_{f}_slope'] = df[f'fft_amp_{f}'].diff()
        df[f'fft_phase_{f}_slope'] = df[f'fft_phase_{f}'].diff()
    
    return df


def hp_filter(series, lamb=50):
    """
    Hodrick-Prescottフィルタで series をトレンド成分とサイクル成分に分解。
    戻り値 (cycle, trend)
    """
    cycle, trend = sm.tsa.filters.hpfilter(series, lamb=lamb)
    return cycle, trend


import numpy as np
from scipy.optimize import minimize

def quinn_fernandes_extrapolation(data, calc_bars, harmonic_period, freq_tolerance, bars_to_render, n_harmonics):
    """
    【安定版】
    Quinn-Fernandes法を用いて周波数を推定し、データを外挿（補外）する。
    呼び出し元の引数に合わせて定義を修正し、かつ動的な周波数計算でIndexErrorを防止。
    """
    # dataがDataFrameやSeriesの可能性があるため、.valuesでNumpy配列に変換する
    if hasattr(data, 'values'):
        x = data.values.flatten()
    else:
        x = np.array(data).flatten()

    # --- IndexErrorを防ぐためのコアロジック ---
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])

    spectrum = np.fft.fft(x)
    # ★★★ 最重要ポイント ★★★
    # 入力データ`x`の長さ`n`と完全に一致する周波数配列を動的に生成する。
    # これにより、`spectrum`のインデックスと`freqs`のインデックスが常に1対1で対応する。
    freqs = np.fft.fftfreq(n)

    # 正の周波数領域（スペクトルの前半）で最もパワーが大きいインデックスを探す
    if n > 1:
        # 正の周波数領域はインデックス 1 から n//2 まで
        # `n//2 > 1` のチェックで、データが非常に短い場合のエラーを防ぐ
        search_area = spectrum[1:n//2] if n//2 > 1 else spectrum[1:]
        if len(search_area) > 0:
            idx = np.argmax(np.abs(search_area)) + 1
        else:
            idx = 0 # 探索エリアがない場合は0
    else:
        idx = 0
        
    # この時点で`idx`は必ず`freqs`の範囲内に収まるため、IndexErrorは発生しない
    w_est = 2.0 * np.pi * freqs[idx]

    # --- 以降の最適化処理（変更なし） ---
    def cost_function(w, x_vals):
        n_vals = len(x_vals)
        t = np.arange(n_vals)
        real_part = np.sum(x_vals * np.cos(w * t))
        imag_part = np.sum(x_vals * np.sin(w * t))
        return -(real_part**2 + imag_part**2)

    try:
        result = minimize(cost_function, w_est, args=(x,), method='Nelder-Mead', options={'xatol': 1e-8})
        w_opt = result.x[0]
    except: # 最適化が失敗した場合
        w_opt = w_est

    t = np.arange(n)
    cos_t = np.cos(w_opt * t)
    sin_t = np.sin(w_opt * t)
    
    A_cos = 2 * np.sum(x * cos_t) / n if n > 0 else 0
    A_sin = 2 * np.sum(x * sin_t) / n if n > 0 else 0
    A = np.sqrt(A_cos**2 + A_sin**2)
    phi_opt = np.arctan2(-A_sin, A_cos)

    reconstructed_past = A * np.cos(w_opt * t + phi_opt)
    
    # 呼び出し側の期待する返り値の長さを考慮
    if bars_to_render == 0:
        return reconstructed_past, np.arange(n)
    else:
        future_t = np.arange(n, n + bars_to_render)
        reconstructed_future = A * np.cos(w_opt * future_t + phi_opt)
        full_reconstruction = np.concatenate([reconstructed_past, reconstructed_future])
        return full_reconstruction, np.arange(len(full_reconstruction))


def rolling_quinn_features_no_future(df, calc_bars=59, harmonic_period=10, 
                                       freq_tolerance=0.01, n_harmonics=15):
    """
    各行ごとに、直前calc_bars件のデータから Quinn-Fernandes再構成を行い、
    再構成値全体（過去のみ、長さ = calc_bars）の各要素から
    当該行の最新closeを引いた上で10倍した値を算出します。
    
    特徴量として取り出すのは diff のうち、インデックス 1, 3, 7, 11, 17, 23 のみです。
    
    返却するDataFrameは、インデックスは計算対象の各行のインデックスとなり、
    新たな特徴量列は "qf_diff_1", "qf_diff_3", "qf_diff_7", ... となります。
    """
    results = []
    indices = []
    # 未来データは参照しないため、再構成値の長さはcalc_barsのみとなる
    L = calc_bars  
    selected_indices = [1, 3, 7, 11, 17, 23]
    
    # ループ対象: calc_bars から df の終端まで
    for i in range(calc_bars, len(df)):
        current_close = df['Open'].iloc[i].item()
        # 過去calc_bars件のデータを抽出
        window_data = df['Open'].iloc[i - calc_bars : i].values
        
        # 再構成値を計算（bars_to_renderを0にすることで未来は参照しない）
        reconst, _ = quinn_fernandes_extrapolation(
            data=window_data,
            calc_bars=calc_bars,
            harmonic_period=harmonic_period,
            freq_tolerance=freq_tolerance,
            bars_to_render=0,  # 未来のバーは使用しない
            n_harmonics=n_harmonics
        )
        # 再構成値各要素から現在のcloseを引いて10倍
        diff_vector = (reconst - current_close) * 10.0
        
        # selected_indices に含まれる位置のみ結果として保存（calc_bars内に収まるかチェック）
        feature_dict = {f"qf_diff_{j}": diff_vector[j] for j in selected_indices if j < L}
        results.append(feature_dict)
        indices.append(df.index[i])
     
    return pd.DataFrame(results, index=indices)



calc_bars_global = 41

def prepare_features_2(master_df, num_1=59, num_2=10, conversion_period=2, base_period=7, span_b_period=45, displacement=22,
                       calc_bars=60, bars_to_render=10, harmonic_period=10):
    """
    【最終修正版】
    特徴量を生成し、元の列を完全に削除して「追加の特徴量のみ」を返すように修正。
    これにより、join時の列名重複(overlap)エラーを根本的に解決する。
    """
    df = pd.DataFrame(master_df)
    
    # ★★★ 最重要修正点① ★★★
    # 特徴量計算を始める前に、元の列名をすべて保存しておく
    original_columns = df.columns.tolist()

    # もし 'close' 列が大文字なら小文字に統一
    if 'close' not in df.columns and 'Close' in df.columns:
        df.rename(columns={'Close': 'close'}, inplace=True)

    # 最低必要データ数チェック
    MIN_DATA_REQUIRED = calc_bars + bars_to_render + 20
    if len(df) < MIN_DATA_REQUIRED:
        raise ValueError(f"データが少なすぎます。最低でも{MIN_DATA_REQUIRED}本必要です。現在は{len(df)}本です。")
    
    # --- 既存の特徴量計算 ---
    df['price_diff_1'] = df['Open'].diff(1) * 10.00
    df = calculate_ichimoku(df, conversion_period, base_period, span_b_period, displacement)
    
    for diff in [1,2,4,8,12,15]:
        df[f'tenkan_diff{diff}'] = df['tenkan_sen'].diff(diff) * 10.00
        df[f'kijun_diff{diff}'] = df['kijun_sen'].diff(diff) * 10.00
        df[f'senkou_a_diff{diff}'] = df['senkou_span_a'].diff(diff) * 10.00
        df[f'senkou_b_diff{diff}'] = df['senkou_span_b'].diff(diff) * 10.00

    df = rolling_fft_features(df, target_col='Open', window=60, freq_list=[1,2,3,5,10])
    
    qf_df = rolling_quinn_features_no_future(
        df, calc_bars=int(60), harmonic_period=20
    )
    
    # 全ての特徴量が含まれたdfと、qf_dfをマージ
    df_merged = df.merge(qf_df, how='inner', left_index=True, right_index=True)
    
    # ★★★ 最重要修正点② ★★★
    # 保存しておいた元の列名をすべて削除する。
    # これにより、この関数は「新しく生成された特徴量」だけを返すことが保証される。
    df_extra_features = df_merged.drop(columns=original_columns, errors='ignore')
    
    # 不要な中間生成物を削除する（念のため）
    df_extra_features = df_extra_features.drop(columns=[
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b'
    ], errors='ignore')
    
    df_extra_features.dropna(inplace=True)

    if df_extra_features.empty or df_extra_features.shape[1] == 0:
        raise ValueError("特徴量が空です。パラメータやデータ量を確認してください。")
    
    return df_extra_features


def create_combined_features(df_price, lookback_days=25, deal_term=5):
    try:
        df_extra = prepare_features_2(df_price.copy())
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

def _load_company_name_map():
    global _COMPANY_NAME_MAP
    if _COMPANY_NAME_MAP is None:
        if not os.path.exists(COMPANY_LIST_FILE): _COMPANY_NAME_MAP = {}
        else:
            df = pd.read_excel(COMPANY_LIST_FILE, dtype=str)
            df["コード"] = df["コード"].str.strip().str.zfill(4)
            _COMPANY_NAME_MAP = dict(zip(df["コード"], df["銘柄名"].str.strip()))
    return _COMPANY_NAME_MAP

def get_company_name(ticker):
    return _load_company_name_map().get(ticker.replace(".T","").zfill(4), "不明")

# -----------------------------------------------------------------------------
# モデル更新・管理クラス
# -----------------------------------------------------------------------------
class ModelUpdater:
    def __init__(self, ticker, lookback_days, deal_term, train_ratio, one_month_ago):
        """
        コンストラクタ：各種設定を初期化し、モデルファイルのパスを定義する。
        """
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.deal_term = deal_term
        self.train_ratio = train_ratio
        self.one_month_ago = one_month_ago
        
        self.X_train_cache = None
        self.y_train_cache = None
        self.X_train_scaled_cache = None
        self.is_data_prepared = False
        
        # パス定義：TabNetは拡張子なしのベースパス名とする
        self.paths = {
            'lgb': os.path.join(OUTPUT_DIR, f"{self.ticker}_lgb_model.pkl"),
            'xgb': os.path.join(OUTPUT_DIR, f"{self.ticker}_xgb_model.pkl"),
            'cat': os.path.join(OUTPUT_DIR, f"{self.ticker}_cat_model.pkl"),
            'ngb': os.path.join(OUTPUT_DIR, f"{self.ticker}_ngb_model.pkl"),
            'tabnet': os.path.join(OUTPUT_DIR, f"{self.ticker}_tabnet_model"), # <- .zipなし
            'scaler': os.path.join(OUTPUT_DIR, f"{self.ticker}_scaler.pkl")
        }

    def check_if_update_needed(self, model_key):
        """
        指定されたモデルキーのファイルが存在し、かつ新しいかをチェックする。
        TabNetの場合は.zip拡張子を付けてチェックする。
        """
        path = self.paths[model_key]

        # TabNetの場合、チェックするパスに.zip拡張子を動的に付与する
        if model_key == 'tabnet':
            check_path = path + ".zip"
        else:
            check_path = path

        # `check_path` を使ってファイルの存在を確認する
        if not os.path.exists(check_path):
            return True
        
        # `check_path` を使ってファイルの更新日時を確認する
        return datetime.datetime.fromtimestamp(os.path.getmtime(check_path)) < self.one_month_ago

    def prepare_data_once(self, df_price_full):
        """
        訓練用の特徴量データを生成する。処理は一度だけ実行される。
        """
        if self.is_data_prepared:
            return True
        
        print(f"  > (初回実行) {self.ticker} の訓練データの特徴量を生成しています...")
        if df_price_full is None or df_price_full.empty or len(df_price_full) < 200:
            print(f"  > データ不足のためスキップ。")
            return False
            
        split_index = int(len(df_price_full) * self.train_ratio)
        df_price_train = df_price_full.iloc[:split_index]
        
        X_train, y_train, _, _ = create_combined_features(df_price_train, lookback_days=self.lookback_days, deal_term=self.deal_term)
        
        if len(X_train) < 50:
            print(f"  > 訓練サンプル数不足({len(X_train)}件)のためスキップ。")
            return False

        self.X_train_cache = X_train.astype(np.float32)
        self.y_train_cache = y_train.astype(np.int64)
        self.is_data_prepared = True
        return True

    def run_updates(self, df_price_full):
        """
        スケーラーと各モデルの更新要否を判断し、必要な学習処理を実行する。
        """
        # --- ステップ1: スケーラーの更新要否を判断 ---
        # スケーラー自体が古いか、あるいは他のいずれかのモデルが更新を必要とするかを確認
        should_update_scaler = self.check_if_update_needed('scaler')
        if not should_update_scaler:
            for model_key in ['lgb', 'xgb', 'cat', 'ngb', 'tabnet']:
                if self.check_if_update_needed(model_key):
                    should_update_scaler = True
                    print(f"  > {model_key}の更新が必要なため、スケーラーも更新対象とします。")
                    break
        
        # もし何の更新も不要なら、ここで処理を終了する
        if not should_update_scaler:
            print(f"  > {self.ticker}: 全てのモデルは最新です。更新をスキップします。")
            return True

        # --- ステップ2: データ準備とスケーラー更新 ---
        # スケーラー更新が必要な場合のみ、データ準備とスケーラーの再学習を行う
        if not self.prepare_data_once(df_price_full):
            return False  # データ準備失敗
        
        print(f"  > スケーラーを再学習・保存します...")
        scaler = StandardScaler().fit(self.X_train_cache)
        joblib.dump(scaler, self.paths['scaler'])
        
        # スケーリング済みデータのキャッシュをクリアし、再生成を促す
        self.X_train_scaled_cache = None
        
        # --- ステップ3: 各モデルの更新 ---
        _scaler_instance = None # ローカルキャッシュ
        def get_scaled_data():
            nonlocal _scaler_instance
            # キャッシュがあればそれを返す
            if self.X_train_scaled_cache is not None:
                return self.X_train_scaled_cache
            
            # データが準備されていなければエラー（通常は通らない）
            if not self.is_data_prepared:
                if not self.prepare_data_once(df_price_full): return None

            # スケーラーをロードしてtransformを実行
            if _scaler_instance is None:
                _scaler_instance = joblib.load(self.paths['scaler'])
            
            self.X_train_scaled_cache = _scaler_instance.transform(self.X_train_cache)
            return self.X_train_scaled_cache

        # 各モデルについて、更新が必要なものだけを学習する
        if self.check_if_update_needed('lgb'):
            print(f"  > LightGBMを更新します...");
            if get_scaled_data() is None: return False
            lgb_model = lgb.LGBMClassifier(random_state=42).fit(get_scaled_data(), self.y_train_cache)
            joblib.dump(lgb_model, self.paths['lgb'])
            
        if self.check_if_update_needed('xgb'):
            print(f"  > XGBoostを更新します...");
            if get_scaled_data() is None: return False
            xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(get_scaled_data(), self.y_train_cache)
            joblib.dump(xgb_model, self.paths['xgb'])

        if self.check_if_update_needed('cat'):
            print(f"  > CatBoostを更新します...");
            if get_scaled_data() is None: return False
            cat_model = CatBoostClassifier(random_state=42, verbose=0, iterations=500).fit(get_scaled_data(), self.y_train_cache)
            joblib.dump(cat_model, self.paths['cat'])
            
        if self.check_if_update_needed('ngb'):
            print(f"  > NGBoostを更新します...");
            if get_scaled_data() is None: return False
            base_learner = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)
            ngb_model = NGBClassifier(Dist=Bernoulli, Base=base_learner, n_estimators=500, learning_rate=0.05, verbose=False, random_state=42)
            ngb_model.fit(get_scaled_data(), self.y_train_cache)
            joblib.dump(ngb_model, self.paths['ngb'])

        if self.check_if_update_needed('tabnet'):
            print(f"  > TabNetを更新します...");
            if get_scaled_data() is None: return False
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            tabnet_model = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                                            scheduler_params={"step_size":10, "gamma":0.9}, scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                            mask_type='sparsemax', device_name=device, verbose=0)
            tabnet_model.fit(X_train=get_scaled_data(), y_train=self.y_train_cache, eval_set=[(get_scaled_data(), self.y_train_cache)],
                             patience=50, max_epochs=1000, batch_size=1024)
            # 保存時は拡張子なしのベースパス名を渡す
            tabnet_model.save_model(self.paths['tabnet'])

        return True

# -----------------------------------------------------------------------------
# 並列処理用のワーカー関数
# -----------------------------------------------------------------------------
def process_ticker_models(ticker, df_price_full, lookback_days, deal_term, train_ratio, one_month_ago):
    print(f"--- [Worker] {ticker} のモデルを処理しています... ---")
    try:
        updater = ModelUpdater(ticker, lookback_days, deal_term, train_ratio, one_month_ago)
        if updater.run_updates(df_price_full):
            return ticker, updater.paths
    except Exception:
        print(f"--- {ticker} の処理中に致命的なエラーが発生 ---")
        traceback.print_exc()
    return None, None

# def run_simulation_for_ticker(args):
#     ticker, df_price, lgb_model, xgb_model, cat_model, tabnet_model, ngb_model, scaler, lookback_days, deal_term = args
#     local_simulation_results = []
#     initial_investment, leverage, stop_loss_threshold = 2000000, 1, 0.3
#     try:
#         X_sim, _, _, indices_sim = create_combined_features(df_price, lookback_days=lookback_days, deal_term=deal_term)
#         if len(X_sim) == 0: return []
#         X_sim_scaled = scaler.transform(X_sim)
#         pred_probs_lgb = lgb_model.predict_proba(X_sim_scaled)[:, 1]
#         pred_probs_xgb = xgb_model.predict_proba(X_sim_scaled)[:, 1]
#         pred_probs_cat = cat_model.predict_proba(X_sim_scaled)[:, 1]
#         pred_probs_tabnet = tabnet_model.predict_proba(X_sim_scaled.astype(np.float32))[:, 1]
#         pred_probs_ngb = ngb_model.predict_proba(X_sim_scaled)[:, 1]
#         pred_probs = (pred_probs_lgb + pred_probs_xgb + pred_probs_cat + pred_probs_tabnet + pred_probs_ngb) / 5.0
#         sim_count = max(1, int(len(X_sim) * 0.1))
#         next_trade_available_idx = 0
#         for sample_idx in range(len(X_sim) - sim_count, len(X_sim)):
#             if sample_idx >= next_trade_available_idx:
#                 start_idx = indices_sim[sample_idx]
#                 end_idx = start_idx + deal_term
#                 if end_idx >= len(df_price) or df_price['Open'].iloc[start_idx] == 0: continue
#                 start_price = df_price['Open'].iloc[start_idx]
#                 exit_price = df_price['Open'].iloc[end_idx]
#                 stop_loss_price = start_price * (1 - stop_loss_threshold)
#                 for i in range(start_idx, end_idx):
#                     if df_price['Low'].iloc[i] <= stop_loss_price:
#                         exit_price = stop_loss_price; break
#                 number_of_shares = initial_investment // start_price
#                 if number_of_shares == 0: continue
#                 profit_loss = (exit_price - start_price) * number_of_shares * leverage
#                 local_simulation_results.append({
#                     "Ticker": ticker, "Company Name": get_company_name(ticker), "Simulation Date": df_price.index[start_idx].date(),
#                     "Start Price": start_price, "Stop Loss Price": stop_loss_price, "End Price": df_price['Open'].iloc[end_idx],
#                     "Exit Price": exit_price, "Price Difference": exit_price - start_price, "Profit/Loss (JPY)": round(profit_loss, 2),
#                     "Predicted Probability": round(pred_probs[sample_idx], 3)
#                 })
#                 next_trade_available_idx = sample_idx + deal_term
#         return local_simulation_results
#     except Exception:
#         print(f"シミュレーションエラー ({ticker}):"); traceback.print_exc(); return []



def run_simulation_for_ticker(args):
    """
    【並列処理用ワーカー関数】
    単一ティッカーのシミュレーションを実行する。
    """
    # 引数をアンパックする
    ticker, df_price, models, scaler, lookback_days, deal_term = args
    lgb_model, xgb_model, cat_model, tabnet_model, ngb_model = models

    local_simulation_results = []
    initial_investment = 100000
    leverage = 5
    stop_loss_threshold = 0.3

    try:
        # ▼▼▼【修正箇所】▼▼▼
        # 'Open'列にNaN（欠損値）が1つでも含まれていれば、このティッカーの処理をスキップ
        if df_price['Open'].isnull().any():
            # print(f"--- スキップ ({ticker}): 'Open'価格に欠損値(NaN)が含まれています ---") # ログが必要な場合はコメントを解除
            return []
        # ▲▲▲【修正完了】▲▲▲

        # 特徴量生成
        X_sim, _, _, indices_sim = create_combined_features(df_price, lookback_days=lookback_days, deal_term=deal_term)
        if len(X_sim) == 0:
            return [] # 結果がない場合は空リストを返す

        # スケーリングと予測
        X_sim_scaled = scaler.transform(X_sim)
        pred_probs_lgb = lgb_model.predict_proba(X_sim_scaled)[:, 1]
        pred_probs_xgb = xgb_model.predict_proba(X_sim_scaled)[:, 1]
        pred_probs_cat = cat_model.predict_proba(X_sim_scaled)[:, 1]
        pred_probs_tabnet = tabnet_model.predict_proba(X_sim_scaled.astype(np.float32))[:, 1]
        pred_probs_ngb = ngb_model.predict_proba(X_sim_scaled)[:, 1]
        pred_probs = (pred_probs_lgb + pred_probs_xgb + pred_probs_cat + pred_probs_tabnet + pred_probs_ngb) / 5.0

        # シミュレーションループ
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
                    if df_price['Low'].iloc[i] <= stop_loss_price:
                        exit_price = stop_loss_price
                        break
                
                number_of_shares = initial_investment // start_price
                if number_of_shares == 0: continue
                
                profit_loss = (exit_price - start_price) * number_of_shares * leverage
                
                local_simulation_results.append({
                    "Ticker": ticker, "Company Name": get_company_name(ticker),
                    "Simulation Date": df_price.index[start_idx].date(),
                    "Start Price": start_price, "Stop Loss Price": stop_loss_price,
                    "End Price": df_price['Open'].iloc[end_idx], "Exit Price": exit_price,
                    "Price Difference": exit_price - start_price,
                    "Profit/Loss (JPY)": round(profit_loss, 2),
                    "Predicted Probability": round(pred_probs[sample_idx], 3)
                })
                next_trade_available_idx = sample_idx + deal_term
        
        return local_simulation_results

    except Exception as e:
        print(f"--- シミュレーションエラー ({ticker}) ---")
        traceback.print_exc()
        return [] # エラー時も空リストを返す

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


# def get_daily_ohlc_from_hourly(ticker, period="10y"):
#     """
#     1時間足データから日足OHLCを生成し、最新の価格を1分足データで補完・更新する関数。
#     """
#     # 1. 1時間足データを取得し、日足にリサンプリング
#     df_hourly = yf.download(ticker, period=period, interval="1h", progress=False)
#     if df_hourly.empty:
#         print(f"  {ticker}: 1時間足データ取得できず。")
#         return pd.DataFrame() # 空のDataFrameを返す

#     df_hourly.index = pd.to_datetime(df_hourly.index, utc=True).tz_convert('Asia/Tokyo') # 日本時間に変換
#     df_hourly = df_hourly.sort_index()
#     daily_open = df_hourly['Open'].resample('D').first()
#     daily_high = df_hourly['High'].resample('D').max()
#     daily_low = df_hourly['Low'].resample('D').min()
#     daily_close = df_hourly['Close'].resample('D').last()
#     df_daily = pd.concat([daily_open, daily_high, daily_low, daily_close], axis=1).dropna()
    
#     # 日本時間の午前10時までは1分足で更新トライ
#     RETURN_TIME_LIMIT = datetime.time(10, 0, 0)
#     current_time_jst = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).time()
    
#     if current_time_jst >= RETURN_TIME_LIMIT:
#         print(f"[{ticker}] 現在時刻 {current_time_jst.strftime('%H:%M:%S')} が {RETURN_TIME_LIMIT.strftime('%H:%M:%S')} を過ぎているため、1分足での更新はスキップします。")
#         return df_daily

#     # 2. 直近の1分足データで更新
#     df_minutely = yf.download(ticker, period="7d", interval="1m", progress=False)
#     if not df_minutely.empty:
#         df_minutely.index = pd.to_datetime(df_minutely.index, utc=True).tz_convert('Asia/Tokyo')
#         latest_daily_open = df_minutely['Open'].resample('D').first()
#         latest_daily_high = df_minutely['High'].resample('D').max()
#         latest_daily_low = df_minutely['Low'].resample('D').min()
#         latest_daily_close = df_minutely['Close'].resample('D').last()
#         df_latest_daily = pd.concat([latest_daily_open, latest_daily_high, latest_daily_low, latest_daily_close], axis=1).dropna()
        
#         if not df_latest_daily.empty:
#             df_daily = pd.concat([df_daily, df_latest_daily])
#             df_daily = df_daily[~df_daily.index.duplicated(keep='last')].sort_index()
#     else:
#         print(f"  {ticker}: 1分足データ取得できず。1時間足データのみで処理を続行します。")

#     return df_daily
def get_daily_ohlc_from_hourly_with_latest_1m(ticker, period="10y"):
    """
    1時間足データから日足OHLCを生成し、最新の価格を1分足データで補完・更新する関数。
    yfinanceの更新遅延を考慮し、当日の「1分足」データが取得できるまで30秒ごとに待機する。

    Args:
        ticker (str): ティッカーシンボル
        period (str, optional): 1時間足データの取得期間。デフォルトは "10y"。
        timeout (int, optional): 待機処理のタイムアウト時間（秒）。デフォルトは 1800秒（30分）。
    """
    timeout=1800
    # 1.【初回のみ】1時間足データを取得し、過去の日足データを生成
    df_hourly = yf.download(ticker, period=period, interval="1h", progress=False)
    if df_hourly.empty:
        print(f"  {ticker}: 1時間足データ取得できず。")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"], dtype=float)
    
    print("=" * 50); print(f"[{ticker}] 過去データとして1時間足データを取得完了")
    df_hourly.index = pd.to_datetime(df_hourly.index, utc=True).tz_convert('Asia/Tokyo')
    df_hourly = df_hourly.sort_index()
    
    daily_open = df_hourly['Open'].resample('D').first()
    daily_high = df_hourly['High'].resample('D').max()
    daily_low = df_hourly['Low'].resample('D').min()
    daily_close = df_hourly['Close'].resample('D').last()
    df_daily_base = pd.concat([daily_open, daily_high, daily_low, daily_close], axis=1, keys=['Open', 'High', 'Low', 'Close']).dropna()
    print(f"[{ticker}] 1時間足から過去の日足データを生成完了")
    
    # =================================================================
    # 2. 当日の1分足データがyfinanceに反映されるまでループして待機
    # =================================================================
    start_time = time.time()
    today_date = pd.Timestamp.now(tz='Asia/Tokyo').normalize()

    print("-" * 50)
    print(f"[{ticker}] 当日 ({today_date.strftime('%Y-%m-%d')}) の1分足データが出現するまで待機します。")
    print(f"タイムアウト: {timeout}秒")
    print("-" * 50)

    df_minutely = pd.DataFrame() 
    while True:
        df_minutely_latest = yf.download(ticker, period="7d", interval="1m", progress=False)
        if not df_minutely_latest.empty:
            df_minutely_latest.index = pd.to_datetime(df_minutely_latest.index, utc=True).tz_convert('Asia/Tokyo')
            if today_date.date() in df_minutely_latest.index.date:
                print(f"\n[{ticker}] 当日の1分足データを取得しました！処理を続行します。")
                df_minutely = df_minutely_latest 
                break
            else:
                latest_date_str = df_minutely_latest.index.max().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\r[{ticker}] 当日データ待機中... (最新取得日時: {latest_date_str})  ", end="")
        else:
            print(f"\r[{ticker}] 1分足データ取得試行中... ", end="")
        if time.time() - start_time > timeout:
            print(f"\n[{ticker}] タイムアウト ({timeout}秒) しました。データ更新を中断します。")
            break
        time.sleep(30)

    # =================================================================
    # 3. 取得した最新の1分足データで日足を更新
    # =================================================================
    if not df_minutely.empty:
        print(f"[{ticker}] 取得した1分足データから最新の日足を生成中...")
        latest_daily_open = df_minutely['Open'].resample('D').first()
        latest_daily_high = df_minutely['High'].resample('D').max()
        latest_daily_low = df_minutely['Low'].resample('D').min()
        latest_daily_close = df_minutely['Close'].resample('D').last()
        df_latest_daily = pd.concat([latest_daily_open, latest_daily_high, latest_daily_low, latest_daily_close], axis=1, keys=['Open', 'High', 'Low', 'Close']).dropna()
        
        df_final = pd.concat([df_daily_base, df_latest_daily])
        df_final = df_final[~df_final.index.duplicated(keep='last')].sort_index()
        print(f"[{ticker}] 日足データの最新化が完了しました。")
    else:
        print(f"\n[{ticker}] 最新の1分足データを取得できなかったため、過去データのみを返します。")
        df_final = df_daily_base

    # =================================================================
    # 4. 【追加】後続処理のため、列のMultiIndexを解除して平坦化する
    # =================================================================
    if isinstance(df_final.columns, pd.MultiIndex):
        # 列が ('Open', '1802.T') のようなタプルになっている場合、最初の要素('Open')だけを取り出す
        df_final.columns = df_final.columns.get_level_values(0)
        print(f"[{ticker}] 列のMultiIndexを解除しました。")


    print("=" * 50); print(f"[{ticker}] 処理完了後の最新日足データ（末尾5件）"); print(df_final.tail()); print("=" * 50)
    return df_final

def create_features_for_single_ticker_today(args):
    """
    【並列処理用ワーカー関数】
    単一ティッカーの今日の予測用特徴量を生成する。
    """
    ticker, lookback_days, deal_term = args
    print(f"--- [Worker] 今日の特徴量を生成中: {ticker} ---")
    
    try:
        # 1. データ取得とOHLC生成
        df_daily = get_daily_ohlc_from_hourly_with_latest_1m(ticker, period="2y")
        if df_daily.empty or len(df_daily) < lookback_days:
            return ticker, None

        # 2. 特徴量生成
        df_extra = prepare_features_2(df_daily.copy())
        
        # 3. 最終的な特徴量ベクトルを作成
        df_daily_with_extra = df_daily.join(df_extra, how='inner')
        if len(df_daily_with_extra) < lookback_days:
            return ticker, None

        window = df_daily_with_extra['Open'].iloc[-lookback_days:]
        returns = window.pct_change().dropna().values
        if len(returns) != lookback_days - 1:
            return ticker, None
            
        # extra特徴量は最新のものを取得
        last_dt = df_daily_with_extra.index[-1]
        extra_features = df_extra.loc[last_dt].values
        
        combined_features = np.hstack([returns, extra_features])
        
        if np.isnan(combined_features).any():
            return ticker, None
            
        return ticker, combined_features

    except Exception:
        print(f"--- 今日の特徴量生成エラー ({ticker}) ---")
        traceback.print_exc()
        return ticker, None # エラー時もタプルを返すが、特徴量はNone


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


# =============================================================================
# ▼▼▼ 4. メイン実行関数 ▼▼▼
# =============================================================================
def main():
    """ このスクリプトのメイン処理 """
    # --- 賢いワーカー数の自動設定 ---
    # 利用可能なCPUコアの総数を取得
    total_cpus = multiprocessing.cpu_count()

    # ルール1: CPU使用率を75%に設定
    CPU_USAGE_RATIO = 0.75
    # ルール2: メモリ枯渇を防ぐための絶対的な上限値を設定
    ABSOLUTE_MAX_WORKERS = 32

    # 使用率に基づいてワーカー数を計算（最低でも1は確保）
    desired_workers = max(1, int(total_cpus * CPU_USAGE_RATIO))

    # 計算結果と上限値のうち、小さい方を採用
    MAX_WORKERS = min(desired_workers, ABSOLUTE_MAX_WORKERS)
    # ▲▲▲▲▲【修正ここまで】▲▲▲▲▲

    lookback_days = 25
    deal_term = 1
    TRAIN_RATIO = 0.9
    # ★ 重要な修正: `datetime.datetime`のようにフルパスで指定する
    ONE_MONTH_AGO = datetime.datetime.now() - datetime.timedelta(days=30)
    initial_investment = 2000000
    leverage = 1
    stop_loss_threshold = 0.3
    
    print(f"最大 {MAX_WORKERS} スレッド/プロセスで並列処理を開始します。")
    load_dotenv()
    folder_path = "./分析"; os.makedirs(folder_path, exist_ok=True)


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
            if (f == "top_30_fx_gaitame_by_growth.csv" or  # 元のCSVファイル
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

        top_ranked = pd.DataFrame(all_data).sort_values(by="スコア", ascending=False).head(10)

        top_ranked["銘柄名"] = top_ranked["銘柄コード"].apply(lambda x: ticker_info_map[x].get('銘柄名', 'N/A'))
        top_ranked["カテゴリ"] = top_ranked["銘柄コード"].apply(lambda x: ticker_info_map[x].get('カテゴリ', 'N/A'))
        
        top_ranked["スコア (%)"] = (top_ranked["スコア"] * 100).round(2)
        top_ranked = top_ranked[["銘柄コード", "銘柄名", "カテゴリ", "スコア (%)"]]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
    # 3. モデルの準備（並列処理）
    # =================================================================================
    print("\nモデルの準備（必要に応じて学習・再学習）を並列で開始します...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lgb_models_dict, xgb_models_dict, cat_models_dict, tabnet_models_dict, ngb_models_dict, scalers_dict = {}, {}, {}, {}, {}, {}
    
    successful_tickers = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = {
            executor.submit(process_ticker_models, ticker, ticker_data_dict[ticker], lookback_days, deal_term, TRAIN_RATIO, ONE_MONTH_AGO): ticker
            for ticker in ticker_symbols if ticker in ticker_data_dict
        }
        for future in as_completed(tasks):
            ticker, model_paths = future.result()
            if ticker and model_paths:
                try:
                    lgb_models_dict[ticker] = joblib.load(model_paths['lgb'])
                    xgb_models_dict[ticker] = joblib.load(model_paths['xgb'])
                    cat_models_dict[ticker] = joblib.load(model_paths['cat'])
                    ngb_models_dict[ticker] = joblib.load(model_paths['ngb'])
                    tabnet_model = TabNetClassifier()
                    tabnet_model.load_model(model_paths['tabnet'] + ".zip")
                    tabnet_models_dict[ticker] = tabnet_model
                    scalers_dict[ticker] = joblib.load(model_paths['scaler'])
                    successful_tickers.append(ticker)
                    print(f"  > {ticker}: 全モデルのメインプロセスへのロード完了。")
                except Exception as e:
                    print(f"--- {ticker} のモデルロード中にエラーが発生: {e} ---")

    ticker_symbols = successful_tickers
    print("\n全ティッカーのモデル準備が完了しました。")
    print(f"ロード済みモデル数: LightGBM({len(lgb_models_dict)}), XGBoost({len(xgb_models_dict)}), CatBoost({len(cat_models_dict)}), TabNet({len(tabnet_models_dict)}), NGBoost({len(ngb_models_dict)})")





# =============================================================================
# ▼▼▼ 5. スクリプトのエントリーポイント ▼▼▼
# =============================================================================
if __name__ == '__main__':
    main()