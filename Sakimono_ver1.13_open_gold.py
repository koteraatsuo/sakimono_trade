
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
import os




import yfinance as yf
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta, time as dtime
import time
import os


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
    high_prices = df['High'].shift(1)
    low_prices = df['Low'].shift(1)
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

def quinn_fernandes_extrapolation(data,
                                  calc_bars=60,
                                  harmonic_period=20,
                                  freq_tolerance=0.01,
                                  bars_to_render=500,
                                  n_harmonics=7):
    """
    Quinn-Fernandes 方式で周波数を一つずつ推定し、ハーモニックを加えて再構成する実装。
    正の周波数成分のインデックスずれを防ぎます。
    """
    if len(data) < calc_bars:
        raise ValueError(f'データが {calc_bars} 本より少ないので計算できません。')

    past = np.asarray(data[-calc_bars:], dtype=float)
    n = calc_bars
    t = np.arange(n)

    # 平均を引いた残差
    mean_val = past.mean()
    model = np.zeros(n)
    residue = past - mean_val

    params = []
    for _ in range(n_harmonics):
        # 1) フル FFT と周波数ベクトル
        fft_res = np.fft.fft(residue)
        freqs   = np.fft.fftfreq(n, d=1.0)

        # 2) 正の周波数のみ抽出
        pos_mask  = freqs > 0
        fft_pos   = fft_res[pos_mask]
        freqs_pos = freqs[pos_mask]
        pos_idx   = np.where(pos_mask)[0]

        # 3) マスク後配列からピーク位置を探し、元配列のインデックスへマッピング
        rel_peak = np.argmax(np.abs(fft_pos))    # 0 ～ len(fft_pos)-1
        abs_peak = pos_idx[rel_peak]             # 0 ～ n-1

        # 4) 角周波数を計算
        w = 2.0 * np.pi * freqs[abs_peak]
        if abs(w) < freq_tolerance:
            w = freq_tolerance

        # 5) 最小二乗で振幅・位相を推定
        X = np.column_stack([np.cos(w * t), np.sin(w * t)])
        a, b = np.linalg.lstsq(X, residue, rcond=None)[0]

        # 6) モデルにハーモニック成分を足し込み、残差を更新
        component = a * np.cos(w * t) + b * np.sin(w * t)
        model    += component
        residue   = past - (mean_val + model)
        params.append((w, a, b))

    # 過去データの再構成
    reconst_past = mean_val + model

    # 未来予測
    future_t = np.arange(n, n + bars_to_render)
    reconst_future = np.full(bars_to_render, mean_val)
    for w, a, b in params:
        reconst_future += a * np.cos(w * future_t) + b * np.sin(w * future_t)

    reconst = np.concatenate([reconst_past, reconst_future])
    future_idx = np.arange(len(data) - calc_bars, len(data) - calc_bars + len(reconst))

    return reconst, future_idx

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
        current_close = df['Open'].iloc[i]
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
    既存の特徴量（価格差, Heikin Ashi, Ichimoku, rolling FFTなど）はそのまま残し、
    各行ごとに過去calc_bars件＋未来bars_to_render件に対するQuinn-Fernandes再構成の全予測値から
    最新のcloseとの差分（10倍したもの）を計算して新たな特徴量として追加する。
    
    最終的に、元のdfとrolling_quinn_features_fullの結果をインデックスでinner mergeして返す。
    """

   
    df = pd.DataFrame(master_df)
    
    # もし 'close' 列が大文字なら小文字に統一
    if 'close' not in df.columns and 'Close' in df.columns:
        df.rename(columns={'Close': 'close'}, inplace=True)


    # print(master_df)
    
    # 最低必要データ数チェック
    MIN_DATA_REQUIRED = calc_bars + bars_to_render + 20
    if len(df) < MIN_DATA_REQUIRED:
        raise ValueError(f"データが少なすぎます。最低でも{MIN_DATA_REQUIRED}本必要です。現在は{len(df)}本です。")
    
    # 既存の特徴量計算
    # 価格差
    df['price_diff_1'] = df['Open'].diff(1) * 10.00
    
    # Heikin Ashi
    # df_ha = process_ha_and_streaks(df)
    # for diff in [1,2,3,4,5]:
    #     df[f'ha_close_diff_{diff}'] = df_ha['ha_close'].diff(diff) * 10.00
    
    # Ichimoku
    df = calculate_ichimoku(df, conversion_period, base_period, span_b_period, displacement)
    

    for diff in [1,2,4,8,12,15]:
        df[f'tenkan_diff{diff}'] = df['tenkan_sen'].diff(diff) * 10.00
        df[f'kijun_diff{diff}'] = df['kijun_sen'].diff(diff) * 10.00
        df[f'senkou_a_diff{diff}'] = df['senkou_span_a'].diff(diff) * 10.00
        df[f'senkou_b_diff{diff}'] = df['senkou_span_b'].diff(diff) * 10.00
    # df['tenkan_diff'] = df['tenkan_sen'] - df['close']
    # df['kijun_diff']  = df['kijun_sen'] - df['close']
    # df['senkou_a_diff'] = df['senkou_span_a'] - df['close']
    # df['senkou_b_diff'] = df['senkou_span_b'] - df['close']
    
    # rolling FFT
    df = rolling_fft_features(df, target_col='Open', window=60, freq_list=[1,2,3,5,10])
    
    
    # df = calculate_gaussian_filter(df, length=5, sigma=10)

    # for diff in [1,2,3,4,5,6,8,12,15]:
    #     df[f'Gaussian_diff{diff}'] = df['Gaussian'].diff(diff) * 10.00
    # 各行ごとに局所的なフーリエ再構成の特徴量を計算する



    qf_df = rolling_quinn_features_no_future(
        df, calc_bars=int(60), harmonic_period=20
    )

    # qf_df = rolling_quinn_features_full_learn(
    #     df,
    #     calc_bars=calc_bars,
    #     bars_to_render=bars_to_render,
    #     harmonic_period=harmonic_period,
    #     freq_tolerance=0.01,
    #     n_harmonics=7
    # )
    
    # # 既存のdfとqf_dfをinner merge（共通インデックス部分のみ）
    df_merged = df.merge(qf_df, how='inner', left_index=True, right_index=True)
    
    # 不要な列の削除（存在しない場合はエラー無視）
    df_merged = df_merged.drop(columns=[
        'ha_color', 'tick_volume', 'spread', 'real_volume', "tenkan_sen", "kijun_sen", "senkou_a_diff", "senkou_b_diff", "Gaussian", "open", "high", "low",
        'ha_close','ha_open','ha_high','ha_low', 'senkou_span_a', 'senkou_span_b', "Adj Close", "Volume", 'High', 'Low', 'close',
    ], errors='ignore')
    
    # print(df_merged)
    df_merged.dropna(inplace=True)
    
    if df_merged.empty or df_merged.shape[1] == 0:
        raise ValueError("特徴量が空です。パラメータやデータ量を確認してください。")
    
    return df_merged











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














futures_tickers = {
    # "Indices": [
    #     "ES=F", "NQ=F", "YM=F", "RTY=F", "VIX=F", "DAX=F", "FTSE=F",
    #     "NK=F", "HSI=F", "KOSPI=F", "S&P500", "NASDAQ100"
    # ],
    # "Commodities": [
    #     "CL=F", "BZ=F", "NG=F", "HO=F", "RB=F",
    #     "GC=F", "SI=F", "PL=F", "PA=F",
    #     "HG=F", "ALI=F", "ZC=F", "ZW=F", "ZS=F",
    #     "CC=F", "KC=F", "LB=F", "CT=F", "OJ=F"
    # ],
    # "Currencies": [
    #     "6E=F", "6J=F", "6A=F", "6C=F", "6B=F", "6N=F", "6S=F",
    #     "DX=F"
    # ],
    # "Interest Rates": [
    #     "ZB=F", "ZN=F", "ZF=F", "ZT=F",
    #     "GE=F", "ED=F"
    # ],
    # "Energy": [
    #     "CL=F", "NG=F", "HO=F", "RB=F", "BZ=F",
    #     "QL=F", "QA=F"
    # ],
    "Metals": [
        "GC=F"
    ],
    # "Agriculture": [
    #     "ZC=F", "ZW=F", "ZS=F", "ZM=F", "ZL=F",
    #     "CC=F", "KC=F", "CT=F", "LB=F", "OJ=F"
    # ],
    # "Softs": [
    #     "SB=F", "JO=F", "CC=F", "KC=F"
    # ],
    # "Global Indices": [
    #     "NK=F", "HSI=F", "DAX=F", "FTSE=F", "CAC=F"
    # ]
}





import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# 保存先フォルダの指定（存在しなければ作成）
save_folder = "./分析"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)



all_data = []
print("各銘柄の過去1年の上昇率（スコア）を計算中...")

# 辞書から業種と銘柄を取り出してループ
for genre, tickers_list in futures_tickers.items():
    for ticker in tickers_list:
        # "BRK.B" のようにドットを含むシンボルはハイフンに置換（Yahoo Financeでの表記対応）
        ticker_fixed = ticker.replace(".", "-")

        try:
            # 調整済み価格で取得
            data = yf.download(ticker_fixed, period="1y", progress=False, auto_adjust=True)
        except Exception as e:
            print(f"{ticker}（{genre}）: データ取得エラー - {e}")
            continue

        if data is None or data.empty:
            print(f"{ticker}（{genre}）: データが不足しています。")
            continue

        # 「Close」または「Adj Close」があれば利用
        if "Close" in data.columns:
            prices = data["Close"].dropna()
        elif "Adj Close" in data.columns:
            prices = data["Adj Close"].dropna()
        else:
            print(f"{ticker}（{genre}）: Close系のデータが見つかりません。")
            continue

        if len(prices) < 2:
            print(f"{ticker}（{genre}）: 十分なデータがありません。")
            continue

        try:
            # 1年上昇率を計算
            first_price = prices.iloc[0]
            last_price = prices.iloc[-1]
            growth = (last_price / first_price) - 1
            # スカラー値に変換
            score = float(growth)

            # 企業名を取得
            try:
                ticker_obj = yf.Ticker(ticker_fixed)
                company_name = ticker_obj.info.get("shortName", "不明")
            except Exception as e:
                company_name = "不明"

            # リストに格納
            all_data.append({
                "業種": genre,
                "銘柄コード": ticker,
                "企業名": company_name,
                "スコア": score
            })

        except Exception as e:
            print(f"{ticker}（{genre}）: 計算エラー - {e}")
            continue

# DataFrame化し、スコアの高い順にソート、上位12銘柄を抽出
df_all = pd.DataFrame(all_data, columns=["業種", "銘柄コード", "企業名", "スコア"])
df_sorted = df_all.sort_values(by="スコア", ascending=False).reset_index(drop=True)
top12 = df_sorted.head(30)

print("\n1年上昇率（スコア）が高い上位12銘柄:")
print(top12)

# 結果をExcelとして保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
top12_path = os.path.join(save_folder, f"cfd_top12_{timestamp}.xlsx")
top12.to_excel(top12_path, index=False, sheet_name="Top Stocks")

print(f"\n上位12銘柄の結果が保存されました: {top12_path}")

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import lightgbm as lgb
import joblib
import json
import os
import time

import warnings
import re

# # "force_all_finite" を含むメッセージの FutureWarning だけを無視
# warnings.filterwarnings(
#     "ignore",
#     message=re.compile(".*force_all_finite.*"),
#     category=FutureWarning
# )

# ---------------------------
# 初期設定・ティッカーリストの読み込み
# ---------------------------
folder_path = "./分析"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Excelファイルからティッカーリストを読み込み（例："cfd_top12_20250301_120249.xlsx" を使用）
# file_path = "cfd_top8_20250302_201017.xlsx"
# file_path = "cfd_top12_20250301_120249.xlsx"
file_path = top12_path

df_tickers = pd.read_excel(file_path)
df_tickers = df_tickers[["業種", "銘柄コード", "企業名"]]
tickers_dict = {row["銘柄コード"]: row["企業名"] for _, row in df_tickers.iterrows()}
ticker_symbols = list(tickers_dict.keys())

# ---------------------------
# データ取得：複数ティッカーの10年分のデータを一括取得し、各ティッカーごとに分割
# ---------------------------
print("データ取得中...")
data = yf.download(ticker_symbols, period="10y", progress=False, auto_adjust=False, group_by='ticker')


ticker_data_dict = {}
available_tickers = set(data.columns.get_level_values(0).unique())
for ticker in ticker_symbols:
    if ticker in available_tickers:
        ticker_data_dict[ticker] = data.xs(ticker, axis=1, level=0)
    else:
        print(f"Ticker {ticker} は取得されていません。")




# 設定
initial_investment = 120000  # 初期投資金額 (円)
leverage = 3  # レバレッジ倍率
stop_loss_threshold = 0.07  # 損切りライン (7%)
price_threshold = 5000  # 最大投資対象株価
lookback_days = 30  # 過去25日間のデータを使用
deal_term = 5  # 投資期間を5日に設定
interest_rate = 0.028  # 年率2.8%の日利換算

# アクティブプランの手数料体系
active_plan_fees = {
    100000: 0,
    200000: 148,
    500000: 198,
    1000000: 385,
    2000000: 880,
    "increment": 440,
}

# 手数料計算関数
def calculate_active_plan_fee(total_trade_amount):
    if total_trade_amount <= 100000:
        return active_plan_fees[100000]
    elif total_trade_amount <= 200000:
        return active_plan_fees[200000]
    elif total_trade_amount <= 500000:
        return active_plan_fees[500000]
    elif total_trade_amount <= 1000000:
        return active_plan_fees[1000000]
    elif total_trade_amount <= 2000000:
        return active_plan_fees[2000000]
    else:
        increments = (total_trade_amount - 2000000) // 1000000
        return active_plan_fees[2000000] + increments * active_plan_fees["increment"]


# 例：yfinanceからティッカーの情報を取得する関数
def get_company_name(ticker_code):
    try:
        ticker_obj = yf.Ticker(ticker_code)
        info = ticker_obj.info
        # shortNameが取得できなければ、longNameを試す
        company_name = info.get("shortName") or info.get("longName")
        # どちらもなければ、ティッカーコードをそのまま返す
        return company_name if company_name is not None else ticker_code
    except Exception as e:
        print(f"Error retrieving info for {ticker_code}: {e}")
        return ticker_code


# エクセルファイルのパスを指定
# # file_path = "top_stocks_全体用固定_0.2_0.2_急激.xlsx"
# file_path = "top_stocks_全体用固定_0.2_0.2.xlsx"
# file_path = "top_stocks_全体用固定_0.2_0.2_20250215_140207.xlsx"
# file_path = top12_path

# df_tickers = pd.read_excel(file_path)
# df_tickers = df_tickers[["業種", "銘柄コード", "企業名"]]
# tickers_dict = {row["銘柄コード"]: row["企業名"] for _, row in df_tickers.iterrows()}
# ticker_symbols = list(tickers_dict.keys())

# # ---------------------------
# # データ取得：複数ティッカーの10年分のデータを一括取得し、各ティッカーごとに分割
# # ---------------------------
# print("データ取得中...")
# data = yf.download(ticker_symbols, period="10y", progress=False, auto_adjust=False, group_by='ticker', threads=False)

# ticker_data_dict = {}
# available_tickers = set(data.columns.get_level_values(0).unique())
# for ticker in ticker_symbols:
#     if ticker in available_tickers:
#         ticker_data_dict[ticker] = data.xs(ticker, axis=1, level=0)
#     else:
#         print(f"Ticker {ticker} は取得されていません。")


simulation_results = []

best_num = 0
best_score = 0
record_investments = []
winrates = []

# ---------------------------
# シミュレーション・学習パラメータ
# ---------------------------
initial_investment = 100000   # 各ティッカーごとに全額投入する例
leverage = 10
stop_loss_threshold = 0.3
lookback_days = 5 * 3
deal_term = 5

# ---------------------------
# 特徴量・ラベル生成関数（各ティッカー用）
# 返り値に「日付リスト」と「開始インデックスリスト」も追加
# ---------------------------


def create_features_labels_dates_for_ticker(df_price, lookback_days=25, deal_term=5):
    """
    ・df_price: yfinanceで取得した日足データ (必須列: "Close")
    ・特徴量: 過去 lookback_days 日間のリターン（pct_change） [形状=(lookback_days-1,)]
    ・ラベル: その直後の deal_term 日後に価格が上昇しているか (1) 下降か (0)
    ・返り値: X, y, dates_list, indices_list
        dates_list: 各サンプルの「取引開始日」（df_price.index[ i + lookback_days - 1 ]）
        indices_list: 各サンプルの開始インデックス (i + lookback_days - 1)
    """
    df_price = df_price[['Open']].dropna()
    features = []
    labels = []
    dates_list = []
    indices_list = []
    for i in range(len(df_price) - lookback_days - deal_term):
        window = df_price['Open'].iloc[i : i + lookback_days]
        if len(window) < lookback_days:
            continue
        returns = window.pct_change().dropna()
        if len(returns) != lookback_days - 1:
            continue
        current_close = df_price['Open'].iloc[i + lookback_days - 1]
        future_close = df_price['Open'].iloc[i + lookback_days - 1 + deal_term]
        label = 1 if future_close > current_close else 0
        features.append(returns.values)
        labels.append(label)
        start_idx = i + lookback_days - 1
        dates_list.append(df_price.index[start_idx])
        indices_list.append(start_idx)
    X = np.array(features)
    y = np.array(labels)
    return X, y, dates_list, indices_list



def create_combined_features(df_price,
                             lookback_days=25,
                             deal_term=5):
    """
    df_price の 'Open' 列を使い、
      1) 過去 lookback_days 日間のリターン（pct_change）
      2) prepare_features_2 の全追加特徴量
    を同一インデックス（日付）ベースでドッキングして返します。

    返り値:
      X_combined: ndarray, shape=(N_samples, (lookback_days-1) + n_extra_features)
      y: ndarray, shape=(N_samples,)
      dates_list: list of pd.Timestamp, サンプル開始日
      indices_list: list of int, サンプル開始インデックス
    """

    # ──(1) 追加特徴量を一括計算──
    df_extra = prepare_features_2(
        df_price,
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
    # print(df_extra.columns)
    n_extra = df_extra.shape[1]
    
    # 出力リスト初期化
    X_combined_list = []
    y_list = []
    dates_list = []
    indices_list = []

    # ──(2) 各サンプルをループで回す──
    for i in range(len(df_price) - lookback_days - deal_term):
        # (a) lookback window
        window = df_price['Open'].iloc[i : i + lookback_days]
        if len(window) < lookback_days:
            continue
        
        # (b) リターン特徴量
        returns = window.pct_change().dropna().values  # shape = (lookback_days-1,)
        if returns.shape[0] != lookback_days - 1:
            continue

        # (c) ラベル作成
        idx_close = i + lookback_days - 1
        current = df_price['Open'].iat[idx_close]
        future  = df_price['Open'].iat[idx_close + deal_term]
        label = 1 if future > current else 0

        # (d) 日付とインデックス
        dt = df_price.index[idx_close]

        # (e) 追加特徴量取得
        if dt in df_extra.index:
            extra = df_extra.loc[dt].values
        else:
            extra = np.zeros(n_extra, dtype=float)

        # (f) ドッキング
        combined = np.hstack([returns, extra])

        # (g) 保存
        X_combined_list.append(combined)
        y_list.append(label)
        dates_list.append(dt)
        indices_list.append(idx_close)

    # ──(3) NumPy 配列に変換して返却──
    X_combined = np.vstack(X_combined_list)
    y = np.array(y_list, dtype=int)

    return X_combined, y, dates_list, indices_list

# 使い方例
# master_df = ...   # 元データフレーム（'Close' or 'close' 列あり）
# df_price  = ...   # yfinance取得データ（'Open' 列あり）


# ---------------------------
# 各ティッカーごとにモデル学習（ここではLightGBMを例示）
# ---------------------------
models_dict = {}
scalers_dict = {}
performance_dict = {}  # 各ティッカーのテストスコアなど

for ticker in ticker_symbols:
    if ticker not in ticker_data_dict:
        print(f"{ticker}: データが存在しないためスキップ。")
        continue
    df_price = ticker_data_dict[ticker]
    # X, y = create_features_labels_dates_for_ticker(df_price, lookback_days, deal_term)[:2]
    
    # print(df_price)
    X, y, dates, idxs = create_combined_features(
                            df_price,
                            lookback_days=25,
                            deal_term=5,
                            # # 以下は prepare_features_2 に必要なパラメータ
                            # num_1=59,
                            # num_2=10,
                            # conversion_period=2,
                            # base_period=7,
                            # span_b_period=45,
                            # displacement=22,
                            # calc_bars=60,
                            # bars_to_render=10,
                            # harmonic_period=10,
                        )
    
    if len(X) < 10:
        print(f"{ticker}: 学習データが不足しているためスキップ。")
        continue

    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    param_grid = {
        'n_estimators': [300],
        'learning_rate': [0.3],
        'max_depth': [7]
    }
    grid_search = GridSearchCV(lgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"[{ticker}] Best Params: {grid_search.best_params_}, Test Score: {test_score:.3f}")
    
    models_dict[ticker] = best_model
    scalers_dict[ticker] = scaler
    performance_dict[ticker] = {"Test Score": test_score}

# ---------------------------
# 各ティッカーごとのシミュレーション（最新のサンプルの後半10%を対象）
# ---------------------------



# stop_loss_threshold =  0.2
for ticker in ticker_symbols:
    if ticker not in ticker_data_dict or ticker not in models_dict or ticker not in scalers_dict:
        continue
    df_price = ticker_data_dict[ticker]
    model = models_dict[ticker]
    scaler = scalers_dict[ticker]
    
    # 特徴量・ラベル・日付・開始インデックスの全サンプルを作成
    X_sim, _, dates_sim, indices_sim = create_combined_features(
                            df_price,
                            lookback_days=25,
                            deal_term=5
                        )
    if len(X_sim) == 0:
        print(f"{ticker}: シミュレーション用データが不足しています。")
        continue
    X_sim_scaled = scaler.transform(X_sim)
    pred_probs = model.predict_proba(X_sim_scaled)[:, 1]
    
    # 最新のサンプル群（後半10%）を対象とする
    sim_count = max(1, int(len(X_sim) * 0.1))
    end_idx = len(X_sim) - sim_count
    next  = 0 
    for sample_idx in range(len(X_sim) - sim_count, len(X_sim)):

        if sample_idx >= next:
            next = sample_idx + deal_term
        # if True:
            simulation_prob = pred_probs[sample_idx]
            start_idx = indices_sim[sample_idx]
            simulation_date = df_price.index[start_idx]
            end_idx = start_idx + deal_term
            if end_idx >= len(df_price):
                print(f"{ticker}: シミュレーション期間が不足しています。")
                continue

            start_price = df_price['Open'].iloc[start_idx]
            end_price = df_price['Open'].iloc[end_idx]
            stop_loss_price = start_price * (1 - stop_loss_threshold)
            exit_price = end_price  # 初期値

            entry_date = df_price.index[start_idx].date()
            for i in range(start_idx, end_idx):
                iter_date = df_price.index[i].date()
                if iter_date == entry_date:
                    if df_price['Low'].iloc[i] <= stop_loss_price:
                        exit_price = stop_loss_price
                        break
                else:
                    current_open = df_price['Open'].iloc[i]
                    if current_open <= stop_loss_price:
                        exit_price = current_open
                        break
                    if df_price['Low'].iloc[i] <= stop_loss_price:
                        exit_price = stop_loss_price
                        break

            number_of_shares = initial_investment // start_price
            if number_of_shares == 0:
                print(f"{ticker}: 取引数量が0のためスキップ。")
                continue
            actual_investment = number_of_shares * start_price
            fee = 0.005 * actual_investment
            price_difference = exit_price - start_price
            profit_loss = price_difference * number_of_shares * leverage - fee

            simulation_results.append({
                "Ticker": ticker,
                "Company Name": tickers_dict.get(ticker, ""),
                "Simulation Date": simulation_date.date(),
                "Start Price": start_price,
                "Stop Loss Price": stop_loss_price,
                "End Price": end_price,
                "Exit Price": exit_price,
                "Price Difference": price_difference,
                "Number of Shares": int(number_of_shares),
                "Profit/Loss (JPY)": round(profit_loss, 2),
                "Predicted Probability": round(simulation_prob, 3)
            })

# ---------------------------
# シミュレーション結果をExcelに保存
# ---------------------------
df_simulation = pd.DataFrame(simulation_results)
simulation_file_name = f"./分析/simulation_results_tickerwise_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
df_simulation.to_excel(simulation_file_name, index=False)
print(f"\nTickerごとのシミュレーション結果が '{simulation_file_name}' に保存されました。")


# ---------------------------
# 各deal（シミュレーション日）ごとに、その日の上位3銘柄で再投資効果と勝率を計算
# ---------------------------
if not df_simulation.empty:
    # シミュレーション日ごとに処理するため、ユニークなSimulation Dateの一覧を取得してソート
    simulation_dates = sorted(df_simulation["Simulation Date"].unique())
    
    current_portfolio = initial_investment  # 初期資産
    portfolio_progression = []              # 各取引ごとの資産推移
    total_trades = 0                        # 総取引数
    winning_trades = 0                      # 勝ちトレード数

    # 各シミュレーション日の処理（例）
    for sim_date in simulation_dates:
        # 該当日の全取引を抽出
        df_deals = df_simulation[df_simulation["Simulation Date"] == sim_date]
        # その日の上位3銘柄を予測確率順に選択
        df_top3_deal = df_deals.sort_values(by="Predicted Probability", ascending=False).head(3)
        
        # さらに確率が0.8以上のものだけ残す
        df_top3_filtered = df_top3_deal[df_top3_deal["Predicted Probability"] >= 0.8]
        
        # フィルタ後の銘柄を処理
        for idx, row in df_top3_filtered.iterrows():
            total_trades += 1

            # 取引のリターン（増減割合）の計算
            percentage_return = ((row["Exit Price"] - row["Start Price"]) / row["Start Price"]) * leverage

            # 勝ちトレードの判定（リターンがプラスの場合）
            if percentage_return > 0:
                winning_trades += 1

            # １トレードあたりのポートフォリオ配分比率
            weight = 1.0 / len(df_top3_deal)
        
            # 再投資として、現在のポートフォリオ額に対してリターンを適用
            current_portfolio *= (1 + (percentage_return * weight))
            # yfinanceを利用してティッカーコードから会社名を取得
            company_name = get_company_name(row["Ticker"])
            
            portfolio_progression.append({
                "Simulation Date": sim_date,
                "Ticker": row["Ticker"],
                "Company Name": company_name,
                "Trade Return (%)": percentage_return * 100,  # パーセント表記
                "Entry Price": row["Start Price"],
                "Exit Price": row["Exit Price"],
                "Stop Loss Price": row["Stop Loss Price"],
                "Updated Portfolio": current_portfolio,
                "Win Rate (%)": (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            })
            print("Simulation Date", row["Simulation Date"],
                  "Ticker", row["Ticker"],
                  "Company Name", company_name,
                  "Trade Return (%)", percentage_return * 100,
                  "Updated Portfolio", current_portfolio)

    # 通算勝率の計算
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    df_portfolio_progression = pd.DataFrame(portfolio_progression)
    print("\n各deal（シミュレーション日）ごとの上位3銘柄による再投資シミュレーション結果:")
    print(df_portfolio_progression)
    
    print(f"\n通算勝率: {win_rate:.2f}% （全{total_trades}取引中、{winning_trades}取引が利益）")
    # # シミュレーション日ごとに処理するため、ユニークなSimulation Dateの一覧を取得してソート
    # simulation_dates = sorted(df_simulation["Simulation Date"].unique())
    
    # current_portfolio = initial_investment  # 初期資産
    # portfolio_progression = []              # 各取引ごとの資産推移
    # total_trades = 0                        # 総取引数
    # winning_trades = 0                      # 勝ちトレード数

    # # 各シミュレーション日の処理
    # for sim_date in simulation_dates:
    #     # 該当日の全取引を抽出
    #     df_deals = df_simulation[df_simulation["Simulation Date"] == sim_date]
    #     # その日の上位3銘柄を予測確率順に選択
    #     df_top3_deal = df_deals.sort_values(by="Predicted Probability", ascending=False).head(3)
        
    #     # 当日の上位3銘柄それぞれについて処理
    #     for idx, row in df_top3_deal.iterrows():
    #         total_trades += 1

    #         # 取引のリターン（増減割合）の計算
    #         percentage_return = ((row["Exit Price"] - row["Start Price"]) / row["Start Price"]) * leverage

    #         # 勝ちトレードの判定（リターンがプラスの場合）
    #         if percentage_return > 0:
    #             winning_trades += 1

    #         # 再投資として、現在のポートフォリオ額に対してリターンを適用
    #         current_portfolio *= (1 + percentage_return)
            
    #         portfolio_progression.append({
    #             "Simulation Date": sim_date,
    #             "Ticker": row["Ticker"],
    #             "Trade Return (%)": percentage_return * 100,  # パーセント表記
    #             "Updated Portfolio": current_portfolio
    #         })
    #         print("Simulation Date", row["Simulation Date"],"Ticker", row["Ticker"],"Trade Return (%)", percentage_return * 100, "Updated Portfolio", current_portfolio)

    # # 通算勝率の計算
    # win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    # df_portfolio_progression = pd.DataFrame(portfolio_progression)
    # print("\n各deal（シミュレーション日）ごとの上位3銘柄による再投資シミュレーション結果:")
    # print(df_portfolio_progression)
    
    # print(f"\n通算勝率: {win_rate:.2f}% （全{total_trades}取引中、{winning_trades}取引が利益）")
    
    # 結果をExcelに保存
    top3_file_name = f"./分析/top3_recommendations_by_deal_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    df_simulation.to_excel(top3_file_name, index=False)
    portfolio_file_name = f"./分析/portfolio_progression_reinvestment_by_deal_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    df_portfolio_progression.to_excel(portfolio_file_name, index=False)
    
    print(f"\nシミュレーション結果全体が '{top3_file_name}' に保存されました。")
    print(f"再投資効果を反映した資産推移が '{portfolio_file_name}' に保存されました。")
else:
    print("シミュレーション結果が存在しません。")
# ---------------------------
# 各モデルの保存（オプション）
# ---------------------------
os.makedirs('./temp_model', exist_ok=True)
for ticker, model in models_dict.items():
    model_file = os.path.join('./temp_model', f"{ticker}_best_model.pkl")
    joblib.dump(model, model_file)
print("各ティッカーごとのモデルを './temp_model' に保存しました。")

with open('optimized_parameters_tickerwise.json', 'w') as f:
    json.dump(performance_dict, f, indent=4)
print("各ティッカーのパフォーマンス情報を 'optimized_parameters_tickerwise.json' に保存しました。")






# `results_df` が提供されたデータと仮定

# Date列をdatetime型に変換
df_simulation["Date"] = pd.to_datetime(df_simulation["Simulation Date"])

# 曜日列を追加（0=月曜日, ..., 6=日曜日）
df_simulation["Weekday"] = df_simulation["Date"].dt.dayofweek

# 曜日ごとに勝率を計算
# 勝ち（Profit/Loss (JPY) > 0）をカウント
weekday_stats = df_simulation.groupby("Weekday").apply(
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
ticker_stats = df_simulation.groupby("Ticker").apply(
    lambda x: pd.Series({
        "勝利数": (x["Profit/Loss (JPY)"] > 0).sum(),
        "取引数": len(x),
        "勝率": (x["Profit/Loss (JPY)"] > 0).mean()
    })
).reset_index()

# ユニークなティッカーごとに会社名を取得
unique_tickers = ticker_stats["Ticker"].unique()
ticker_to_company = { ticker: get_company_name(ticker) for ticker in unique_tickers }

# 取得した会社名をDataFrameに追加（マッピング）
ticker_stats["Company Name"] = ticker_stats["Ticker"].map(ticker_to_company)

# カラムの並び順を調整（任意）
ticker_stats = ticker_stats[["Ticker", "Company Name", "勝利数", "取引数", "勝率"]]

# 結果を見やすく表示
print("\nTickerごとの勝率:")
print(ticker_stats)

# 結果をエクセルに保存
ticker_winrates_output_file = f"./分析/ticker_winrates_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
ticker_stats.to_excel(ticker_winrates_output_file, index=False)
print(f"\nTickerごとの勝率が '{ticker_winrates_output_file}' に保存されました。")
now = datetime.now()
# 予測対象の銘柄と、その銘柄コード→企業名の辞書
file_path = file_path
df = pd.read_excel(file_path)
df = df[["業種", "銘柄コード", "企業名"]]
tickers = {row["銘柄コード"]: row["企業名"] for _, row in df.iterrows()}
ticker_symbols = list(tickers.keys())

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
        threshold = dtime(6, 0, 0)
    else:
        # 冬時間の場合
        threshold = dtime(7, 0, 0)
    
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

# --- 各ティッカーのモデルを用いて本日の予測確率を算出 ---
predictions = []
if len(X_today) > 0:
    # 各ティッカーごとに個別のモデルで予測（X_todayの各行は shape (lookback_days-1,) になっているので、2次元化して利用）
    for ticker, feature_vector in zip(tickers_today, X_today):
        if ticker in models_dict and ticker in scalers_dict:
            scaler = scalers_dict[ticker]
            model = models_dict[ticker]
            x_2d = feature_vector.reshape(1, -1)
            x_scaled = scaler.transform(x_2d)
            prob = model.predict_proba(x_scaled)[0, 1]
            predictions.append({"Ticker": ticker, "Probability": prob})
        else:
            print(f"{ticker}: モデルまたはスケーラーが存在しません。")
    today_data = pd.DataFrame(predictions).sort_values("Probability", ascending=False)
else:
    today_data = pd.DataFrame(columns=["Ticker", "Probability"])

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

for top_n, purchase_recommendations in [(3, purchase_recommendations_top3), (5, purchase_recommendations_top5), (6, purchase_recommendations_top6)]:
    top_stocks_rec = recommendation_data.head(top_n)
    for idx, row in top_stocks_rec.iterrows():
        ticker_ = row["Ticker"]
        company_name = tickers.get(ticker_, "")
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

    raito = current_portfolio / initial_investment
    profit = current_portfolio - initial_investment

    # HTML形式の表を作成
    weekday_table_html = weekday_stats.to_html(index=False, justify="center", border=1)
    ticker_table_html = ticker_stats.to_html(index=False, justify="center", border=1)
    top5_stocks_html = purchase_df_top5.to_html(index=False, justify="center", border=1) if not purchase_df_top5.empty else ""
    df_portfolio_progression["Win Rate (%)"] = (df_portfolio_progression["Win Rate (%)"]).round(2)
    df_portfolio_progression = df_portfolio_progression.round(2)
    simulation_results_html = df_portfolio_progression.to_html(index=False, justify="center", float_format="{:.2f}".format, border=1, escape=False) if not df_portfolio_progression.empty else ""

    for recipient in recipient_list:
        # メールの作成
        msg = MIMEMultipart("related")
        msg["From"] = GMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = f"Metal_V2　購入リストのフィクションおすすめ結果 ({current_date}) {int(current_portfolio)}円 {raito:.2f}倍"

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

