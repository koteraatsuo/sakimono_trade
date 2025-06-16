
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





















import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# 保存先フォルダの指定（存在しなければ作成）
save_folder = "./分析"
os.makedirs(save_folder, exist_ok=True)

# クロス円通貨ペアのリスト（Yahoo Finance形式）
# currency_pairs = [
#     "EURJPY=X",  # ユーロ／円
#     "GBPJPY=X",  # 英ポンド／円
#     "AUDJPY=X",  # 豪ドル／円
#     "NZDJPY=X",  # NZドル／円
#     "CADJPY=X",  # カナダドル／円
#     "CHFJPY=X",  # スイスフラン／円
# ]

currency_pairs = [

        "ES=F",    # S&P 500 先物
        "NQ=F",    # NASDAQ-100 先物
        "YM=F",    # Dow Jones 先物
        "RTY=F",   # Russell2000 先物
        # "^GSPC",   # S&P 500 インデックス
        # "^NDX",    # NASDAQ-100 インデックス
    ]



print("クロス円通貨ペアの過去1年データを取得中...")

# 各ペアのデータを取得してリストに格納
all_data = {}
ticker_symbols = currency_pairs
# for symbol in currency_pairs:
#     try:
#         df = yf.download(symbol, period="1y", progress=False, auto_adjust=True)
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

# # ---------------------------
# # 前回のダウンロードデータと比較する処理
# # ---------------------------
# prev_file = "./analysis/prev_ticker_data.pkl"
# data_changed = False

# if os.path.exists(prev_file):
#     with open(prev_file, "rb") as f:
#         prev_ticker_data_dict = pickle.load(f)
    
#     for ticker, df_current in ticker_data_dict.items():
#         if ticker in prev_ticker_data_dict:
#             df_prev = prev_ticker_data_dict[ticker]
#             # DataFrame の内容が全く同じかを判定（インデックスや列も含む）
#             if df_current.equals(df_prev):
#                 print(f"{ticker}: ダウンロードデータは前回と同じです。")
#             else:
#                 print(f"{ticker}: ダウンロードデータは前回と異なります。")
#                 data_changed = True
#         else:
#             print(f"{ticker}: 前回のデータが存在しません。")
#             data_changed = True
# else:
#     print("前回のダウンロードデータが存在しません。")
#     data_changed = True

# if data_changed:
#     print("いくつかのティッカーでダウンロードデータに差分があります。")
# else:
#     print("すべてのティッカーでダウンロードデータは前回と同じです。")

# # ---------------------------
# # 今回のダウンロードデータを保存（次回比較用）
# # ---------------------------
# os.makedirs("./analysis", exist_ok=True)
# with open(prev_file, "wb") as f:
#     pickle.dump(ticker_data_dict, f)
# print("今回のダウンロードデータを保存しました。")










# 設定
initial_investment = 120000  # 初期投資金額 (円)
leverage = 3  # レバレッジ倍率
# stop_loss_threshold = 0.07  # 損切りライン (7%)
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
leverage = 5
take_profit_threshold = 0.02
stop_loss_threshold = 0.02

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

    # ──(3) NumPy 配列に変換aして返却──
    X_combined = np.vstack(X_combined_list)
    y = np.array(y_list, dtype=int)

    return X_combined, y, dates_list, indices_list

# 使い方例
# master_df = ...   # 元データフレーム（'Close' or 'close' 列あり）
# df_price  = ...   # yfinance取得データ（'Open' 列あり）

# ----------------------------------------
# ④ モデル学習：ロング/ショート別に
# ----------------------------------------
models_dict  = {}
scalers_dict = {}
perf_dict    = {}

for ticker, df_price in ticker_data_dict.items():
    # 特徴量・ラベル作成
    X, y, dates, idxs = create_combined_features(
        df_price,
        lookback_days=lookback_days,
        deal_term=deal_term
    )
    if len(X) < 20:
        print(f"{ticker}: データ不足でスキップ")
        continue

    # ラベル分割
    y_long  = y.copy()           # 上昇なら1, 下落なら0
    y_short = 1 - y_long         # ショートは下落成功が1

    # 学習/テスト分割
    split = int(len(X) * 0.9)
    X_tr, X_te       = X[:split], X[split:]
    yL_tr, yL_te     = y_long[:split],  y_long[split:]
    yS_tr, yS_te     = y_short[:split], y_short[split:]

    # スケーラー
    scaler = StandardScaler().fit(X_tr)
    Xtr_s = scaler.transform(X_tr)
    Xte_s = scaler.transform(X_te)

    # グリッドサーチ設定
    base    = lgb.LGBMClassifier(random_state=42, verbose=-1)
    param   = {'n_estimators':[300], 'learning_rate':[0.3], 'max_depth':[7]}

    # ロングモデル
    gsL = GridSearchCV(base, param, cv=3, scoring='roc_auc', n_jobs=-1)
    gsL.fit(Xtr_s, yL_tr)
    long_model = gsL.best_estimator_
    scoreL     = long_model.score(Xte_s, yL_te)

    # ショートモデル
    gsS = GridSearchCV(base, param, cv=3, scoring='roc_auc', n_jobs=-1)
    gsS.fit(Xtr_s, yS_tr)
    short_model = gsS.best_estimator_
    scoreS      = short_model.score(Xte_s, yS_te)

    print(f"[{ticker}] Long AUC: {scoreL:.3f}, Short AUC: {scoreS:.3f}")

    models_dict[ticker]  = {"Long": long_model, "Short": short_model}
    scalers_dict[ticker] = scaler
    perf_dict[ticker]    = {"Long": scoreL, "Short": scoreS}

# モデル辞書を保存（必要なら）
with open(os.path.join(save_folder, "models_long_short.pkl"), "wb") as f:
    joblib.dump(models_dict, f)

# ----------------------------------------
# ⑤ シミュレーション：ロング／ショート両対応
# ----------------------------------------
simulation_results = []

for ticker, mdl_pair in models_dict.items():
    df_price = ticker_data_dict[ticker]
    
    
    scaler   = scalers_dict[ticker]

    # シミュレーション用の全サンプル
    X_sim, _, dates_sim, idxs_sim = create_combined_features(df_price,
        lookback_days=lookback_days,
        deal_term=deal_term
    )
    if len(X_sim) == 0:
        continue

    Xs = scaler.transform(X_sim)
    probs = {
        "Long":  mdl_pair["Long"].predict_proba(Xs)[:,1],
        "Short": mdl_pair["Short"].predict_proba(Xs)[:,1]
    }

    sim_count = max(1, int(len(Xs) * 0.1))
    start_sim = len(Xs) - sim_count
    
    # 予測確率の閾値

    pred_threshold = 0.9
    for position in ["Long", "Short"]:
        for i in range(start_sim, len(Xs)):
            prob = probs[position][i]
            if prob < pred_threshold:
                continue

            bar = idxs_sim[i]
            if bar + deal_term >= len(df_price):
                continue

            # 位置ベースで取得
            entry_price = df_price['Open'].iat[bar]
            exit_price  = df_price['Open'].iat[bar + deal_term]

            # ストップロス水準
            if position == "Long":
                tp_price = entry_price * (1 + take_profit_threshold)
                sl_price = entry_price * (1 - stop_loss_threshold)
                high_series = df_price['High'].iloc[bar:bar+deal_term+1]
                low_series = df_price['Low'].iloc[bar:bar+deal_term+1]
                if (high_series >= tp_price).any():
                    exit_price = tp_price
                if (low_series <= sl_price).any():
                    exit_price = sl_price
            else:
                tp_price = entry_price * (1 - take_profit_threshold)
                sl_price = entry_price * (1 + stop_loss_threshold)
                high_series = df_price['High'].iloc[bar:bar+deal_term+1]
                low_series = df_price['Low'].iloc[bar:bar+deal_term+1]
                if (high_series >= sl_price).any():
                    exit_price = sl_price
                if (low_series <= tp_price).any():
                    exit_price = tp_price

            # 日付取得（必要に応じて）
            sim_date = df_price.index[bar].date()

            # 取引数量・手数料
            num_shares = int(initial_investment // entry_price)
            if num_shares == 0:
                continue
            invest = num_shares * entry_price
            fee    = 0.000 * invest

            # 損益・リターン計算
            if position == "Long":
                pnl = (exit_price - entry_price) * num_shares - fee
                ret = (exit_price / entry_price - 1) * leverage
            else:
                pnl = (entry_price - exit_price) * num_shares - fee
                ret = (entry_price / exit_price - 1) * leverage

            simulation_results.append({
                "Ticker":          ticker,
                "Position":        position,
                "Simulation Date": sim_date,
                "Entry Price":     entry_price,
                "Stop Loss Price": sl_price,
                "Exit Price":      exit_price,
                "Predicted Prob":  round(prob, 3),
                "Number of Shares": num_shares,
                "Profit/Loss (JPY)": round(pnl, 2),
                "Return (%)":      round(ret * 100, 2)
            })


# 結果を DataFrame 化して保存
df_simulation = pd.DataFrame(simulation_results)
simulation_file_name = f"./分析/simulation_long_short_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
df_simulation.to_excel(simulation_file_name, index=False)
print(f"ロング／ショート両対応のシミュレーション結果を保存しました: {simulation_file_name}")

# # ---------------------------
# # シミュレーション結果をExcelに保存
# # ---------------------------
# df_simulation = pd.DataFrame(simulation_results)
# simulation_file_name = f"./分析/simulation_results_tickerwise_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
# df_simulation.to_excel(simulation_file_name, index=False)
# print(f"\nTickerごとのシミュレーション結果が '{simulation_file_name}' に保存されました。")

df_sim = df_simulation  

# ----------------------------------------
# 各deal（シミュレーション日）ごとに、その日の上位1銘柄で再投資効果と勝率を計算
# deal_term ごとに飛ぶように修正
# ----------------------------------------
if not df_simulation.empty:
    # ユニークなシミュレーション日をソートしてリスト化
    simulation_dates = sorted(df_simulation["Simulation Date"].unique())
    
    current_portfolio   = initial_investment
    total_trades        = 0
    winning_trades      = 0
    portfolio_progression = []

    # インデックスを手動管理
    i = 0
    while i < len(simulation_dates):
        sim_date = simulation_dates[i]

        for position in ["Long", "Short"]:
            # 当日＋ポジションで絞り込み
            df_deals = df_simulation[
                (df_simulation["Simulation Date"] == sim_date) &
                (df_simulation["Position"]        == position)
            ]
            # 上位1銘柄を確率順に取得
            df_top = (
                df_deals
                .sort_values(by="Predicted Prob", ascending=False)
                .head(1)
            )
            # 閾値フィルタ

            if position == "Long":
                df_top = df_top[df_top["Predicted Prob"] >= 12]
            else:
                df_top = df_top[df_top["Predicted Prob"] >= 0.9]

            for _, row in df_top.iterrows():
                total_trades += 1
                entry_price = row["Entry Price"]
                exit_price  = row["Exit Price"]

                # Long/Short別にリターン計算
                if position == "Long":
                    pct_return = (exit_price - entry_price) / entry_price * leverage
                else:
                    pct_return = (entry_price - exit_price) / entry_price * leverage

                if pct_return > 0:
                    winning_trades += 1

                # 再投資

                print(pct_return, current_portfolio)
                current_portfolio *= (1 + pct_return)

                company_name = get_company_name(row["Ticker"])

                current_portfolio_rato = winning_trades / total_trades * 100
                portfolio_progression.append({
                    "Simulation Date": sim_date,
                    "Position":        position,
                    "Ticker":          row["Ticker"],
                    "Company Name":    company_name,
                    "Trade Return (%)": pct_return * 100,
                    "Entry Price":     entry_price,
                    "Exit Price":      exit_price,
                    "Stop Loss Price": row["Stop Loss Price"],
                    "Updated Portfolio": current_portfolio,
                    "Win Rate (%)":    winning_trades / total_trades * 100
                })
                print(f"Date: {sim_date}, Pos: {position}, Ticker: {row['Ticker']}, \
WinRate: {winning_trades/total_trades*100:.2f}%, Portfolio: {current_portfolio:.0f}")

        # ここでdeal_term分インデックスを飛ばす
        i += deal_term

    # 通算勝率
    win_rate = winning_trades / total_trades * 100
    print(f"\n通算勝率: {win_rate:.2f}% （全{total_trades}取引中、{winning_trades}取引が利益）\n")

    # 結果保存
    df_portfolio = pd.DataFrame(portfolio_progression)
    filename = os.path.join(
        "./分析",
        f"portfolio_progression_by_deal_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    )
    df_portfolio.to_excel(filename, index=False)
    print(f"資産推移を保存しました: {filename}")
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

# with open('optimized_parameters_tickerwise.json', 'w') as f:
#     json.dump(performance_dict, f, indent=4)
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
# file_path = file_path
# df = pd.read_excel(file_path)
# df = df[["業種", "銘柄コード", "企業名"]]
# tickers = {row["銘柄コード"]: row["企業名"] for _, row in df.iterrows()}
# ticker_symbols = list(tickers.keys())

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
                                       lookback_days=lookback_days,
                                       deal_term=deal_term)
print(f"\n最終的な X_today.shape = {X_today.shape}")
print(f"銘柄数 = {len(tickers_today)}")

# ----------------------------------------
# ⑤ 本日の予測 & 購入リスト生成
# ----------------------------------------
# 今日の特徴量作成
# from feature_module import create_today_features

preds = []
if len(X_today)>0:
    for ticker, vec in zip(tickers_today, X_today):
        if ticker in models_dict:
            sc = scalers_dict[ticker]
            m_long  = models_dict[ticker]["Long"]
            m_short = models_dict[ticker]["Short"]
            x2 = vec.reshape(1,-1)
            p_s = sc.transform(x2)
            pL  = m_long.predict_proba(p_s)[0,1]
            pS  = m_short.predict_proba(p_s)[0,1]
            preds.append({
                "Ticker": ticker,
                "Long Prob": pL,
                "Short Prob": pS
            })
today_df = pd.DataFrame(preds).sort_values(by="Long Prob", ascending=False)
print("\n今日の予測（Long優先ソート）")
print(today_df)


from pandas.tseries.offsets import BusinessDay
# Exit日付
today      = datetime.now()
exit_date  = (today + BusinessDay(deal_term)).date()

# 上位3/5/6リスト作成
def make_purchase_list(df, side, top_n):
    out = []
    for _, r in df.sort_values(f"{side} Prob", ascending=False).head(top_n).iterrows():
        ticker = r["Ticker"]
        price  = float(ticker_data_dict[ticker]['Open'].iloc[-1])
        # shares = int((initial_investment/top_n)//price)
        # sl_amt = stop_loss_threshold*price*shares
        sl_pr  = price*(1-stop_loss_threshold if side=="Long" else 1+stop_loss_threshold)
        out.append({
            "Entry Date": today.strftime("%Y-%m-%d"),
            "Exit Date":  exit_date.strftime("%Y-%m-%d"),
            "Term(Days)": deal_term,
            "Ticker": ticker,
            "Current Price": price,
            # "Shares": shares,
            "Stop Loss Price": sl_pr,
            # "Stop Loss Amt": sl_amt,
            # "Investment": round(price*shares,2),
            f"{side} Prob (%)": round(r[f"{side} Prob"]*100,2)
        })
    return pd.DataFrame(out)

purchase_top3_L = make_purchase_list(today_df, "Long", 3)
purchase_top5_L = make_purchase_list(today_df, "Long", 5)
purchase_top6_L = make_purchase_list(today_df, "Long", 6)
purchase_top3_S = make_purchase_list(today_df, "Short", 3)
purchase_top5_S = make_purchase_list(today_df, "Short", 5)
purchase_top6_S = make_purchase_list(today_df, "Short", 6)



# # 保存
# for df, n in [(purchase_top3_L,3),(purchase_top5_L,5),(purchase_top6_L,6)]:
#     fn = os.path.join(save_folder,
#         f"purchase_{n}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
#     )
#     df.to_excel(fn, index=False)
#     print(f"Saved purchase_top{n} to {fn}")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name_top3_L = f"./分析/過去25日間_銘柄選定結果_15_top3_{timestamp}.xlsx"
file_name_top5_L = f"./分析/過去25日間_銘柄選定結果_15_top5_{timestamp}.xlsx"
file_name_top6_L = f"./分析/過去25日間_銘柄選定結果_15_top6_{timestamp}.xlsx"
purchase_top3_L.to_excel(file_name_top3_L, index=False)
purchase_top5_L.to_excel(file_name_top5_L, index=False)
purchase_top6_L.to_excel(file_name_top6_L, index=False)

file_name_top3_S = f"./分析/過去25日間_銘柄選定結果_15_top3_{timestamp}.xlsx"
file_name_top5_S = f"./分析/過去25日間_銘柄選定結果_15_top5_{timestamp}.xlsx"
file_name_top6_S = f"./分析/過去25日間_銘柄選定結果_15_top6_{timestamp}.xlsx"
purchase_top3_S.to_excel(file_name_top3_S, index=False)
purchase_top5_S.to_excel(file_name_top5_S, index=False)
purchase_top6_S.to_excel(file_name_top6_S, index=False)





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
    # "yukikimura1124@gmail.com"
]


# スクリプトのディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

# ファイルパスをスクリプトのパスに基づいて定義
file_path_top3_l = os.path.join(script_dir, file_name_top3_L)
file_path_top5_l = os.path.join(script_dir, file_name_top5_L)

file_path_top3_s = os.path.join(script_dir, file_name_top3_S)
file_path_top5_s = os.path.join(script_dir, file_name_top5_S)


simulation_file = os.path.join(script_dir, simulation_file_name)
weekday_winrates_output_file_file = os.path.join(script_dir, weekday_winrates_output_file)
ticker_winrates_output_file_file = os.path.join(script_dir, ticker_winrates_output_file)

file_paths = [
    file_path_top3_l,
    file_path_top5_l,
    file_path_top3_s,
    file_path_top5_s,
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

# chart_files = generate_charts_for_top_stocks(
#     purchase_df_top5["Ticker"].tolist(),
#     save_dir="./charts",
#     period="5d",    # ※15分足の場合、取得期間を短くすることが望ましい
#     interval="1h"
# )

try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()  # TLS 暗号化を開始
    server.login(GMAIL_USER, GMAIL_PASSWORD)  # ログイン

    raito = current_portfolio / initial_investment
    profit = current_portfolio - initial_investment




    # --- チャート生成（Long & Short 両対応） ---
    chart_files = []
    # ロング上位5
    chart_files += generate_charts_for_top_stocks(
        purchase_top5_L["Ticker"].tolist(),
        save_dir=os.path.join(script_dir, "charts", "long"),
        period="5d",
        interval="1h"
    )
    # ショート上位5
    chart_files += generate_charts_for_top_stocks(
        purchase_top5_S["Ticker"].tolist(),
        save_dir=os.path.join(script_dir, "charts", "short"),
        period="5d",
        interval="1h"
    )

    # --- HTMLテーブル作成も置き換え ---
    long_table_html  = purchase_top5_L.to_html(index=False, justify="center", border=1)
    short_table_html = purchase_top5_S.to_html(index=False, justify="center", border=1)









    # HTML形式の表を作成
    weekday_table_html = weekday_stats.to_html(index=False, justify="center", border=1)
    ticker_table_html = ticker_stats.to_html(index=False, justify="center", border=1)
    # top5_stocks_html = purchase_df_top5.to_html(index=False, justify="center", border=1) if not purchase_df_top5.empty else ""
    df_portfolio["Win Rate (%)"] = (df_portfolio["Win Rate (%)"]).round(2)
    df_portfolio_progression = df_portfolio.round(2)
    simulation_results_html = df_portfolio_progression.to_html(index=False, justify="center", float_format="{:.2f}".format, border=1, escape=False) if not df_portfolio_progression.empty else ""

    for recipient in recipient_list:
        # メールの作成
        msg = MIMEMultipart("related")
        msg["From"] = GMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = f"指数v2 ショートおすすめ結果 ({current_date}) {int(current_portfolio)}円 {raito:.2f}倍"

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
            <p>現在の投資額: {int(current_portfolio):,} 円</p>
            <p>初期投資額: {int(initial_investment):,} 円</p>
            <p>レバレッジ: {leverage} 倍</p>
            <p>取引期間: {deal_term} 営業日</p>
            <p>総合勝率: {win_rate :.2f} %</p>

            <h3>上位5件のLONG推奨銘柄: 90%以下はダメ:</h3>
            {long_table_html}
            <h3>上位5件のShort推奨銘柄: 90%以下はダメ:</h3>
            {short_table_html}

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

