# -*- coding: utf-8 -*-
"""
資料前處理與特徵工程模組

本模組包含專案所需的所有資料清理與特徵建構函式，
主要分為三大部分：
1.  initial_data_cleaning: 對原始 CSV 進行初步的格式清理。
2.  build_gnn_node_features: 為 GNN 模型建構節點特徵。
3.  create_xgb_feature_set: 為 XGBoost 模型建構表格特徵。
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 修正 3: 新增必要的 import
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

try:
    import networkx as nx
except ImportError:
    nx = None

# 修正 2: 正確地從 config 匯入設定
from config import ProjectConfig


def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    嘗試使用多種編碼讀取 CSV 檔案。
    (此處 docstring 省略)
    """
    encodings_to_try = ["utf-8-sig", "utf-8", "big5hkscs", "big5", "cp950"]
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"使用所有編碼嘗試後，仍無法讀取 {path}。")


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """對 DataFrame 中的字串欄位進行標準化清理。"""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype("string").str.strip()
    return df_clean


def initial_data_cleaning(input_dir: Path, output_dir: Path) -> None:
    """
    讀取原始資料，進行清理後儲存至 processed 資料夾。
    (此處 docstring 省略)
    """
    print("--- 步驟 1.1: 清理原始 CSV 檔案 ---")
    filenames = ["acct_alert.csv", "acct_predict.csv", "acct_transaction.csv"]
    for filename in filenames:
        csv_path = input_dir / filename
        if not csv_path.exists():
            print(f"警告: 找不到檔案 {filename}，已跳過。", file=sys.stderr)
            continue
        df = read_csv_safely(csv_path)
        df_clean = process_dataframe(df)
        df_clean.to_csv(output_dir / filename, index=False, encoding='utf-8')
        print(f"  - 清理完成並儲存: {output_dir / filename}")
    print("所有檔案清理完畢。\n")


def build_gnn_node_features(txns, all_acct_list, cutoff_time):
    """
    為 GNN 構建一個輕量級、高資訊密度的節點特徵集。
    (此處 docstring 省略)
    """
    print(f'--- 正在建構 GNN 專用的局部與動態節點特徵 (截止時間: {cutoff_time.date()}) ---')
    
    FINAL_ACTIVE_DAYS = 7
    features_df = pd.DataFrame(index=pd.Index(all_acct_list, name='acct'))
    txns['week'] = txns['datetime'].dt.isocalendar().week

    # 1. 基礎統計類
    sent_stats = txns.groupby('from_acct')['amount_twd'].agg(['count', 'sum', 'std', 'max']).fillna(0)
    sent_stats.columns = ['sent_count', 'sent_sum', 'sent_std', 'sent_max']
    features_df = features_df.join(sent_stats)

    received_stats = txns.groupby('to_acct')['amount_twd'].agg(['count', 'sum', 'std', 'max']).fillna(0)
    received_stats.columns = ['received_count', 'received_sum', 'received_std', 'received_max']
    features_df = features_df.join(received_stats)

    features_df['out_degree'] = txns.groupby('from_acct')['to_acct'].nunique()
    features_df['in_degree'] = txns.groupby('to_acct')['from_acct'].nunique()

    # 2. 變化趨勢類
    final_days_txns = txns[txns['datetime'] >= (cutoff_time - pd.Timedelta(days=FINAL_ACTIVE_DAYS))]
    features_df['final_7d_sent_sum'] = final_days_txns.groupby('from_acct')['amount_twd'].sum()
    features_df['final_7d_received_sum'] = final_days_txns.groupby('to_acct')['amount_twd'].sum()

    last_day_txns = txns[txns['datetime'] >= (cutoff_time - pd.Timedelta(days=1))]
    last_day_sent_sum = last_day_txns.groupby('from_acct')['amount_twd'].sum()
    last_day_received_sum = last_day_txns.groupby('to_acct')['amount_twd'].sum()
    last_day_total_sum = last_day_sent_sum.add(last_day_received_sum, fill_value=0)
    avg_7d_sum = (features_df['final_7d_sent_sum'].fillna(0) + features_df['final_7d_received_sum'].fillna(0)) / FINAL_ACTIVE_DAYS
    features_df['final_7d_amount_acceleration'] = last_day_total_sum / (avg_7d_sum + 1e-6)

    weekly_sent_sum = txns.groupby(['from_acct', 'week'])['amount_twd'].sum()
    features_df['weekly_sent_sum_std'] = weekly_sent_sum.groupby('from_acct').std()
    weekly_received_sum = txns.groupby(['to_acct', 'week'])['amount_twd'].sum()
    features_df['weekly_received_sum_std'] = weekly_received_sum.groupby('to_acct').std()

    # 3. 異常行為類
    all_txns_for_lifespan = pd.concat([
        txns[['from_acct', 'datetime']].rename(columns={'from_acct': 'acct'}),
        txns[['to_acct', 'datetime']].rename(columns={'to_acct': 'acct'})
    ])
    first_tx = all_txns_for_lifespan.groupby('acct')['datetime'].min()
    features_df['time_since_first_tx_days'] = (cutoff_time - first_tx).dt.days
    
    final_7d_sent_count = final_days_txns.groupby('from_acct').size()
    final_7d_received_count = final_days_txns.groupby('to_acct').size()
    features_df['final_7d_tx_count'] = final_7d_sent_count.add(final_7d_received_count, fill_value=0)
    
    final_7d_out_degree = final_days_txns.groupby('from_acct')['to_acct'].nunique()
    final_7d_in_degree = final_days_txns.groupby('to_acct')['from_acct'].nunique()
    features_df['final_7d_unique_counterparties'] = final_7d_out_degree.add(final_7d_in_degree, fill_value=0)

    # 4. 交易平穩性特徵
    txns_sorted = txns.sort_values(by=['from_acct', 'datetime'])
    time_diffs = txns_sorted.groupby('from_acct')['datetime'].diff().dt.total_seconds()
    time_diffs.index = txns_sorted.index
    
    interval_stats = time_diffs.groupby(txns['from_acct']).agg(['mean', 'std']).fillna(0)
    interval_stats.columns = ['sent_interval_mean_seconds', 'sent_interval_std_seconds']
    features_df = features_df.join(interval_stats)

    # 最終處理
    features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
    print(f"✅ 成功生成 {features_df.shape[1]} 維 GNN 專用節點特徵")
    return features_df


# 修正 1 & 4: 統一函式名稱為 create_xgb_feature_set 並移除重複的空函式
def create_xgb_feature_set(transactions_df, all_acct_list, alerts_df=None):
    """
    為 XGBoost 建立一套全面的、精煉的表格特徵集。
    (此處 docstring 省略)
    """
    print("--- 正在建立 XGBoost 精煉特徵集 (已包含所有新舊特徵)... ---")
    
    # Part 1: 執行交易數據預處理
    print("  - Part 1: 執行交易數據預處理...")
    txns = transactions_df.copy()
    txns['currency'] = txns['currency'].fillna('TWD')
    # 修正 2: 使用 ProjectConfig.EXCHANGE_RATES
    txns['exchange_rate'] = txns['currency'].map(ProjectConfig.EXCHANGE_RATES).fillna(1.0)
    txns['amount_twd'] = txns['amount'] * txns['exchange_rate']
    txns['amount'] = txns['amount_twd']

    if txns['time'].dtype == 'int64' or (txns['time'].astype(str).str.isnumeric().all()):
        txns['time'] = txns['time'].astype(str).str.zfill(6).str.replace(r'(\d{2})(\d{2})(\d{2})', r'\1:\2:\3', regex=True)
    
    elapsed_days_in_seconds = txns['txn_date'] * 24 * 60 * 60
    elapsed_time_in_seconds = pd.to_timedelta(txns['time'], errors='coerce').dt.total_seconds().fillna(0)
    txns['total_seconds'] = elapsed_days_in_seconds + elapsed_time_in_seconds
    txns['hour'] = pd.to_numeric(txns['time'].str.split(':', expand=True)[0], errors='coerce')
    features_df = pd.DataFrame(index=pd.Index(all_acct_list, name='acct'))
    
    # Part 2: 計算核心統計與資金流動特徵
    print("  - Part 2: 計算核心統計與資金流動特徵...")
    agg_funcs = ['count', 'sum', 'std', 'max', 'median']
    from_feats = txns.groupby('from_acct')['amount'].agg(agg_funcs).rename(columns=lambda x: f'sent_{x}')
    from_feats['out_degree'] = txns.groupby('from_acct')['to_acct'].nunique()
    to_feats = txns.groupby('to_acct')['amount'].agg(agg_funcs).rename(columns=lambda x: f'received_{x}')
    to_feats['in_degree'] = txns.groupby('to_acct')['from_acct'].nunique()
    features_df = features_df.join(from_feats, how='left').join(to_feats, how='left').fillna(0)
    features_df['net_flow'] = features_df['received_sum'] - features_df['sent_sum']
    
    # Part 3: 計算關係網絡特徵
    print("  - Part 3: 計算關係網絡特徵...")
    sent_concentration = txns.groupby(['from_acct', 'to_acct'])['amount'].sum().reset_index()
    max_sent_to_partner = sent_concentration.groupby('from_acct')['amount'].max()
    features_df['sent_concentration_ratio'] = max_sent_to_partner / (features_df['sent_sum'] + 1e-6)
    received_concentration = txns.groupby(['to_acct', 'from_acct'])['amount'].sum().reset_index()
    max_received_from_partner = received_concentration.groupby('to_acct')['amount'].max()
    features_df['received_concentration_ratio'] = max_received_from_partner / (features_df['received_sum'] + 1e-6)
    pagerank = {}
    if nx:
        try:
            G = nx.from_pandas_edgelist(txns, 'from_acct', 'to_acct', create_using=nx.DiGraph())
            pagerank = nx.pagerank(G, alpha=0.85)
            features_df['pagerank'] = features_df.index.map(pd.Series(pagerank))
        except Exception as e:
            print(f"    - 計算 PageRank 失敗: {e}，跳過此特徵。")
            features_df['pagerank'] = 0
    else:
        print("    - 未安裝 networkx，跳過 PageRank。")
        features_df['pagerank'] = 0
        
    # Part 4: 計算行為模式與風險標籤特徵
    print("  - Part 4: 計算行為模式與風險標籤特徵...")
    from_tx = txns[['from_acct', 'total_seconds', 'txn_date', 'hour']].rename(columns={'from_acct': 'acct'})
    to_tx = txns[['to_acct', 'total_seconds', 'txn_date', 'hour']].rename(columns={'to_acct': 'acct'})
    all_tx_view = pd.concat([from_tx, to_tx]).dropna(subset=['acct'])
    analysis_end_day = all_tx_view['txn_date'].max()
    first_tx_day = all_tx_view.groupby('acct')['txn_date'].min()
    features_df['time_since_first_tx_days'] = analysis_end_day - features_df.index.map(first_tx_day)
    txns['is_midnight'] = ((txns['hour'] >= 0) & (txns['hour'] < 6)).astype(int)
    features_df['sent_period_midnight_ratio'] = txns.groupby('from_acct')['is_midnight'].mean()
    features_df['received_period_midnight_ratio'] = txns.groupby('to_acct')['is_midnight'].mean()
    Q3 = txns["amount"].quantile(0.75)
    IQR = Q3 - txns["amount"].quantile(0.25)
    upper_bound = Q3 + 1.5 * IQR
    txns['is_large_amt'] = (txns['amount'] > upper_bound).astype(int)
    features_df['sent_is_large_amt_ratio'] = txns.groupby('from_acct')['is_large_amt'].mean()
    features_df['received_is_large_amt_ratio'] = txns.groupby('to_acct')['is_large_amt'].mean()
    
    # Part 5: 計算 '王牌' 特徵
    print("  - Part 5: 計算 '王牌' 特徵...")
    all_tx_view_sorted = all_tx_view.sort_values(by=['acct', 'total_seconds'])
    time_diffs_seconds = all_tx_view_sorted.groupby('acct')['total_seconds'].diff()
    avg_holding_time_minutes = (time_diffs_seconds / 60).groupby(all_tx_view_sorted['acct']).median()
    features_df['avg_fund_holding_time_minutes'] = features_df.index.map(avg_holding_time_minutes)
    lifespan = features_df['time_since_first_tx_days'].replace(0, 1)
    features_df['turnover_to_balance_ratio'] = (features_df['sent_sum'] + features_df['received_sum']) / lifespan

    daily_counts = all_tx_view.groupby(['acct', 'txn_date']).size()
    features_df['max_tx_per_day'] = daily_counts.groupby('acct').max()
    hourly_counts = all_tx_view.groupby(['acct', 'txn_date', 'hour']).size()
    features_df['max_tx_per_hour'] = hourly_counts.groupby('acct').max()
    total_txns = all_tx_view.groupby('acct').size()
    avg_daily_freq = total_txns / lifespan
    features_df['freq_burst_ratio'] = features_df['max_tx_per_day'] / (avg_daily_freq + 1e-6)
    
    # Part 6: 計算非監督式異常分數 (Isolation Forest)
    print("  - Part 6: 計算非監督式異常分數 (Isolation Forest)...")
    try:
        iforest_features = ['sent_sum', 'sent_count', 'received_sum', 'received_count', 'in_degree', 'out_degree', 'pagerank', 'net_flow']
        iforest_features_exist = [f for f in iforest_features if f in features_df.columns]
        iforest_data = features_df[iforest_features_exist].fillna(0)
        if not iforest_data.empty:
            iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
            iso_forest.fit(iforest_data)
            features_df['iforest_anomaly_score'] = iso_forest.decision_function(iforest_data)
    except Exception as e:
        print(f"    - 計算 Isolation Forest 分數失敗: {e}，跳過此特徵。")
    features_df['iforest_anomaly_score'] = features_df.get('iforest_anomaly_score', 0)

    # Part 7: 計算進階時間特徵
    print("  - Part 7: 計算進階特徵 (時間、網絡、生命週期)...")
    last_txn_dates = all_tx_view.groupby('acct')['txn_date'].max().rename('last_txn_date')
    txns_with_last_date = txns.merge(last_txn_dates, left_on='from_acct', right_index=True, how='left').merge(last_txn_dates, left_on='to_acct', right_index=True, how='left', suffixes=('_from', '_to'))
    txns_final_7d_from = txns_with_last_date[txns_with_last_date['txn_date'] > (txns_with_last_date['last_txn_date_from'] - 7)]
    txns_final_7d_to = txns_with_last_date[txns_with_last_date['txn_date'] > (txns_with_last_date['last_txn_date_to'] - 7)]
    txns_final_1d_from = txns_with_last_date[txns_with_last_date['txn_date'] > (txns_with_last_date['last_txn_date_from'] - 1)]
    txns_final_1d_to = txns_with_last_date[txns_with_last_date['txn_date'] > (txns_with_last_date['last_txn_date_to'] - 1)]
    final_7d_sent_sum = txns_final_7d_from.groupby('from_acct')['amount'].sum()
    final_7d_received_sum = txns_final_7d_to.groupby('to_acct')['amount'].sum()
    final_7d_sent_count = txns_final_7d_from.groupby('from_acct').size()
    final_7d_received_count = txns_final_7d_to.groupby('to_acct').size()
    features_df['final_7d_sent_sum'] = final_7d_sent_sum
    features_df['final_7d_received_sum'] = final_7d_received_sum
    features_df['final_7d_sent_count'] = final_7d_sent_count
    features_df['final_7d_received_count'] = final_7d_received_count
    features_df['final_7d_amount_sum'] = final_7d_sent_sum.add(final_7d_received_sum, fill_value=0)
    features_df['final_7d_tx_count'] = final_7d_sent_count.add(final_7d_received_count, fill_value=0)
    features_df['final_7d_unique_counterparties'] = txns_final_7d_from.groupby('from_acct')['to_acct'].nunique().add(txns_final_7d_to.groupby('to_acct')['from_acct'].nunique(), fill_value=0)
    final_1d_amount_sum = txns_final_1d_from.groupby('from_acct')['amount'].sum().add(txns_final_1d_to.groupby('to_acct')['amount'].sum(), fill_value=0)
    daily_avg_final_7d = features_df['final_7d_amount_sum'] / 7
    features_df['final_7d_amount_acceleration'] = final_1d_amount_sum / (daily_avg_final_7d + 1e-6)
    features_df['final_7d_received_sum_ratio'] = final_7d_received_sum / (features_df['received_sum'] + 1e-6)
    features_df['final_7d_received_count_ratio'] = final_7d_received_count / (features_df['received_count'] + 1e-6)
    features_df['sent_dispersion_ratio'] = features_df['sent_std'] / (features_df['sent_sum'] / (features_df['sent_count'] + 1e-6) + 1e-6)
    features_df['received_dispersion_ratio'] = features_df['received_std'] / (features_df['received_sum'] / (features_df['received_count'] + 1e-6) + 1e-6)
    if alerts_df is not None:
        alert_accts = set(alerts_df['acct'])
        txns['from_is_alert'] = txns['from_acct'].isin(alert_accts)
        received_from_alert = txns[txns['from_is_alert']].groupby('to_acct')['amount'].agg(['sum', 'count'])
        features_df['received_from_alert_sum'] = received_from_alert['sum']
        features_df['received_from_alert_count'] = received_from_alert['count']
    acct_features_map = pd.DataFrame({'pagerank': pd.Series(pagerank),'out_degree': from_feats['out_degree']}).reindex(all_acct_list).fillna(0)
    merged_to = txns.merge(acct_features_map, left_on='to_acct', right_index=True, how='left')
    to_counterparty_feats = merged_to.groupby('from_acct')[['out_degree', 'pagerank']].mean().rename(columns=lambda x: f'to_counterparty_avg_{x}')
    features_df = features_df.join(to_counterparty_feats)
    merged_from = txns.merge(acct_features_map, left_on='from_acct', right_index=True, how='left')
    from_counterparty_feats = merged_from.groupby('to_acct')[['pagerank']].mean().rename(columns=lambda x: f'from_counterparty_avg_{x}')
    features_df = features_df.join(from_counterparty_feats)

    # Part 8: 計算進階波動特徵
    txns['week'] = (txns['txn_date'] - 1) // 7
    weekly_sent = txns.groupby(['from_acct', 'week'])['amount'].agg(['sum', 'count']).reset_index()
    weekly_received = txns.groupby(['to_acct', 'week'])['amount'].agg(['sum', 'count']).reset_index()
    sent_volatility = weekly_sent.groupby('from_acct')['sum'].agg(['mean', 'std', 'max']).rename(columns=lambda x: f'weekly_sent_sum_{x}')
    sent_volatility['weekly_sent_burst_ratio'] = sent_volatility['weekly_sent_sum_max'] / (sent_volatility['weekly_sent_sum_mean'] + 1e-6)
    features_df = features_df.join(sent_volatility)
    received_volatility = weekly_received.groupby('to_acct')['sum'].agg(['mean', 'std', 'max']).rename(columns=lambda x: f'weekly_received_sum_{x}')
    received_volatility['weekly_received_burst_ratio'] = received_volatility['weekly_received_sum_max'] / (received_volatility['weekly_received_sum_mean'] + 1e-6)
    features_df = features_df.join(received_volatility)

    # Part 9: 計算進階生命週期特徵
    acct_lifecycles = all_tx_view.groupby('acct')['txn_date'].agg(['min', 'max'])
    acct_lifecycles['midpoint_date'] = acct_lifecycles['min'] + (acct_lifecycles['max'] - acct_lifecycles['min']) / 2
    txns_with_lifecycle = txns.merge(acct_lifecycles, left_on='from_acct', right_index=True, how='left').rename(columns={'midpoint_date': 'midpoint_date_from'})
    txns_with_lifecycle = txns_with_lifecycle.merge(acct_lifecycles[['midpoint_date']], left_on='to_acct', right_index=True, how='left').rename(columns={'midpoint_date': 'midpoint_date_to'})
    sent_last_half = txns_with_lifecycle[txns_with_lifecycle['txn_date'] >= txns_with_lifecycle['midpoint_date_from']]
    received_last_half = txns_with_lifecycle[txns_with_lifecycle['txn_date'] >= txns_with_lifecycle['midpoint_date_to']]
    sent_sum_last_half = sent_last_half.groupby('from_acct')['amount'].sum()
    features_df['lifecycle_sent_sum_ratio_last_half'] = sent_sum_last_half / (features_df['sent_sum'] + 1e-6)
    received_sum_last_half = received_last_half.groupby('to_acct')['amount'].sum()
    features_df['lifecycle_received_sum_ratio_last_half'] = received_sum_last_half / (features_df['received_sum'] + 1e-6)

    # 最終處理
    features_df = features_df.fillna(0)
    print(f"精煉特徵集建立完成，共 {features_df.shape[1]} 個特徵。")
    return features_df


def build_graph_data_with_edge_features(txns, acct_to_idx, scaler_edge=None, all_txns_for_dummies=None):
    """
    從交易紀錄 DataFrame 建立 PyTorch Geometric 的圖資料格式，包含邊特徵。
    (此處 docstring 省略)
    """
    print(f'\n--- 正在為 {len(txns)} 筆交易構建圖資料格式 ---')
    src_mapped = txns['from_acct'].map(acct_to_idx)
    dst_mapped = txns['to_acct'].map(acct_to_idx)
    valid_mask = src_mapped.notna() & dst_mapped.notna()
    valid_txns = txns[valid_mask].copy()
    valid_txns.sort_values('datetime', inplace=True)
    
    src = src_mapped[valid_mask].astype(int)
    dst = dst_mapped[valid_mask].astype(int)
    edge_index = torch.tensor([src.values, dst.values], dtype=torch.long)
    
    # 邊特徵
    amount_feat = valid_txns[['amount_twd']].values
    time_delta_feat = valid_txns.groupby('from_acct')['datetime'].diff().dt.total_seconds().fillna(-1).values.reshape(-1, 1)
    timestamp_feat = (valid_txns['datetime'].astype(np.int64) // 10**9).values.reshape(-1, 1)
    
    # One-hot 編碼
    currency_feat = pd.get_dummies(valid_txns['currency'], prefix='curr')
    channel_feat = pd.get_dummies(valid_txns['channel'], prefix='chan')
    
    # 確保訓練和測試時的欄位一致
    all_currencies = all_txns_for_dummies['currency'].unique()
    all_channels = all_txns_for_dummies['channel'].unique()
    currency_feat = currency_feat.reindex(columns=[f'curr_{c}' for c in all_currencies], fill_value=0)
    channel_feat = channel_feat.reindex(columns=[f'chan_{c}' for c in all_channels], fill_value=0)
    
    edge_features_combined = np.concatenate([amount_feat, time_delta_feat, timestamp_feat, currency_feat.values, channel_feat.values], axis=1)
    
    if scaler_edge is None:
        scaler_edge = StandardScaler()
        edge_attr_scaled = scaler_edge.fit_transform(edge_features_combined)
    else:
        edge_attr_scaled = scaler_edge.transform(edge_features_combined)
        
    edge_attr = torch.tensor(edge_attr_scaled, dtype=torch.float)
    print(f"✅ 成功生成 {edge_attr.shape[1]} 維邊特徵")
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    return data, scaler_edge