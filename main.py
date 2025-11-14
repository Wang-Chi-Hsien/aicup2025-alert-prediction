# -*- coding: utf-8 -*-
"""
警示帳戶預測專案主執行腳本

本腳本為整個預測流程的進入點 (Entry Point)，依序執行以下三大步驟：
1.  資料前處理：對原始 CSV 檔案進行基礎清理。
2.  GNN 模型流程：
    -   建構 GNN 所需的節點與邊特徵。
    -   執行 GraphMAE 無監督預訓練。
    -   執行監督式微調。
    -   產生並儲存 GNN 衍生特徵 (`gnn_derived_features_for_xgboost.parquet`) 供下游使用。
3.  XGBoost 模型流程：
    -   建構 XGBoost 所需的表格特徵。
    -   讀取 GNN 衍生特徵並整合。
    -   訓練 XGBoost 模型並進行預測。
    -   產生最終的 `submission.csv` 檔案。

執行方式：
    python main.py
"""
import sys
import pandas as pd
import torch
import random
import numpy as np

# 匯入設定檔與各模組
from config import GNNConfig, XGBConfig, ProjectConfig
from Preprocess.feature_engineering import (
    initial_data_cleaning,
    build_gnn_node_features,
    build_graph_data_with_edge_features,
    create_xgb_feature_set
)
from Model.gat_model import (
    run_gnn_pipeline
)
from Model.xgb_model import (
    run_xgb_pipeline
)

def set_seed(seed):
    """設定全域隨機種子以確保結果可復現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """專案主函式"""
    try:
        # 步驟 0: 初始化設定
        set_seed(ProjectConfig.SEED)
        ProjectConfig.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        ProjectConfig.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # 步驟 1: 執行資料前處理
        print("="*60)
        print("步驟 1: 開始執行資料前處理...")
        print("="*60)
        initial_data_cleaning(ProjectConfig.RAW_DIR, ProjectConfig.PROCESSED_DIR)

        # 步驟 2: 執行 GNN 模型流程
        print("\n" + "="*60)
        print("步驟 2: 開始執行 GNN 模型流程...")
        print("="*60)
        run_gnn_pipeline()

        # 步驟 3: 執行 XGBoost 模型流程
        print("\n" + "="*60)
        print("步驟 3: 開始執行 XGBoost 模型流程...")
        print("="*60)
        run_xgb_pipeline()

        print("\n" + "="*60)
        print("✅ 專案所有流程執行完畢！")
        print("="*60)

    except FileNotFoundError as e:
        print(f"\n錯誤: 找不到必要的檔案 '{e.filename}'。", file=sys.stderr)
        print("請確認 `acct_transaction.csv`, `acct_alert.csv`, `acct_predict.csv` 皆位於專案根目錄。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n程式執行時發生未預期的錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()