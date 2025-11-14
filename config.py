# -*- coding: utf-8 -*-
"""
專案全域設定檔

本檔案集中管理專案所需的所有超參數、路徑及固定設定，
以便於模型調校與結果復現。
"""
from pathlib import Path

class GNNConfig:
    """GNN 模型 (GAT) 相關設定"""
    # 尺寸與架構
    INPUT_FEATURES_K = 25
    HIDDEN_DIM = 16
    GAT_HEADS = 4
    GNN_LAYERS = 2
    DROPOUT_RATE = 0.6
    
    # GraphMAE 預訓練
    PRETRAIN_EPOCHS = 40
    PRETRAIN_LR = 0.001
    PRETRAIN_BATCH_SIZE = 2048
    WEIGHT_DECAY = 1e-5
    MASK_RATE = 0.75
    
    # 監督式微調
    FINETUNE_EPOCHS = 200
    FINETUNE_LR = 0.0003 # 學習率排程器優化後的初始學習率
    FINETUNE_BATCH_SIZE = 512
    FINETUNE_PATIENCE = 20
    
    # 損失函數
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0

class XGBConfig:
    """XGBoost 模型相關設定"""
    N_SPLITS = 5
    # 若要啟用遞迴特徵刪除，可設定此值 (例如 20)
    FEATURES_TO_DROP = 0 
    # 若要手動指定預測門檻，可設定此值 (例如 0.35)
    MANUAL_THRESHOLD = None
    PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.01,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'nthread': -1,
        'verbosity': 0
    }

class ProjectConfig:
    """專案全域設定"""
    # 通用設定
    SEED = 42
    PYTHON_VERSION = "3.10" # 或您的 Python 版本, e.g., "3.11"
    
    # 資料分割
    TIME_SPLIT_RATIO = 0.8
    USE_XGB_FEATURE_SELECTION_FOR_GNN = False # 決定 GNN 是否使用 XGB 預先篩選特徵
    
    # 路徑設定
    RAW_DIR = Path("./")
    PROCESSED_DIR = Path("./clean_data")
    RESULT_DIR = Path("./result")
    
    # GNN 模型與特徵的儲存路徑
    PRETRAIN_MODEL_PATH = RESULT_DIR / 'graphmae_pretrained_weights.pt'
    FINETUNE_BEST_MODEL_PATH = RESULT_DIR / 'gat_finetuned_best.pt'
    GNN_DERIVED_FEATURES_PATH = RESULT_DIR / 'gnn_derived_features_for_xgboost.parquet'
    
    # 匯率 (與原始碼一致)
    EXCHANGE_RATES = {
        'TWD': 1.0, 'USD': 30.0, 'JPY': 0.2, 'CNY': 4.3, 'EUR': 33.0, 
        'HKD': 3.9, 'AUD': 20.0, 'GBP': 38.0, 'CAD': 22.0, 'SGD': 22.5, 
        'NZD': 19.0, 'THB': 0.85, 'CHF': 34.0, 'SEK': 2.8, 'ZAR': 1.6, 'MXN': 1.8
    }