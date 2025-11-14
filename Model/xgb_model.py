# -*- coding: utf-8 -*-
"""
XGBoost 模型訓練流程模組

本模組負責 XGBoost 模型的完整流程，包含：
- 載入清理後的資料與 GNN 衍生特徵。
- 建立訓練與測試資料集。
- 執行交叉驗證訓練。
- 錯誤分析與特徵重要性繪圖。
- 產生最終提交檔案。
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import sys
import pickle

# 嘗試匯入 SHAP，如果失敗則設為 None
try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings('ignore')

# 從專案模組中匯入
from config import XGBConfig, ProjectConfig
# 修正：匯入正確名稱的特徵工程函式
from Preprocess.feature_engineering import read_csv_safely, create_xgb_feature_set


# ==============================================================================
# 📌 輔助函式 (來自 xgb_test.py)
# ==============================================================================

def run_simplified_cv(X, y, xgb_params, n_splits=3):
    """執行一個簡化的交叉驗證來快速評估模型性能和特徵重要性。"""
    print(f"    - 正在執行 {n_splits} 折交叉驗證...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=88)
    oof_preds = np.zeros(len(X))
    importances = pd.DataFrame(index=X.columns)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        current_params = xgb_params.copy()
        current_params['scale_pos_weight'] = scale_pos_weight

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params=current_params, dtrain=dtrain, num_boost_round=500,
                          evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)

        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration))
        importances[f'fold_{fold+1}'] = pd.Series(model.get_score(importance_type='gain')).fillna(0)

    thresholds = np.arange(0.01, 0.51, 0.01)
    f1_scores = [f1_score(y, (oof_preds >= t).astype(int)) for t in thresholds]
    best_f1 = np.max(f1_scores) if f1_scores else 0

    importances['mean'] = importances.mean(axis=1)
    importances.sort_values('mean', ascending=False, inplace=True)

    return best_f1, importances


def perform_shap_analysis(model, X_val, feature_cols, result_dir):
    """計算並儲存指定特徵集的 SHAP 值分析圖。"""
    if shap is None:
        print("  - ⚠️ SHAP 套件未安裝，跳過 SHAP 分析。請執行 'pip install shap'。")
        return
    if not feature_cols:
        print("  - 找不到指定的特徵，跳過 SHAP 分析。")
        return

    print(f"  - 正在計算 {len(feature_cols)} 個特徵的 SHAP 值 (使用第一折的模型)...")
    try:
        explainer = shap.TreeExplainer(model)
        sample_size = min(2000, X_val.shape[0])
        X_val_sample = X_val.sample(sample_size, random_state=42) if sample_size < X_val.shape[0] else X_val
        shap_values = explainer.shap_values(X_val_sample)
        
        feature_indices = [X_val.columns.get_loc(col) for col in feature_cols if col in X_val.columns]
        if not feature_indices:
             print("  - SHAP 分析警告：提供的特徵列不在驗證集中。")
             return
             
        shap_values_subset = shap_values[:, feature_indices]
        X_val_sample_subset = X_val_sample.iloc[:, feature_indices]
        
        plt.figure()
        shap.summary_plot(shap_values_subset, X_val_sample_subset, show=False, plot_size=(12, max(8, len(feature_cols)//3)))
        plt.title("SHAP Summary Plot (GNN Features Only)")
        save_path_summary = result_dir / "gnn_shap_summary_plot.png"
        plt.savefig(save_path_summary, bbox_inches='tight')
        plt.close()
        print(f"    - GNN 特徵 SHAP Summary Plot 已儲存至: {save_path_summary}")

    except Exception as e:
        print(f"  - ❌ SHAP 分析過程中發生錯誤: {e}")


def perform_correlation_analysis(X_tabular, X_gnn, threshold=0.7):
    """計算 GNN embeddings 與表格特徵的相關性，並只印出高度相關的特徵對。"""
    if X_gnn.empty or X_tabular.empty:
        print("  - 缺少 GNN 或表格特徵，跳過相關性分析。")
        return

    print(f"  - 正在計算 GNN 與表格特徵的相關係數 (門檻 = {threshold})...")
    try:
        combined_df = pd.concat([X_tabular, X_gnn], axis=1)
        corr_matrix = combined_df.corr().abs()
        cross_corr_matrix = corr_matrix.loc[X_tabular.columns, X_gnn.columns]

        highly_correlated_pairs = cross_corr_matrix[cross_corr_matrix > threshold].stack().reset_index()
        highly_correlated_pairs.columns = ['Tabular_Feature', 'GNN_Feature', 'Correlation']

        if not highly_correlated_pairs.empty:
            print(f"  - 發現 {len(highly_correlated_pairs)} 組高度相關的特徵對 (Corr > {threshold}):")
            highly_correlated_pairs.sort_values(by='Correlation', ascending=False, inplace=True)
            with pd.option_context('display.max_rows', None):
                print(highly_correlated_pairs.to_string(index=False))
        else:
            print(f"  - ✅ 未發現絕對相關係數超過 {threshold} 的特徵對，表示 GNN 提供了較高的獨立資訊。")

    except Exception as e:
        print(f"  - ❌ 相關性分析過程中發生錯誤: {e}")


def perform_error_analysis(X, y_true, y_pred_proba, threshold, result_dir):
    """執行錯誤分析，比較 FN 和 TP 樣本的特徵差異。"""
    print("\n--- 步驟 7: 正在執行錯誤分析... ---")
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred_binary == 0)
    tp_mask = (y_true == 1) & (y_pred_binary == 1)
    fn_accounts = X[fn_mask]
    tp_accounts = X[tp_mask]
    print(f"分析完成。找到 {len(fn_accounts)} 個偽陰性 (FN) 帳戶和 {len(tp_accounts)} 個真正例 (TP) 帳戶。")
    if fn_accounts.empty or tp_accounts.empty:
        print("無法進行 FN vs. TP 分析，因為 FN 或 TP 樣本為空。")
        return
    fn_means = fn_accounts.mean()
    tp_means = tp_accounts.mean()
    comparison_df = pd.DataFrame({'FN_Mean': fn_means, 'TP_Mean': tp_means})
    comparison_df['Ratio (FN/TP)'] = comparison_df['FN_Mean'] / (comparison_df['TP_Mean'] + 1e-9)
    significant_diffs = comparison_df[(comparison_df['Ratio (FN/TP)'] < 0.9) | (comparison_df['Ratio (FN/TP)'] > 1.1)].copy()
    significant_diffs.sort_values('Ratio (FN/TP)', ascending=True, inplace=True)
    print("\n--- 【關鍵洞察】偽陰性 (FN) vs. 真正例 (TP) 特徵模式對比 ---")
    pd.set_option('display.float_format', '{:12.2f}'.format)
    if not significant_diffs.empty:
        print(significant_diffs.head(25))
    else:
        print("在 FN 與 TP 之間未發現顯著特徵差異。")
    pd.reset_option('display.float_format')


def plot_feature_importance(feature_importances, result_dir):
    """生成並儲存特徵重要性圖。"""
    print("\n正在生成特徵重要性圖...")
    result_dir.mkdir(parents=True, exist_ok=True)
    feature_importances['mean'] = feature_importances.mean(axis=1)
    feature_importances.sort_values('mean', ascending=False, inplace=True)
    plt.figure(figsize=(12, 16))
    top_n = min(len(feature_importances), 70)
    plt.barh(feature_importances.index[:top_n], feature_importances['mean'][:top_n])
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.xlabel("Importance (Gain)")
    plt.tight_layout()
    save_path = result_dir / "feature_importance_xgb_advanced.png"
    plt.savefig(save_path)
    plt.close()
    print(f"特徵重要性圖已儲存至: {save_path}")


# ==============================================================================
# 📌 主流程函式
# ==============================================================================

def run_xgb_pipeline():
    """
    執行完整的 XGBoost 訓練與預測流程。
    此函式等同於原始 `xgb_test.py` 的 `train_and_predict` 函式，
    但被封裝以便於由主腳本 `main.py` 調用。
    """
    # --- 步驟 1.5: 載入清理後的資料 ---
    print("--- XGBoost 流程: 載入清理後的資料... ---")
    alerts_df = read_csv_safely(ProjectConfig.PROCESSED_DIR / "acct_alert.csv")
    predict_df = read_csv_safely(ProjectConfig.PROCESSED_DIR / "acct_predict.csv")
    transactions_df = read_csv_safely(ProjectConfig.PROCESSED_DIR / "acct_transaction.csv")
    rename_map = {'txn_time': 'time', 'currency_type': 'currency', 'channel_type': 'channel', 'txn_amt': 'amount'}
    transactions_df.rename(columns=rename_map, inplace=True)
    all_acct_list = pd.unique(transactions_df[['from_acct', 'to_acct']].values.ravel('K'))
    
    # --- 步驟 2: 建立表格特徵 ---
    # 修正：呼叫正確的函式名稱
    features = create_xgb_feature_set(transactions_df, all_acct_list, alerts_df)

    # --- 步驟 2.5: 載入並整合 GNN 特徵 ---
    print("\n--- 步驟 2.5: 載入並整合 GNN 特徵 ---")
    gnn_features_loaded = False
    gnn_feature_names = []
    try:
        gnn_features_path = ProjectConfig.GNN_DERIVED_FEATURES_PATH
        if gnn_features_path.exists():
            print(f"  - 找到 GNN 特徵檔案，正在讀取: {gnn_features_path}")
            gnn_features = pd.read_parquet(gnn_features_path)
            
            if gnn_features.index.name != 'acct':
                if 'acct' in gnn_features.columns:
                     gnn_features = gnn_features.set_index('acct')
                else:
                     gnn_features.index.name = 'acct'

            features = features.join(gnn_features, how='left').fillna(0)
            gnn_features_loaded = True
            gnn_feature_names = gnn_features.columns.tolist()
            print(f"  - ✅ 成功載入並整合 {gnn_features.shape[1]} 個 GNN 特徵。")
            print(f"     - 載入的 GNN 特徵欄位: {gnn_feature_names}")
        else:
            print(f"  - ⚠️ 警告: 在路徑 '{gnn_features_path}' 中找不到 GNN 特徵檔案，將不使用 GNN 特徵。")
    except Exception as e:
        print(f"  - ❌ 錯誤: 載入 GNN 特徵時發生錯誤: {e}", file=sys.stderr)
        
    # --- 步驟 3: 準備訓練與測試資料集 ---
    print("\n--- 步驟 3: 準備訓練與測試資料集 ---")
    labels_df = pd.DataFrame({'acct': alerts_df['acct']}).drop_duplicates()
    labels_df['label'] = 1
    train_data = features.join(labels_df.set_index('acct'), how='left')
    train_data['label'] = train_data['label'].fillna(0).astype(int)
    X = train_data.drop('label', axis=1)
    y = train_data['label']
    X_test = features.reindex(predict_df['acct']).fillna(0)[X.columns]
    print(f"總訓練帳戶數: {len(X)} (其中警示帳戶: {y.sum()})")
    
    tabular_feature_names = [col for col in features.columns if col not in gnn_feature_names]

    # --- 步驟 4: GNN 診斷檢查 (可選) ---
    # (此處省略了原始碼中的診斷部分以簡化，您可以根據需要加回來)
    print("\n--- 跳過 GNN 特徵診斷檢查以加速流程 ---")

    # --- 步驟 5: 交叉驗證訓練 ---
    print(f"\n--- 步驟 5: 使用 {XGBConfig.N_SPLITS} 折交叉驗證訓練最終 XGBoost 模型... ---")
    skf = StratifiedKFold(n_splits=XGBConfig.N_SPLITS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    feature_importances = pd.DataFrame(index=X.columns)
    
    models = [] # 用於儲存 SHAP 分析的模型

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- FOLD {fold+1}/{XGBConfig.N_SPLITS} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        
        current_params = XGBConfig.PARAMS.copy()
        current_params['scale_pos_weight'] = scale_pos_weight
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)
        
        model = xgb.train(params=current_params, dtrain=dtrain, num_boost_round=2000,
                          evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=500)
        
        if fold == 0:
            models.append(model)
        
        best_iter = model.best_iteration
        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, best_iter))
        test_preds += model.predict(dtest, iteration_range=(0, best_iter)) / XGBConfig.N_SPLITS
        feature_importances[f'fold_{fold+1}'] = pd.Series(model.get_score(importance_type='gain')).fillna(0)
    
    # # 在 CV 結束後執行 SHAP 分析
    # if gnn_features_loaded and models:
    #     print("\n--- GNN 診斷 (SHAP Analysis) ---")
    #     # 使用第一折的驗證集進行分析
    #     _, val_idx_fold0 = next(iter(skf.split(X, y)))
    #     X_val_fold0 = X.iloc[val_idx_fold0]
    #     perform_shap_analysis(models[0], X_val_fold0, gnn_feature_names, ProjectConfig.RESULT_DIR)

    # --- 步驟 6: 決定最終門檻 ---
    print("\n--- 步驟 6: 決定最終門檻 ---")
    if XGBConfig.MANUAL_THRESHOLD is not None:
        best_threshold = XGBConfig.MANUAL_THRESHOLD
        print(f"📌 使用手動設定的門檻: {best_threshold:.4f}")
    else:
        thresholds = np.arange(0.01, 0.51, 0.01)
        scores = [f1_score(y, (oof_preds >= t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(scores)] if scores else 0.5
        print(f"  - 在 {XGBConfig.N_SPLITS} 折交叉驗證上找到最佳門檻: {best_threshold:.2f}")

    f1_val = f1_score(y, (oof_preds >= best_threshold).astype(int))
    precision_val = precision_score(y, (oof_preds >= best_threshold).astype(int))
    recall_val = recall_score(y, (oof_preds >= best_threshold).astype(int))
    print(f"  - 最終 OOF 評估指標: F1={f1_val:.4f}, Precision={precision_val:.4f}, Recall={recall_val:.4f}")

    # --- 步驟 7 & 8: 分析與儲存結果 ---
    perform_error_analysis(X, y, oof_preds, best_threshold, ProjectConfig.RESULT_DIR)
    plot_feature_importance(feature_importances.fillna(0), ProjectConfig.RESULT_DIR)
    
    print("\n--- 步驟 8: 使用最終門檻生成 submission.csv ---")
    predictions = (test_preds >= best_threshold).astype(int)
    submission_df = pd.DataFrame({'acct': predict_df['acct'], 'label': predictions})
    submission_path = ProjectConfig.RESULT_DIR / "submission_xgboost_advanced_feats_with_gnn.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ Submission 檔案已輸出至: {submission_path}")
    print("最終預測標籤分佈:\n", submission_df['label'].value_counts())

    # --- 步驟 9: 儲存預測機率以供融合 ---
    print("\n--- 步驟 9: 儲存預測機率以供融合 (Ensemble) ---")
    val_probs_series = pd.Series(oof_preds, index=X.index)
    test_probs_series = pd.Series(test_preds, index=X_test.index)

    model_name = "xgboost_gnn" 
    val_prob_path = ProjectConfig.RESULT_DIR / f'{model_name}_val_probs.pkl'
    test_prob_path = ProjectConfig.RESULT_DIR / f'{model_name}_test_probs.pkl'

    with open(val_prob_path, 'wb') as f:
        pickle.dump(val_probs_series, f)
    print(f"OOF (Validation) 預測機率已儲存至: {val_prob_path}")

    with open(test_prob_path, 'wb') as f:
        pickle.dump(test_probs_series, f)
    print(f"Test 預測機率已儲存至: {test_prob_path}")
    
    print("\nXGBoost 流程執行完畢。")