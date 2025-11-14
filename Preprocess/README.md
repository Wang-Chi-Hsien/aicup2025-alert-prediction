# Preprocess 資料夾說明

本資料夾存放所有與資料前處理及特徵工程相關的程式碼。

## 檔案說明

-   `__init__.py`: 將此資料夾標記為 Python 模組。
-   `feature_engineering.py`: 核心特徵工程腳本，包含以下主要功能：
    -   `initial_data_cleaning`: 負責讀取原始 CSV 檔案，進行編碼處理和基礎的格式清理，並將結果儲存至 `clean_data/` 資料夾。
    -   `build_gnn_node_features`: 專為 GNN 模型設計的節點特徵生成函式。它會基於訓練集的時間切點，計算每個帳戶的局部統計、近期趨勢與異常行為特徵，以供 GNN 學習。
    -   `create_xgb_feature_set`: 專為 XGBoost 模型設計的表格特徵生成函式。它涵蓋了更廣泛的特徵維度，包括資金流動、網絡結構、時間序列、生命週期分析等，建構一個全面的特徵集。
    