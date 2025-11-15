
# 警示帳戶預測專案

本專案旨在利用圖神經網路 (GNN) 和 XGBoost 模型，從交易數據中預測潛在的警示帳戶。專案採用兩階段的 Stacking 策略：首先使用 GNN (GATv2) 學習圖結構中的高階互動特徵，再將這些衍生特徵與傳統的表格特徵結合，一同輸入至 XGBoost 模型進行最終預測。


## 專案結構
```
├── Preprocess # 資料前處理相關程式碼
│ ├── feature_engineering.py # 特徵工程函數模組
│ └── README.md # 說明文件
├── Model # 模型相關程式碼
│ ├── gat_model.py # GNN 模型定義與訓練流程
│ ├── xgb_model.py # XGBoost 模型訓練流程
│ └── README.md # 說明文件
├── config.py # 全域設定檔 (超參數、路徑等)
├── main.py # 主執行腳本
├── requirements.txt # Python 套件依賴清單
└── README.md # 專案根目錄說明
```

## 系統需求

-   **Python 版本:** 3.10 (建議使用 3.9 ~ 3.11)
-   **主要套件:** 請參考 `requirements.txt`。

## 設定與安裝

1.  **Clone 專案庫:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <repository_name>
    ```

2.  **建立虛擬環境 (建議):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **安裝依賴套件:**
    ```bash
    pip install -r requirements.txt
    ```
    *注意: PyTorch 和 PyG (torch_geometric) 的安裝可能因您的作業系統和 CUDA 版本而異。如果上述指令失敗，請參考其官方網站的安裝指南。*

4.  **準備資料:**
    請將主辦方提供的三個 CSV 檔案 (`acct_alert.csv`, `acct_predict.csv`, `acct_transaction.csv`) 放置於專案的根目錄下。

## 如何執行

所有流程已整合至 `main.py`。只需執行單一指令即可復現結果：

```bash
python main.py
