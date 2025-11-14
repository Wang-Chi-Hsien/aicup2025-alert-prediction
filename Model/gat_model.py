# -*- coding: utf-8 -*-
"""
GNN 模型 (GAT) 訓練流程模組

本模組封裝了 GNN 模型從資料準備、模型定義、預訓練、微調到
衍生特徵導出的完整流程。
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# 從專案模組中匯入
from config import GNNConfig as CONFIG  # GNN超參數使用 CONFIG 別名，以最小化程式碼修改
from config import ProjectConfig
from Preprocess.feature_engineering import build_gnn_node_features, build_graph_data_with_edge_features


# ==============================================================================
# 📌 模型與類別定義 (來自 gat_best.py)
# ==============================================================================

class GAT_Model(torch.nn.Module):
    """GATv2 模型架構，整合了 GraphMAE 的編碼器-解碼器結構與用於微調的分類器。"""
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads, num_layers=CONFIG.GNN_LAYERS):
        super().__init__()
        self.encoder = nn.ModuleList()
        # Encoder
        self.encoder.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=CONFIG.DROPOUT_RATE, edge_dim=edge_dim))
        for _ in range(num_layers - 1):
             self.encoder.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=CONFIG.DROPOUT_RATE, edge_dim=edge_dim))
        
        encoder_out_dim = hidden_channels * heads
        
        # Decoder (for GraphMAE)
        self.decoder = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim),
            nn.GELU(),
            nn.Linear(encoder_out_dim, in_channels)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels))
        
        # Classifier (for fine-tuning)
        self.classifier = nn.Linear(encoder_out_dim, out_channels)

    def get_embedding(self, x, edge_index, edge_attr):
        """通過 GAT 編碼器獲取節點嵌入。"""
        embedding = x
        for conv in self.encoder:
            embedding = F.gelu(conv(embedding, edge_index, edge_attr))
        return embedding

    def forward(self, x, edge_index, edge_attr):
        """監督式學習的前向傳播，輸出分類 logits。"""
        embedding = self.get_embedding(x, edge_index, edge_attr)
        embedding_for_classifier = F.dropout(embedding, p=CONFIG.DROPOUT_RATE, training=self.training)
        logits = self.classifier(embedding_for_classifier)
        return logits.squeeze(-1)

    def reconstruct(self, x, edge_index, edge_attr):
        """用於計算 gnn_recon_error 的輔助函數。"""
        embedding = self.get_embedding(x, edge_index, edge_attr)
        x_reconstructed = self.decoder(embedding)
        return x_reconstructed

    def pretrain_forward(self, x, edge_index, edge_attr, mask_nodes):
        """GraphMAE 預訓練的前向傳播，計算重建損失。"""
        x_masked = x.clone()
        x_masked[mask_nodes] = self.mask_token
        h = self.get_embedding(x_masked, edge_index, edge_attr)
        h_masked = h[mask_nodes]
        x_recon = self.decoder(h_masked)
        x_original_masked = x[mask_nodes]
        loss = F.cosine_similarity(x_recon, x_original_masked.detach(), dim=1)
        return 1.0 - loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss 實現，用於處理類別不平衡問題。"""
    def __init__(self, alpha=CONFIG.FOCAL_LOSS_ALPHA, gamma=CONFIG.FOCAL_LOSS_GAMMA, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss


# ==============================================================================
# 📌 訓練與評估函式 (來自 gat_best.py)
# ==============================================================================

def pretrain_unsupervised(model, data, device):
    """執行 GraphMAE 風格的無監督預訓練。"""
    print("\n--- 階段一：開始 GraphMAE 風格的無監督預訓練 ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.PRETRAIN_LR, weight_decay=CONFIG.WEIGHT_DECAY)
    
    loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1] * CONFIG.GNN_LAYERS,
                            batch_size=CONFIG.PRETRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    
    for epoch in range(1, CONFIG.PRETRAIN_EPOCHS + 1):
        total_loss = 0
        processed_batches = 0
        for batch in loader:
            batch = batch.to(device)
            if batch.num_nodes == 0: continue
            optimizer.zero_grad()
            
            num_center_nodes = batch.batch_size
            perm = torch.randperm(num_center_nodes, device=device)
            num_mask_nodes = int(CONFIG.MASK_RATE * num_center_nodes)
            mask_nodes_local = perm[:num_mask_nodes]

            loss = model.pretrain_forward(batch.x, batch.edge_index, batch.edge_attr, mask_nodes_local)
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                processed_batches += 1
                
        avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
        if epoch % 5 == 0 or epoch == CONFIG.PRETRAIN_EPOCHS:
            print(f"預訓練 Epoch {epoch:03d}: 平均遮蔽重建損失 (1 - CosineSimilarity): {avg_loss:.6f}")
            
    torch.save(model.state_dict(), ProjectConfig.PRETRAIN_MODEL_PATH)
    print(f"✅ GraphMAE 預訓練模型已儲存至: {ProjectConfig.PRETRAIN_MODEL_PATH}")


def finetune_supervised(model, full_data, masks, device):
    """執行監督式微調，包含學習率排程與早停機制。"""
    print("\n--- 階段二：開始監督式微調 ---")
    train_mask, val_mask = masks
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.FINETUNE_LR, weight_decay=CONFIG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, min_lr=1e-6)
    loss_fn = FocalLoss().to(device)
    
    train_loader = NeighborLoader(full_data, input_nodes=train_mask, num_neighbors=[20, 15], batch_size=CONFIG.FINETUNE_BATCH_SIZE, shuffle=True, num_workers=4)
    eval_loader = NeighborLoader(full_data, input_nodes=val_mask, num_neighbors=[20, 15], batch_size=CONFIG.FINETUNE_BATCH_SIZE * 2, shuffle=False, num_workers=4)
    
    best_val_auc = 0
    patience_counter = 0

    for epoch in range(1, CONFIG.FINETUNE_EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = loss_fn(logits[:batch.batch_size], batch.y[:batch.batch_size].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 修正 2A: 準備真實標籤並傳遞給 evaluate 函式
        y_true_val = full_data.y[val_mask]
        val_auc = evaluate(model, eval_loader, device, y_true_val)
        scheduler.step(val_auc)

        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 5 == 0 or epoch == CONFIG.FINETUNE_EPOCHS or epoch == 1:
            print(f"微調 Epoch {epoch:03d}: Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, LR: {current_lr:.6f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), ProjectConfig.FINETUNE_BEST_MODEL_PATH)
            print(f"🚀 新的最佳驗證集 AUC: {best_val_auc:.4f}！模型已儲存。")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG.FINETUNE_PATIENCE:
                print(f"--- 早停觸發！在 {patience_counter} 個 epochs 內 Val AUC 未提升。---")
                break
    
    print("✅ 微調完成。")
    print(f"--- 正在載入最佳模型 (AUC: {best_val_auc:.4f}) ---")
    model.load_state_dict(torch.load(ProjectConfig.FINETUNE_BEST_MODEL_PATH, map_location=device))


# 修正 1: 修改 evaluate 函式以適應新版 torch_geometric API
@torch.no_grad()
def evaluate(model, loader, device, y_true):
    """在給定的資料集上評估模型 AUC 分數。"""
    model.eval()
    all_preds = []
    
    y_true = y_true.cpu() # 確保 y_true 在 CPU 上

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        all_preds.append(logits[:batch.batch_size].cpu())
    
    all_preds = torch.cat(all_preds, dim=0).sigmoid()
    
    # 檢查 y_true 是否包含多個類別
    if len(y_true.unique()) < 2:
        print("警告: 評估集中只存在一個類別，無法計算 AUC。返回 0.0")
        return 0.0
    
    return roc_auc_score(y_true, all_preds)


@torch.no_grad()
def export_xgboost_features(model, full_data, device, acct_idx_to_acct):
    """從訓練好的 GNN 中生成 Level-2 特徵供 XGBoost 使用。"""
    print("\n--- 步驟 4: 生成用於 XGBoost Stacking 的 GNN 衍生 (Level-2) 特徵 ---")
    model.eval()
    data = full_data.to(device)
    num_nodes = data.num_nodes
    
    print("  - 正在計算 gnn_fraud_prob (GNN 風險機率)...")
    logits = model(data.x, data.edge_index, data.edge_attr)
    gnn_fraud_prob = torch.sigmoid(logits)

    print("  - 正在計算 gnn_recon_error (GraphMAE 重建誤差)...")
    pretrain_model = GAT_Model(in_channels=full_data.num_node_features, hidden_channels=CONFIG.HIDDEN_DIM, 
                               out_channels=1, edge_dim=full_data.num_edge_features, heads=CONFIG.GAT_HEADS).to(device)
    pretrain_model.load_state_dict(torch.load(ProjectConfig.PRETRAIN_MODEL_PATH, map_location=device))
    pretrain_model.eval()
    reconstructed_x = pretrain_model.reconstruct(data.x, data.edge_index, data.edge_attr)
    gnn_recon_error = F.mse_loss(data.x, reconstructed_x, reduction='none').mean(dim=1)

    print("  - 正在計算 gnn_alert_neighbor_risk (警示鄰居風險)...")
    edge_index = data.edge_index
    source_nodes, dest_nodes = edge_index[0], edge_index[1]
    is_alert_node = (data.y == 1)
    is_source_alert = is_alert_node[source_nodes]
    alert_source_nodes = source_nodes[is_source_alert]
    alert_dest_nodes = dest_nodes[is_source_alert]
    risk_from_alert_neighbors = gnn_fraud_prob[alert_source_nodes]
    gnn_alert_neighbor_risk = scatter_mean(risk_from_alert_neighbors, alert_dest_nodes, dim=0, dim_size=num_nodes)
    
    print("  - 正在計算 gnn_flow_risk_imbalance (資金流風險不平衡度)...")
    in_risk = scatter_mean(gnn_fraud_prob[source_nodes], dest_nodes, dim=0, dim_size=num_nodes)
    out_risk = scatter_mean(gnn_fraud_prob[dest_nodes], source_nodes, dim=0, dim_size=num_nodes)
    gnn_flow_risk_imbalance = out_risk - in_risk

    print("  - 正在整合所有衍生特徵...")
    xgb_features_df = pd.DataFrame({
        'gnn_fraud_prob': gnn_fraud_prob.cpu().numpy(),
        'gnn_recon_error': gnn_recon_error.cpu().numpy(),
        'gnn_alert_neighbor_risk': gnn_alert_neighbor_risk.cpu().numpy(),
        'gnn_flow_risk_imbalance': gnn_flow_risk_imbalance.cpu().numpy(),
    })
    xgb_features_df['acct'] = xgb_features_df.index.map(acct_idx_to_acct)
    xgb_features_df = xgb_features_df.set_index('acct')
    
    out_path = ProjectConfig.GNN_DERIVED_FEATURES_PATH
    xgb_features_df.to_parquet(out_path)
    print(f"\n✅ GNN 衍生特徵已成功儲存為 Parquet 格式至: {out_path}")
    return xgb_features_df


# ==============================================================================
# 📌 主流程函式
# ==============================================================================

def run_gnn_pipeline():
    """
    執行完整的 GNN 訓練與特徵導出流程。
    此函式等同於原始 `gat_best.py` 的 `main` 函式，但被封裝以便於
    由主腳本 `main.py` 調用。
    """
    print('--- GNN 流程開始: 載入與準備資料 ---')
    txns_raw = pd.read_csv(ProjectConfig.RAW_DIR / 'acct_transaction.csv')
    alerts = pd.read_csv(ProjectConfig.RAW_DIR / 'acct_alert.csv')
    predict_df = pd.read_csv(ProjectConfig.RAW_DIR / 'acct_predict.csv')

    print(f"\n--- 正在過濾交易資料，僅保留玉山銀行帳戶間 (type=1) 的交易 ---")
    if 'from_acct_type' in txns_raw.columns and 'to_acct_type' in txns_raw.columns:
        original_txns_count = len(txns_raw)
        txns_raw = txns_raw[(txns_raw['from_acct_type'] == 1) & (txns_raw['to_acct_type'] == 1)].copy()
        print(f"✅ 篩選完成。交易筆數從 {original_txns_count} 筆，大幅減少至 {len(txns_raw)} 筆。")
    else:
        print("⚠️ 警告: 'acct_transaction.csv' 中未找到 'from_acct_type' 或 'to_acct_type' 欄位，將使用所有交易資料。")
    
    txns = txns_raw.rename(columns={'txn_amt':'amount', 'currency_type':'currency', 'channel_type':'channel', 'txn_time':'time_str', 'txn_date':'date_days'})
    if txns['time_str'].dtype == 'int64' or (txns['time_str'].astype(str).str.isnumeric().all()):
        txns['time_str'] = txns['time_str'].astype(str).str.zfill(6).str.replace(r'(\d{2})(\d{2})(\d{2})', r'\1:\2:\3', regex=True)
    min_date_days = txns['date_days'].min()
    start_date = pd.to_datetime('2023-01-01')
    days_offset = txns['date_days'] - min_date_days
    txns['datetime'] = start_date + pd.to_timedelta(days_offset, unit='D') + pd.to_timedelta(txns['time_str'], errors='coerce')
    txns['amount_twd'] = txns['amount'] * txns['currency'].map(ProjectConfig.EXCHANGE_RATES).fillna(1.0)
    txns.sort_values(by='datetime', inplace=True)
    
    print(f"\n--- 步驟 1.3: 根據時間 ({ProjectConfig.TIME_SPLIT_RATIO}) 切分數據 ---")
    min_ts_day = txns['date_days'].min()
    max_ts_day = txns['date_days'].max()
    split_point_day = min_ts_day + (max_ts_day - min_ts_day) * ProjectConfig.TIME_SPLIT_RATIO
    train_txns = txns[txns['date_days'] <= split_point_day].copy()
    
    all_accts = list(pd.concat([txns['from_acct'], txns['to_acct'], alerts['acct'], predict_df['acct']]).unique())
    acct_to_idx = {acct: i for i, acct in enumerate(all_accts)}
    print(f"篩選後的總獨立帳戶數量為: {len(all_accts)}")

    max_train_date = train_txns['datetime'].max()
    features_df = build_gnn_node_features(train_txns, all_accts, cutoff_time=max_train_date)
    
    train_alerts = alerts[alerts['event_date'] <= split_point_day].copy().sort_values('event_date')
    
    alert_accts_in_map = train_alerts['acct'][train_alerts['acct'].isin(acct_to_idx)].unique()
    alert_indices = [acct_to_idx[acct] for acct in alert_accts_in_map]
    
    shared_y = torch.zeros(len(all_accts), dtype=torch.long)
    if alert_indices:
        shared_y[torch.tensor(alert_indices)] = 1
    print(f"\n訓練時間段內共有 {shared_y.sum().item()} 個正樣本 (警示帳戶)")
    
    if ProjectConfig.USE_XGB_FEATURE_SELECTION_FOR_GNN:
        print(f"\n--- 正在啟用 XGBoost 進行特徵篩選 (目標 Top {CONFIG.INPUT_FEATURES_K}) ---")
        y_np = shared_y.numpy()
        scale_pos_weight = (y_np == 0).sum() / (y_np == 1).sum() if (y_np == 1).sum() > 0 else 1
        baseline_xgb = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=ProjectConfig.SEED, n_jobs=-1)
        baseline_xgb.fit(features_df, y_np)
        feature_importances = pd.Series(baseline_xgb.feature_importances_, index=features_df.columns)
        top_k_features = feature_importances.sort_values(ascending=False).head(CONFIG.INPUT_FEATURES_K).index.tolist()
        print(f"✅ 已篩選出 Top {len(top_k_features)} 個特徵:")
        features_df_selected = features_df[top_k_features]
    else:
        print("\n--- 已跳過 XGBoost 特徵篩選，將使用所有生成的特徵 ---")
        features_df_selected = features_df

    scaler_node = StandardScaler()
    shared_x = torch.tensor(scaler_node.fit_transform(features_df_selected), dtype=torch.float)
    print(f"最終 GNN 輸入的節點特徵維度: {shared_x.shape[1]}")

    train_data_graph_parts, _ = build_graph_data_with_edge_features(train_txns, acct_to_idx, scaler_edge=None, all_txns_for_dummies=txns)
    full_data = Data(x=shared_x, y=shared_y, **train_data_graph_parts.to_dict())
    
    print("\n--- 正在以「時間序列」劃分有標籤數據的訓練/驗證集 ---")
    train_mask = torch.zeros(full_data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(full_data.num_nodes, dtype=torch.bool)
    
    if not train_alerts.empty:
        # 只處理存在於 acct_to_idx 中的警示帳戶
        known_alerts = train_alerts[train_alerts['acct'].isin(acct_to_idx)].copy()
        known_alerts['idx'] = known_alerts['acct'].map(acct_to_idx)
        
        val_ratio = 0.2
        if len(known_alerts) > 1:
            split_index = int(len(known_alerts) * (1 - val_ratio))
            train_set_alerts = known_alerts.iloc[:split_index]
            val_set_alerts = known_alerts.iloc[split_index:]

            train_pos_indices = torch.tensor(train_set_alerts['idx'].values, dtype=torch.long)
            val_pos_indices = torch.tensor(val_set_alerts['idx'].values, dtype=torch.long)
            
            all_pos_indices = torch.cat([train_pos_indices, val_pos_indices])
            all_neg_indices = torch.where(full_data.y == 0)[0]
            
            # 確保負樣本不包含任何正樣本
            neg_indices_for_sampling = torch.from_numpy(np.setdiff1d(all_neg_indices.numpy(), all_pos_indices.numpy()))
            
            # 為訓練集抽樣負樣本
            train_neg_sample_size = min(len(neg_indices_for_sampling), len(train_pos_indices) * 10)
            sampled_train_neg_indices = neg_indices_for_sampling[torch.randperm(len(neg_indices_for_sampling))[:train_neg_sample_size]]
            
            train_mask[train_pos_indices] = True
            train_mask[sampled_train_neg_indices] = True

            # 從剩餘的負樣本中為驗證集抽樣
            remaining_neg_indices = torch.from_numpy(np.setdiff1d(neg_indices_for_sampling.numpy(), sampled_train_neg_indices.numpy()))
            val_neg_sample_size = min(len(remaining_neg_indices), len(val_pos_indices) * 10)
            sampled_val_neg_indices = remaining_neg_indices[torch.randperm(len(remaining_neg_indices))[:val_neg_sample_size]]
            
            val_mask[val_pos_indices] = True
            val_mask[sampled_val_neg_indices] = True
        else:
             print("警告: 已知的警示帳戶數量不足以劃分訓練/驗證集。")
    else:
        print("警告: 訓練時間段內沒有任何標籤，無法創建監督式微調所需的數據集。")

    print(f"✅ 劃分完成 - 訓練節點: {train_mask.sum()}, 驗證節點: {val_mask.sum()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n將使用裝置: {device}")
    
    full_data = full_data.to(device)
    
    model = GAT_Model(in_channels=full_data.num_node_features, hidden_channels=CONFIG.HIDDEN_DIM, 
                      out_channels=1, edge_dim=full_data.num_edge_features, heads=CONFIG.GAT_HEADS).to(device)
    print(f"\nGNN 模型已初始化:\n{model}")
    
    pretrain_unsupervised(model, full_data, device)
    
    if train_mask.sum() > 0 and val_mask.sum() > 0:
        print(f"\n--- 載入預訓練模型 ({ProjectConfig.PRETRAIN_MODEL_PATH}) 準備微調 ---")
        model.load_state_dict(torch.load(ProjectConfig.PRETRAIN_MODEL_PATH, map_location=device))
        
        masks = (train_mask.to(device), val_mask.to(device))
        finetune_supervised(model, full_data, masks, device)
        
        print("\n--- 最終模型評估 ---")
        final_eval_loader = NeighborLoader(full_data, input_nodes=val_mask.to(device), num_neighbors=[15]*CONFIG.GNN_LAYERS, batch_size=CONFIG.FINETUNE_BATCH_SIZE * 2, shuffle=False, num_workers=0)
        
        # 修正 2B: 準備真實標籤並傳遞給 evaluate 函式
        y_true_final = full_data.y[val_mask.to(device)]
        final_val_auc = evaluate(model, final_eval_loader, device, y_true_final)
        
        print(f"🎉 最終驗證集上的 AUC (使用最佳模型): {final_val_auc:.4f}")
    else:
        print("警告: 由於訓練集或驗證集為空，跳過監督式微調和評估。")
    
    acct_idx_to_acct = {i: acct for acct, i in acct_to_idx.items()}
    export_xgboost_features(model, full_data, device, acct_idx_to_acct)
    
    print("\nGNN 流程執行完畢。")