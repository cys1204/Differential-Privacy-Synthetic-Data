import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. 定義殘差區塊 (核心進化：防止梯度消失) ---
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.activation(out) 
        out = self.dropout(out)
        return out + residual

# --- 2. 定義完整的 AdvancedMLP 架構 ---
class AdvancedMLP(nn.Module):
    def __init__(self, input_dim, config=None):
        super(AdvancedMLP, self).__init__()
        dropout_rate = config.get('dropout_rate', 0.2)
        self.input_layer = nn.Linear(input_dim, 64)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, dropout_rate),
            ResidualBlock(64, dropout_rate),
        )
        self.output_layer = nn.Linear(64, 1) # 輸出 Logits
        
    def forward(self, x):
        out = torch.nn.functional.leaky_relu(self.input_layer(x), 0.1)
        out = self.res_blocks(out)
        return self.output_layer(out)

class MLPTrainEvaluator:
    def __init__(self,  target_col='income'):
        self.target_col = target_col
        self.best_params = None

    def find_best_params(self, k1_path, param_grid, max_workers):
        print("--- 啟動基準參數搜索 (K=1) ---")
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        results = []
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
            future_to_config = {
                executor.submit(mlp_worker, 1, k1_path, self.target_col, cfg): cfg 
                for cfg in combinations
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                res = future.result()
                if res:
                    res.update(config)
                    results.append(res)

        self.best_params = max(results, key=lambda x: x['AUC'])
        metrics_cols = ['K-Level', 'Accuracy', 'Misclassification', 'Precision', 'Recall', 'AUC']
        best_res = {}
        for col in metrics_cols:
            best_res[col] = self.best_params.get(col, None)
            self.best_params.pop(col, None)
        print(f"搜索完成！最優參數鎖定為: {self.best_params}")
        for metric, value in best_res.items():
            print(f" - {metric}: {value}")
        return self.best_params

    def run_mlp_evaluation(self, data_paths, mlp_config, parallel=True, max_workers=4):
        results = []
        current_params = mlp_config
        first_key = list(data_paths.keys())[0] if data_paths else None
        if first_key is not None and isinstance(mlp_config, dict) and 'lr' in mlp_config and isinstance(mlp_config['lr'], list):
            # 偵測到傳入的是列表，自動執行搜索
            current_params = self.find_best_params(data_paths[first_key], mlp_config, max_workers)
        if parallel:
            print(f"模式：並行運算 (Workers: {max_workers})...")
            import multiprocessing as mp
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
                futures = {
                    executor.submit(
                        mlp_worker,
                        k, 
                        path, 
                        self.target_col, 
                        current_params
                    ): k for k, path in data_paths.items()
                }
                
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        results.append(res)
                        print(f"Done! K={res['K-Level']}")
        else:
            print("模式：序列運算...")
            for k, path in data_paths.items():
                res = mlp_worker(k, path, self.target_col, mlp_config)
                if res:
                    results.append(res)
                    print(f"Done! K={res['K-Level']}")
            
        return {
            'metadata': current_params,
            'data': sorted(results, key=lambda x: x['K-Level'])
        }

# --- 3. 訓練 Worker (並行呼叫的入口) ---
def mlp_worker(k, train_path, target_col, config):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    # 限制執行緒，避免並行時 CPU 爆炸
    torch.set_num_threads(1)
    
    try:
        if not os.path.exists(train_path): return None
        
        # 數據預處理
        df = pd.read_csv(train_path)
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        
        # 深度學習一定要做標準化 (Mean=0, Std=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 轉換為 PyTorch 的張量 (Tensor)
        device = torch.device("cpu") # 並行建議用 CPU 較穩
        X_tensor = torch.from_numpy(X_scaled).to(device)
        y_tensor = torch.from_numpy(y).view(-1, 1).to(device)
        
        # 準備模型與優化器
        model = AdvancedMLP(X_scaled.shape[1], config).to(device)
        # 使用 WithLogits 版本，數值更穩定
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        
        # 訓練循環 (Training Loop)
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = config.get('batch_size', 128)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        epochs = config.get('epochs', 50)
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
        # 評估結果
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            y_probs = torch.sigmoid(outputs).numpy()
            y_pred = (y_probs > 0.5).astype(int)
            
        acc = accuracy_score(y, y_pred)
        return {
            'K-Level': k,
            'Accuracy': acc,
            'Misclassification': 1 - acc,
            'Precision': precision_score(y, y_pred, zero_division=0),
            'Recall': recall_score(y, y_pred, zero_division=0),
            'AUC': roc_auc_score(y, y_probs) # 使用機率值計算 AUC
        }
    except Exception as e:
        print(f"Error at K={k}: {e}")
        return None