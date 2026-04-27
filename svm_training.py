import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

class SVMTrainEvaluator:
    def __init__(self, target_col='income'):
        self.target_col = target_col
        self.best_params = None

    def find_best_params(self, path_k1, param_grid):
        """核心功能：針對 K=1 數據尋找最佳超參數組合"""
        print(f"--- 啟動基準參數搜索 (K=1) ---")
        df = pd.read_csv(path_k1)
        qi_cols = [c for c in df.columns if c != self.target_col]
        X, y = df[qi_cols], df[self.target_col]
        X_scaled = StandardScaler().fit_transform(X)

        # 預設開啟 probability=True 方便後續擴展 AUC 計算
        svc = SVC(probability=True, random_state=42)
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(X_scaled, y)
        
        self.best_params = grid.best_params_
        # 確保必要的非搜索參數也在裡面
        self.best_params['probability'] = True
        self.best_params['random_state'] = 42
        
        print(f"搜索完成！最優參數鎖定為: {self.best_params}")
        return self.best_params

    def run_svm_evaluation(self, data_paths, svm_params_config, parallel=True, max_workers=4):
        """
        高階入口：自動判斷是否需要搜索參數，並執行全體 K 值評估
        :param data_paths: {k: path} 的字典
        :param svm_params_config: 可以是固定的參數字典，或是用於搜索的 param_grid
        """
        results = []
        
        # 1. 決定最終使用的參數 (如果有 K=1 且提供的是 Grid 則進行搜索)
        current_params = svm_params_config
        first_key = list(data_paths.keys())[0] if data_paths else None
        if first_key is not None and isinstance(svm_params_config, dict) and 'C' in svm_params_config and isinstance(svm_params_config['C'], list):
            # 偵測到傳入的是列表，自動執行搜索
            current_params = self.find_best_params(data_paths[first_key], svm_params_config)
        
        # 2. 執行訓練
        if parallel:
            print(f"模式：並行運算 (Workers: {max_workers})...")
            import multiprocessing as mp
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
                futures = {
                    executor.submit(svm_worker, k, path, self.target_col, current_params): k 
                    for k, path in data_paths.items()
                }
                for future in as_completed(futures):
                    res = future.result()
                    if res: 
                        results.append(res)
                        print(f"Done! K={res['K-Level']}")
        else:
            print("模式：序列運算...")
            for k, path in data_paths.items():
                res = svm_worker(k, path, self.target_col, current_params)
                if res:
                    results.append(res)
                    print(f"Done! K={res['K-Level']}")
        return {
        'metadata': self.best_params if self.best_params else svm_params_config,
        'data': sorted(results, key=lambda x: x['K-Level'])
    }
    

def svm_worker(k, train_path, target_col, svm_params):
    """
    這是一個獨立的頂層函數，最適合並行運算。
    """
    # 在內部再次 import 確保子進程環境完整
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_auc_score

    try:
        df = pd.read_csv(train_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用 para.py 傳進來的參數
        model = SVC(**svm_params)
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
        acc = accuracy_score(y, y_pred)
        
        res = {
            'K-Level': k,
            'Accuracy': acc,
            'Misclassification': 1 - acc,
            'Precision': precision_score(y, y_pred, zero_division=0),
            'Recall': recall_score(y, y_pred, zero_division=0),
            'AUC': roc_auc_score(y, y_prob)
        }
            
        return res
    except Exception as e:
        return None

if __name__ == "__main__":
    data_dir = "./anonymized_data"
    k_list = [1, 2, 4]
    
    evaluator = SVMTrainEvaluator(target_col='income')
    results = []

    for k in k_list:
        train_csv = os.path.join(data_dir, f"train_k{k}.csv")
        if os.path.exists(train_csv):
            res = evaluator.run_experiment(k, train_csv)
            results.append(res)

    # 輸出結果表格
    report_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("Training Data Sensitivity Analysis (Feature Auto-Detection)")
    print("="*70)
    print(report_df.to_string(index=False))