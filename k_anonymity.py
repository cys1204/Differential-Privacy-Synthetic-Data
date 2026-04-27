import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed

class K_Anonymity:
    def __init__(self, k=10, feature_columns=None, config=None):
        """
        初始化 Mondrian k-anonymity 處理器
        :param k: 隱私層級 (每個分區至少包含 k 筆資料)
        :param feature_columns: 準識別碼 (Quasi-identifiers)，即需要被泛化的特徵
        :param config: 設定字典，支援 use_pca / pca_components 等選項
        """
        self.k = k
        self.feature_columns = feature_columns
        self.config = config if config is not None else {}
        self.use_pca = self.config.get('use_pca', False)
        self.pca_components = self.config.get('pca_components', 0.95)
        self.partitions = []
        self.scaler = StandardScaler()


    def _get_spans(self, df, partition, feature_columns):
        """計算當前分區中各個特徵的取值範圍(或類別數量)"""
        spans = {}
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                span = df[col].loc[partition].max() - df[col].loc[partition].min()
            else:
                span = len(df[col].loc[partition].unique())
            spans[col] = span
        return spans

    def _split_partition(self, df, partition, feature_columns):
        """遞迴切割分區，直到無法再切出滿足 k 的子分區為止"""
        if len(partition) < 2 * self.k:
            return [partition]
        
        spans = self._get_spans(df, partition, feature_columns)
        
        # 按照跨度從大到小嘗試切割維度
        for col, _ in sorted(spans.items(), key=lambda x: x[1], reverse=True):
            # 取得中位數作為切割點
            vals = df[col].loc[partition].sort_values()
            split_idx = len(vals) // 2
            
            # 建立左、右子分區
            lhs = vals.index[:split_idx]
            rhs = vals.index[split_idx:]
            
            # 檢查是否滿足 k-anonymity 條件
            if len(lhs) >= self.k and len(rhs) >= self.k:
                return (self._split_partition(df, lhs, feature_columns) + 
                        self._split_partition(df, rhs, feature_columns))
        
        return [partition]

    def anonymize(self, df):
        """執行去識別化並回傳泛化後的 DataFrame"""
        if self.k <= 1 and not self.use_pca:
            return df.copy()  # k=1 不需要泛化

        df_work = df.copy()
        current_qi = list(self.feature_columns)
        original_qi = list(self.feature_columns)

        if self.use_pca:
            scaled_data = self.scaler.fit_transform(df[original_qi])
            pca = PCA(n_components=self.pca_components)
            pca_data = pca.fit_transform(scaled_data)
            
            pca_cols = [f'PC{i+1}' for i in range(pca_data.shape[1])]
            df_work = pd.DataFrame(pca_data, columns=pca_cols, index=df.index)
            current_qi = pca_cols # 更新局部變數
        
        # 執行 Mondrian 切割
        self.partitions = self._split_partition(df_work, df_work.index, current_qi)

        # 建立結果容器
        df_anonymized = df_work.copy()

        for partition in self.partitions:
            partition_median = df_work.loc[partition, current_qi].median()
            for col in current_qi:
                df_anonymized.loc[partition, col] = partition_median[col]

        # 補回非特徵欄位
        if self.use_pca:
            pc_values = df_anonymized[pca_cols].values
            restored = self.scaler.inverse_transform(pca.inverse_transform(pc_values))
            df_restored = pd.DataFrame(restored, columns=original_qi, index=df.index)

            other_cols = [c for c in df.columns if c not in original_qi]
            
            for col in other_cols:
                df_restored[col] = df.loc[df_anonymized.index, col]
            
            df_anonymized = df_restored
            df_anonymized[original_qi] = df_anonymized[original_qi].clip(lower=0)
        else:
            other_cols = [c for c in df.columns if c not in self.feature_columns]
            for col in other_cols:
                df_anonymized[col] = df[col]
                    
        return df_anonymized

    def parallel_process(self, df, k_list, K_ANONY_DATA_DIR="data_out"):
        """整合多進程並行處理多個 k 值"""
        os.makedirs(K_ANONY_DATA_DIR, exist_ok=True)
        results = {}
        with ProcessPoolExecutor() as executor:
            future_to_k = {
                executor.submit(k_anonymity_worker, k, df, self.feature_columns, K_ANONY_DATA_DIR, self.config): k 
                for k in k_list
            }
        
        for future in as_completed(future_to_k):
            k = future_to_k[future]
            try:
                k_val, path = future.result()
                results[k_val] = path
                print(f"Done! K={k_val:3d}")
            except Exception as exc:
                print(f"K={k} 產生了異常: {exc}")
        return results

def k_anonymity_worker(k, df, qi, K_ANONY_DATA_DIR, config):
    """多進程用的靜態方法"""
    engine = K_Anonymity(k=k, feature_columns=qi, config=config)
    df_res = engine.anonymize(df)
    suffix = "_pca" if config.get('use_pca') else ""
    path = os.path.join(K_ANONY_DATA_DIR, f"train_k{k}{suffix}.csv")
    df_res.to_csv(path, index=False)
    return k, path
    
# 使用範例:
if __name__ == "__main__":
    # 建立簡單測試數據
    data = pd.DataFrame({
        'Age': [25, 26, 25, 23, 40, 42, 45, 44],
        'Zip': [100, 101, 102, 103, 200, 201, 202, 203],
        'Income': [50, 60, 55, 45, 100, 110, 105, 95]
    })
    
    # 設定 k=2，針對 Age 和 Zip 進行泛化
    # anonymizer = K_Anonymity(k=2, feature_columns=['Age', 'Zip'])
    # result = anonymizer.anonymize(data)
    
    # print("Original Data:")
    # print(data)
    # print("\nAnonymized Data (k=2):")
    # print(result)

    # 並行產生所有泛化數據
    k_engine = K_Anonymity(feature_columns=['Age', 'Zip'])
    k_list = [1, 2, 4]
    data_paths = k_engine.parallel_process(data, k_list, K_ANONY_DATA_DIR="anonymized_data")