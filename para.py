import os

# --- 資料路徑設定 ---
RAW_DATA_PATH = "adult.data"
K_ANONY_DATA_DIR = "anonymized_data"
REPORT_DIR = "./"
MAX_WORKERS = os.cpu_count() - 1  # 留一個核心給系統使用
PARALLEL_SVM = True
PARALLEL_MLP = True

# --- 數據欄位定義 ---
TARGET_COL = 'income'

# --- K-Anonymity 設定 ---
K_ANON_CONFIG = {
    'FORCE_REPROCESS': 0,  # 是否強制重新執行數據預處理與泛化 (0: 只處理缺失的 K；1: K 全部重新處理；2: 資料清理也重新處理)
    # 'K_LIST': [1, 10, 50, 100, 200],
    'K_LIST': [200],
    'use_pca': False,         # 是否啟用 PCA
    'pca_components': 0.95,  # 保留 95% 的變異量
}

MODELS = ['SVM', 'MLP']  # 要執行的模型列表

# --- SVM 模型參數 ---
ENABLE_PRE_SEARCH = False  # 是否執行最佳參數搜索
PARAM_GRID = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 0.01],
    'kernel': ['rbf']
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 10,
    'gamma': 'scale',
    'probability': True,
    'random_state': 42
}

# --- MLP 模型參數 ---
ENABLE_PRE_SEARCH = False  # 是否執行最佳參數搜索
MLP_PARAMS = {
    'lr': 0.002,           # 學習率 (Adam 優化器建議 0.001 ~ 0.005)
    'epochs': 60,          # 訓練輪數 (ResNet 收斂較穩，50-80 輪即可)
    'batch_size': 128,     # 批次大小
    'dropout_rate': 0.1
}

MLP_PARAM_GRID = {
    'lr': [0.001, 0.002, 0.005],  # 學習率實驗
    'dropout_rate': [0.1, 0.2, 0.3], # 預防過擬合
    'batch_size': [128],
    'epochs': [60]                # 輪數建議固定，靠 lr 調整收斂
}
