import os
import argparse
import pandas as pd
import para  # 匯入參數檔
from k_anonymity import K_Anonymity
from svm_training import SVMTrainEvaluator
from mlp import MLPTrainEvaluator
from data_preprocess import AdultProcessor
from dp_wgan import train_dp_wgan

def main():
    parser = argparse.ArgumentParser(description="K-Anonymity ML Experiment")
    parser.add_argument('--reprocess', action='store_true', help='強制重新執行數據預處理與泛化')
    parser.add_argument('--method', type=str, default='k-anonymity', choices=['k-anonymity', 'dp-wgan'], help='選擇隱私保護方法')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[2.0], help='DP-WGAN 的隱私預算 epsilon 列表 (可輸入多個)')
    args = parser.parse_args()
    if args.reprocess:
        para.K_ANON_CONFIG['FORCE_REPROCESS'] = args.reprocess

    # 1. 數據清洗與預處理
    print(f"--- 處理原始數據 ---")
    os.makedirs(para.K_ANONY_DATA_DIR, exist_ok=True)
    file_path = os.path.join(para.K_ANONY_DATA_DIR, "adult_cleaned.csv")
    if os.path.exists(file_path) and para.K_ANON_CONFIG['FORCE_REPROCESS'] != 2:
        print(f"數據已處理")
        df_clean = pd.read_csv(file_path)
    else:
        processor = AdultProcessor(para.RAW_DATA_PATH)
        df_clean = processor.clean_data().encode_categorical()
        df_clean.to_csv(file_path, index=False)
    
    # 2. 確定 QI 特徵 (優先從 para 讀取，若無則自動提取)
    qi_cols = [c for c in df_clean.columns if c != para.TARGET_COL]

    data_paths = {}
    if args.method == 'k-anonymity':
        # 3. 執行 K-Anonymity
        print(f"--- 執行 K-Anonymity ---")
        k_engine = K_Anonymity(feature_columns=qi_cols, config=para.K_ANON_CONFIG)
        needed_k = []
        for k in para.K_ANON_CONFIG['K_LIST']:
            suffix = "_pca" if para.K_ANON_CONFIG.get('use_pca') else ""
            file_path = os.path.join(para.K_ANONY_DATA_DIR, f"train_k{k}{suffix}.csv")
            if para.K_ANON_CONFIG['FORCE_REPROCESS'] or para.K_ANON_CONFIG['FORCE_REPROCESS'] in [1, 2] or not os.path.exists(file_path):
                needed_k.append(k)
    
        if needed_k:
            print(f"執行並行泛化 (K_List={needed_k})...")
            k_engine.parallel_process(df_clean, needed_k, K_ANONY_DATA_DIR=para.K_ANONY_DATA_DIR)
            
        data_paths = {k: os.path.join(para.K_ANONY_DATA_DIR, f"train_k{k}{suffix}.csv") for k in para.K_ANON_CONFIG['K_LIST']}
        if para.K_ANON_CONFIG.get('use_pca'):
            print("使用 PCA 泛化數據:")
        else:
            print("使用一般泛化數據:")
            
    elif args.method == 'dp-wgan':
        print(f"--- 執行 DP-WGAN 隱私資料生成 ---")
        data_paths = {}
        for eps in args.epsilons:
            dp_wgan_path = os.path.join(para.K_ANONY_DATA_DIR, f"train_dp_wgan_eps{eps}.csv")
            if args.reprocess or not os.path.exists(dp_wgan_path):
                print(f"訓練 DP-WGAN (Epsilon = {eps})...")
                train_dp_wgan(
                    data_path=os.path.join(para.K_ANONY_DATA_DIR, "adult_cleaned.csv"),
                    output_path=dp_wgan_path,
                    epsilon=eps
                )
            data_paths[f'DP-eps{eps}'] = dp_wgan_path
        print(f"使用 DP-WGAN 合成數據 (Epsilons: {args.epsilons}):")


    results = []
    metadata = {}
    if 'SVM' in para.MODELS:
        print('開始 SVM 評估...')
        evaluator = SVMTrainEvaluator(target_col=para.TARGET_COL)
        svm_package = evaluator.run_svm_evaluation(
            data_paths=data_paths,
            svm_params_config=para.PARAM_GRID if getattr(para, 'ENABLE_PRE_SEARCH', False) else para.SVM_PARAMS,
            parallel=para.PARALLEL_SVM,
            max_workers=para.MAX_WORKERS
        )
        # 提取資料並標記模型
        svm_df = pd.DataFrame(svm_package['data'])
        svm_df['Model_Type'] = 'SVM'
        results.append(svm_df)
        metadata['SVM'] = svm_package['metadata']

    # --- MLP 部分 ---
    if 'MLP' in para.MODELS:
        print('開始 MLP 評估...')
        evaluator = MLPTrainEvaluator(target_col=para.TARGET_COL)
        mlp_package = evaluator.run_mlp_evaluation(
            data_paths=data_paths,
            mlp_config=para.MLP_PARAM_GRID if getattr(para, 'ENABLE_PRE_SEARCH', False) else para.MLP_PARAMS,
            parallel=para.PARALLEL_MLP,
            max_workers=para.MAX_WORKERS
        )
        # 提取資料並標記模型
        mlp_df = pd.DataFrame(mlp_package['data'])
        mlp_df['Model_Type'] = 'Advanced_MLP'
        results.append(mlp_df)
        metadata['MLP'] = mlp_package['metadata']

    # 5. 輸出報告
    final_df = pd.concat(results, ignore_index=True)
    
    # 調整欄位順序，讓 Model_Type 排在前面比較好讀
    cols = ['K-Level', 'Model_Type'] + [c for c in final_df.columns if c not in ['K-Level', 'Model_Type']]
    final_df = final_df[cols].sort_values(['K-Level', 'Model_Type'])
    final_df = final_df.round(4)

    # 打印結果
    print("\n" + "="*80)
    print(" 最終實驗對照表 ")
    print("="*80)
    print("實驗參數:")
    for model_name, params in metadata.items():
        print(f"[{model_name} Parameters]:")
        for k, v in params.items():
            print(f"  - {k}: {v}")
    print("="*74)
    print(final_df.to_string(index=False))

    os.makedirs(para.REPORT_DIR, exist_ok=True)
    suffix = "_pca" if para.K_ANON_CONFIG.get('use_pca') else ""
    result_path = os.path.join(para.REPORT_DIR, f"experiment_report{suffix}.csv")
    with open(result_path, mode='w', encoding='utf-8', newline='') as f:
        f.write("# --- Experiment Metadata ---\n")
        for model_name, params in metadata.items():
            f.write(f"# [{model_name} Parameters]:\n")
            for k, v in params.items():
                f.write(f"#   - {k}: {v}\n")
        f.write("# -------------------------------\n")
        final_df.to_csv(f, index=False)
    
    plot_all_metrics(final_df)

import matplotlib.pyplot as plt

def plot_all_metrics(df):
    # 定義要畫的指標
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    suffix = "_pca" if para.K_ANON_CONFIG.get('use_pca') else ""
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for model in df['Model_Type'].unique():
            subset = df[df['Model_Type'] == model]
            # 確保 K-Level 有排序，連線才不會亂掉 (如果包含字串就把它轉字串排)
            subset = subset.sort_values('K-Level', key=lambda col: col.astype(str))
            plt.plot(subset['K-Level'].astype(str), subset[metric], marker='o', label=model)
        
        plt.title(f'Impact of Privacy Method on {metric}', fontsize=14)
        plt.xlabel('Privacy Level (K or Epsilon)', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 存檔名稱範例: Accuracy_pca.png 或 Recall_raw.png
        filename = f"{metric}{suffix}.png"
        result_path = os.path.join(para.REPORT_DIR, filename)
        plt.savefig(result_path, dpi=300)
        plt.close() # 關閉畫布節省記憶體
        print(f"[V] 已生成：{filename}")

# 在 main.py 最後面調用

if __name__ == "__main__":
    main()
    # plot_all_metrics(pd.read_csv('result/experiment_report.csv', comment='#'))
    # plot_all_metrics(pd.read_csv('result/experiment_report_pca.csv', comment='#'))