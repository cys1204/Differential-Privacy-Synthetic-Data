import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_separate_metrics(csv_paths, output_dir, suffix=""):
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p, comment='#'))
    
    if not dfs:
        print(f"找不到檔案: {csv_paths}")
        return
        
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 分離 K-Anonymity 和 DP-WGAN 的資料
    # 將 K-Level 轉為字串進行過濾
    df['K-Level_str'] = df['K-Level'].astype(str)
    
    k_anon_mask = ~df['K-Level_str'].str.startswith('DP')
    k_anon_df = df[k_anon_mask].copy()
    dp_wgan_df = df[~k_anon_mask].copy()
    
    # 確保數值型別正確，K_anon_df 的 K-Level 應轉為整數排序
    if not k_anon_df.empty:
        k_anon_df['K-Level_num'] = pd.to_numeric(k_anon_df['K-Level'], errors='coerce')
        k_anon_df = k_anon_df.sort_values(['Model_Type', 'K-Level_num'])
        
    if not dp_wgan_df.empty:
        # DP-eps2.0 取出數字部分
        dp_wgan_df['Epsilon'] = dp_wgan_df['K-Level_str'].str.extract(r'DP-eps([\d\.]+)').astype(float)
        dp_wgan_df = dp_wgan_df.sort_values(['Model_Type', 'Epsilon'])

    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    
    # 1. 繪製 K-Anonymity 專屬圖表
    if not k_anon_df.empty:
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            for model in k_anon_df['Model_Type'].unique():
                subset = k_anon_df[k_anon_df['Model_Type'] == model]
                plt.plot(subset['K-Level_str'], subset[metric], marker='o', linewidth=2, markersize=8, label=model)
            
            plt.title(f'K-Anonymity Impact on {metric}', fontsize=14, fontweight='bold')
            plt.xlabel('K Level', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            filename = os.path.join(output_dir, f"K-Anon_{metric}{suffix}.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"已生成 K-Anonymity 專屬圖表: {filename}")

    # 2. 繪製 DP-WGAN 專屬圖表
    if not dp_wgan_df.empty:
        for metric in metrics:
            plt.figure(figsize=(6, 5))  # 稍微調小一點
            for model in dp_wgan_df['Model_Type'].unique():
                subset = dp_wgan_df[dp_wgan_df['Model_Type'] == model]
                if len(subset) == 1:
                    # 只有一個點，畫長條圖並固定寬度與邊界
                    plt.bar([f"Eps={subset['Epsilon'].iloc[0]}"], [subset[metric].iloc[0]], width=0.3, label=model, color='#1f77b4')
                    plt.xlim(-0.5, 0.5)  # 避免長條圖佔滿整個畫面
                else:
                    plt.plot(subset['Epsilon'].astype(str), subset[metric], marker='s', linewidth=2, markersize=8, label=model)
            
            plt.title(f'DP-WGAN Impact on {metric}', fontsize=14, fontweight='bold')
            plt.xlabel('Epsilon ($\epsilon$)', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.ylim(0, 1.1)  # 固定 Y 軸
            if len(dp_wgan_df['Epsilon'].unique()) > 1:
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            filename = os.path.join(output_dir, f"DP-WGAN_{metric}{suffix}.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"已生成 DP-WGAN 專屬圖表: {filename}")

    # 3. 繪製終極對比長條圖 (Baseline vs Best K-Anon vs Best DP-WGAN)
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        models = df['Model_Type'].unique()
        bar_width = 0.25
        index = np.arange(len(models))
        
        # 尋找三種代表性數據
        baseline_vals = []
        k_anon_vals = []
        dp_vals = []
        
        # 抓取要對比的標籤
        best_k = k_anon_df['K-Level_num'].max() if not k_anon_df.empty else "N/A"
        k_label = f"K-Anon (K={best_k})"
        
        best_eps = dp_wgan_df['Epsilon'].max() if not dp_wgan_df.empty else "N/A"
        dp_label = f"DP-WGAN ($\epsilon$={best_eps})"

        for model in models:
            model_df = df[df['Model_Type'] == model]
            
            # Baseline (K=1)
            b_val = model_df[model_df['K-Level_str'] == '1'][metric]
            baseline_vals.append(b_val.iloc[0] if not b_val.empty else 0)
            
            # 最佳 K-Anon (假設以 K 最大者為代表)
            k_val = model_df[model_df['K-Level_str'] == str(int(best_k))][metric] if not k_anon_df.empty else pd.Series(dtype=float)
            k_anon_vals.append(k_val.iloc[0] if not k_val.empty else 0)
            
            # DP-WGAN (假設以 eps=2.0 找，或是已有的)
            d_val = model_df[model_df['K-Level_str'] == f'DP-eps{best_eps}'][metric] if not dp_wgan_df.empty else pd.Series(dtype=float)
            dp_vals.append(d_val.iloc[0] if not d_val.empty else 0)
            
        # 開始畫圖
        plt.bar(index, baseline_vals, bar_width, label='Baseline (Original)', color='#2ca02c', alpha=0.8)
        if not k_anon_df.empty:
            plt.bar(index + bar_width, k_anon_vals, bar_width, label=k_label, color='#ff7f0e', alpha=0.8)
        if not dp_wgan_df.empty:
            plt.bar(index + 2 * bar_width, dp_vals, bar_width, label=dp_label, color='#1f77b4', alpha=0.8)
            
        plt.title(f'Overall Comparison: {metric}', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(index + bar_width, models, fontsize=12)
        plt.ylim(0.0, 1.1)  # 統一 Y 軸範圍 0~1
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"Comparison_{metric}{suffix}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"已生成綜合對比圖表: {filename}")


if __name__ == "__main__":
    output_dir = "result"
    
    # 處理普通版本
    csv_normal_paths = ["experiment_report.csv", "result/experiment_report.csv"]
    print("========== 處理一般資料報告 ==========")
    plot_separate_metrics(csv_normal_paths, output_dir, suffix="")
        
    # 處理 PCA 版本
    csv_pca_paths = ["experiment_report_pca.csv", "result/experiment_report_pca.csv"]
    print("========== 處理 PCA 資料報告 ==========")
    plot_separate_metrics(csv_pca_paths, output_dir, suffix="_pca")
