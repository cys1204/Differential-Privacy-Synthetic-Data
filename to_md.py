import pandas as pd

def process_and_print_markdown(file_path, title):
    df = pd.read_csv(file_path, comment='#')
    
    # 簡化模型名稱與欄位映射
    df['Model'] = df['Model_Type'].map({'Advanced_MLP': 'MLP', 'SVM': 'SVM'})
    df = df.rename(columns={'K-Level': 'K', 'Misclassification': 'Miss'})
    
    # 數值處理
    numeric_cols = ['Accuracy', 'Miss', 'Precision', 'Recall', 'AUC']
    df[numeric_cols] = df[numeric_cols].round(4)
    
    # 重新排序
    final_cols = ['K', 'Model', 'Accuracy', 'Miss', 'Precision', 'Recall', 'AUC']
    df = df[final_cols]
    
    print(f"### {title}")
    # tablefmt="pipe" 是 HackMD 標準格式
    # colalign 定義對齊：第一欄靠右(right)，其餘置中(center)
    align = ("right", "center", "center", "center", "center", "center", "center")
    print(df.to_markdown(index=False, tablefmt="pipe", floatfmt=".4f", colalign=align))
    print("\n")

# 執行
process_and_print_markdown('result/experiment_report.csv', "原始空間匿名化結果 (Raw Mode Table)")
# process_and_print_markdown('result/experiment_report_pca.csv', "PCA 空間匿名化結果 (PCA Mode Table)")