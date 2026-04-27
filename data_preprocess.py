import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from k_anonymity import K_Anonymity

class AdultProcessor:
    def __init__(self, file_path):
        self.columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        self.df = pd.read_csv(file_path, names=self.columns, sep=',\s', engine='python')
        self.label_encoders = {}

    def clean_data(self):
        # 1. 處理缺失值
        self.df.replace('?', np.nan, inplace=True)
        self.df.dropna(inplace=True)
        
        # 2. 移除冗餘或無用特徵
        # education 與 education-num 重複；fnlwgt 是權重，對分類預測意義不大
        self.df.drop(['education', 'fnlwgt'], axis=1, inplace=True)
        
        # 3. 目標值轉換 (Target)
        self.df['income'] = self.df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
        
        return self

    def encode_categorical(self):
        # 定義哪些是類別型 QI 或特徵
        categorical_cols = [
            'workclass', 'marital-status', 'occupation', 
            'relationship', 'race', 'sex', 'native-country'
        ]
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            
        return self.df

# 使用範例
if __name__ == "__main__":
    K_ANONY_DATA_DIR = "anonymized_data"
    processor = AdultProcessor("adult.data")
    df_clean = processor.clean_data().encode_categorical()
    df_clean.to_csv(K_ANONY_DATA_DIR + "adult_cleaned.csv", index=False)
    print("數據預處理完成，已儲存至 adult_cleaned.csv")
    # --- run K-Anonymity ---
    k_engine = K_Anonymity(feature_columns=df_clean.columns[:-1])
    k_list = [1, 2, 4]
    data_paths = k_engine.parallel_process(df_clean, k_list, K_ANONY_DATA_DIR=K_ANONY_DATA_DIR)
    print("K-Anonymity 處理完成。")