# 隱私保護合成資料生成與評估 (Differential Privacy Synthetic Data)

本專案為隱私資安專題實作，旨在研究與比較不同的隱私保護技術（**K-Anonymity** 與 **DP-WGAN**）在生成合成資料時的隱私保護能力，並透過下游機器學習模型（SVM、MLP）評估其資料可用性（Utility）。

---

## 🚀 核心功能與技術

* **K-Anonymity (K-匿名化)**：傳統去識別化技術，透過資料泛化與抑制達到隱私保護。
* **DP-WGAN (Differential Privacy WGAN)**：結合差分隱私（Differential Privacy）的生成對抗網路，在模型訓練過程中加入雜訊，確保產出的合成資料具備嚴格的隱私數學保證。
* **下游任務評估 (Downstream Evaluation)**：使用 **SVM** 與 **MLP** 進行分類任務，藉此比較原始資料、K-Anonymity 資料與 DP-WGAN 資料的效能指標（Accuracy, Precision, Recall, AUC）。

---

## 📁 專案架構

```text
├── data_preprocess.py     # 原始資料預處理
├── k_anonymity.py         # K-Anonymity 演算法實作
├── dp_wgan.py             # 具差分隱私的 WGAN 模型架構與訓練
├── para.py                # 參數設定檔案 (Hyperparameters)
├── main.py                # 主程式 (串接預處理、訓練與評估流程)
├── svm_training.py        # 下游分類任務：支援向量機 (SVM) 評估
├── mlp.py                 # 下游分類任務：多層感知器 (MLP) 評估
├── plot_results.py        # 數據視覺化腳本 (繪製長條圖/折線圖)
├── pyproject.toml         # 專案依賴與環境設定
├── AUC.png                # 實驗結果：AUC 比較圖
├── Accuracy.png           # 實驗結果：準確度比較圖
├── Precision.png          # 實驗結果：精準度比較圖
└── Recall.png             # 實驗結果：召回率比較圖
```
## 🛠️ 快速開始
1. 環境安裝本專案使用 pyproject.toml 管理套件，請確保已安裝相關依賴：
   ```
   pip install -r requirements.txt
   # 或者使用你的環境管理工具安裝  
2. 執行完整流程執行 main.py 將會自動進行資料預處理、生成匿名化與合成資料，並運行下游模型評估：
   ```Bash
   python main.py
3. 繪製實驗圖表
   ```Bash
   python plot_results.py
## 📊 實驗結果與分析
https://www.notion.so/34e02c84822080f997a8e4c6f71a66e6


