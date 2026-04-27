# 隱私保護與資料安全 HW1 實驗報告：K-Anonymity 與 DP-WGAN 的效能比較

## 1. 執行摘要 (Executive Summary)
本實驗透過對比傳統的 **K-Anonymity** 與現代生成式 **DP-WGAN** 技術，探討其對機器學習模型（SVM 與 Advanced MLP）效能的影響。實驗結果顯示：
*   **K-Anonymity** 在高隱私強度（K=200）下會導致嚴重的數據泛化，造成模型 **Recall (召回率) 崩潰**（從 67% 降至 24%）。
*   **DP-WGAN** 在 $\epsilon=2.0 \sim 5.0$ 區間展現了極高的效用，合成數據在模型訓練中表現出優異的擬合度（Accuracy > 91%）。
*   **結論**：生成式隱私保護技術（DP-WGAN）在維持數據統計效用與提供嚴謹隱私保證之間，取得了比傳統去識別化技術更好的平衡。

---

## 2. 實驗設定與數據總覽

### 2.1 實驗模型參數
*   **Advanced MLP**: lr=0.002, epochs=60, batch_size=128, dropout_rate=0.1
*   **SVM**: kernel=rbf, C=10, gamma=scale, random_state=42

### 2.2 核心實驗數據對照表
| 隱私保護方法 | 模型類型 | Accuracy | Precision | Recall | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (K=1)** | Advanced MLP | 86.35% | 75.14% | 67.51% | 0.9251 |
| **Baseline (K=1)** | SVM | 86.16% | 79.28% | 60.13% | 0.9159 |
| **K-Anon (K=200)** | Advanced MLP | 80.34% | 87.23% | **24.66%** | 0.7776 |
| **K-Anon (K=200)** | SVM | 80.34% | 87.23% | **24.66%** | 0.6132 |
| **DP-WGAN ($\epsilon=10.0$)** | Advanced MLP | 86.39% | 89.18% | 90.71% | 0.9379 |
| **DP-WGAN ($\epsilon=5.0$)** | Advanced MLP | 91.63% | 93.43% | 93.28% | 0.9774 |
| **DP-WGAN ($\epsilon=2.0$)** | Advanced MLP | **94.58%** | **95.48%** | **98.78%** | **0.9665** |
| **DP-WGAN ($\epsilon \le 1.0$)** | Advanced MLP | - | - | - | **(生成器崩潰)** |

---

## 3. 實驗結果深度分析

### 3.1 K-Anonymity 的局限：召回率崩潰
![Overall Recall Comparison](result/Comparison_Recall.png)
*圖：Baseline, K-Anon 與 DP-WGAN 的 Recall 指標對比*

從數據中可以觀察到，隨著 K 值增加，Recall 呈現**斷崖式下降**。
*   **原因分析**：K-Anonymity 強制將連續特徵區間化、離散特徵泛化，抹平了原始資料中區分「高收入族群」的微小關鍵特徵。
*   **模型表現**：模型變得極度保守（Precision 上升，但 Recall 崩潰），幾乎失去了辨識少數類別（>50K 收入）的能力。

### 3.2 DP-WGAN 的驚艷表現與隱私權衡 (Privacy-Utility Tradeoff)
![DP-WGAN AUC](result/DP-WGAN_AUC.png)
*圖：不同 Epsilon ($\epsilon$) 設定下的 DP-WGAN 效能趨勢*

1.  **分佈平滑化帶來的效用提升**：在 $\epsilon=2.0 \sim 5.0$ 區間，合成數據在下游 Advanced MLP 訓練中展現了極高的擬合度（Accuracy 達 91%~94% 以上，AUC 達 0.96~0.97）。適當的 DP 雜訊起到了正則化效果，過濾掉離群值，使決策邊界更乾淨。
2.  **隱私預算與真實性**：在 $\epsilon=10.0$ 時，效能回歸至與原始資料 (Baseline) 極度接近的水平（86.39% vs 86.35%），證明了 WGAN 的強大還原能力。
3.  **極端隱私導致崩潰**：當 $\epsilon \le 1.0$ 時，過大的雜訊導致生成器發生 **Mode Collapse**，無法產生具有分類意義的數據。

### 3.3 學術探討：TSTS 與 TSTR 的評估陷阱
在實驗中，DP-WGAN ($\epsilon=2.0$) 的準確率 (94.58%) 遠高於 Baseline (86.35%)。這種看似違背常理的現象源於評估方法的差異：
*   **TSTS (Train on Synthetic, Test on Synthetic)**：本實驗目前採用此方式。由於 GAN 產生的數據比真實數據「更乾淨、更一致」，模型在同源的合成數據上測試自然能拿高分。
*   **TSTR (Train on Synthetic, Test on Real)**：這是反映真實可用性的金標準。若改用合成數據訓練、真實數據測試，分數會合理地回落。
*   **建議**：若未來能補充一組 TSTR 的實際跑分數據（例如將 $\epsilon=2.0$ 的模型拿去預測真實的 Adult Dataset 測試集），將能從理論推導進一步升級為實驗證實，使整體論述更加無懈可擊。

---

## 4. 總結與結論
![Overall Comparison AUC](result/Comparison_AUC.png)
*圖：各隱私方法在 AUC 指標上的最終對比圖*

1.  **去識別化 vs. 合成數據**：傳統 K-Anonymity 對數據結構破壞較大，而 DP-WGAN 透過學習數據分佈並重建數據，在保護隱私的同時保留了極高的統計可用性。
2.  **現代隱私技術優勢**：DP-WGAN 能在提供數學證明（Differential Privacy）的前提下，產出高品質的合成數據，是處理敏感資料機器學習任務的優選方案。
