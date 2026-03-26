# ============================================================
# training/loss_functions.py — 損失函數
#
# 實作 MultipleNegativesRankingLoss (MNRL)，這是雙編碼器 (Bi-Encoder)
# 檢索模型訓練中最主流的對比學習損失函數。
#
# MNRL 的工作原理:
#   給定 B 個 (query, positive) 配對組成的 mini-batch，
#   每個 query 以其對應的 positive 為正例，
#   以 batch 內其他所有 positive (以及可選的 hard_negative) 為負例。
#   等效於在 B (或 2B) 個選項中做多分類。
#
# ★ 8GB VRAM 的根本限制:
#   MNRL 的品質強依賴 in-batch negatives 的數量。
#   大型訓練 (如原始 BGE) 使用 effective batch=256，即每個 query 有 255 個負例。
#   本專案 batch_size=4~8 意味著每個 query 只有 3~7 個 in-batch negatives。
#   透過加入 hard_negatives 可以補償部分效能差距，但無法完全彌補。
#   在評估結果時應牢記此硬體妥協。
#
# 溫度參數 (temperature) 說明:
#   - temperature 控制 softmax 的「尖銳程度」
#   - 小 temperature (如 0.01) → 分布更尖銳 → 對排名錯誤的懲罰更大
#   - BGE 論文使用 temperature=0.01
#   - sentence-transformers 使用 scale=20 (等效 temperature=0.05)
#   - 本實作預設 temperature=0.05，在穩定性與區分能力間取平衡
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Optional, Tuple


class MultipleNegativesRankingLoss(nn.Module):
    """
    MultipleNegativesRankingLoss (MNRL) — 支援 Hard Negative。

    輸入: L2 正規化後的 embedding 向量
          (確保 dot product = cosine similarity，值域在 [-1, 1])

    損失計算流程:
    1. 拼接 positive_embs 與 hard_negative_embs → all_doc_embs (2B, D)
    2. 計算 query_embs 與 all_doc_embs 的相似度矩陣 (B, 2B)
    3. 除以 temperature 做縮放
    4. 以 labels = [0, 1, ..., B-1] 做 cross-entropy
       (query i 的正確答案是 index i，即 positive_embs[i])

    為何 cross-entropy 而非 hinge loss:
    - Cross-entropy 考慮所有負例的相對排名，梯度訊號更豐富
    - Hinge loss 只考慮最難的負例，在負例數量少時效果有限
    """

    def __init__(self, temperature: float = 0.05):
        """
        Args:
            temperature: 相似度縮放溫度。
                         - 值越小，分布越尖銳，對排名錯誤懲罰越重
                         - 建議範圍: [0.01, 0.1]
                         - 預設 0.05 在穩定性與效果之間取平衡
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embs: torch.Tensor,          # (B, D): 已 L2 正規化的 query embeddings
        positive_embs: torch.Tensor,       # (B, D): 已 L2 正規化的 positive embeddings
        negative_embs: Optional[torch.Tensor] = None,  # (B, D): hard negatives (可選)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算 MNRL 損失。

        Args:
            query_embs: (B, D) L2 正規化的 query embedding 矩陣
            positive_embs: (B, D) L2 正規化的 positive embedding 矩陣
            negative_embs: (B, D) L2 正規化的 hard negative embedding 矩陣 (可選)

        Returns:
            (loss, accuracy): loss 為 MNRL 損失值，accuracy 為訓練 batch 準確率
                              accuracy 為輔助監控指標，不參與反向傳播
        """
        batch_size = query_embs.size(0)

        # ================================================================
        # Step 1: 建構文件池 (Document Pool)
        # ================================================================
        if negative_embs is not None:
            # 有 hard negatives: 文件池 = [positives | hard_negatives]
            # 大小: (2B, D)
            # 配置說明:
            # - index 0 ~ B-1: positives (每個 query 的正確答案)
            # - index B ~ 2B-1: hard negatives (顯式困難負例)
            # - 所有 query i 的「他人 positive」同時充當隱式 in-batch negatives
            all_doc_embs = torch.cat([positive_embs, negative_embs], dim=0)
        else:
            # 無 hard negatives: 文件池 = 只有 positives
            # 大小: (B, D)
            all_doc_embs = positive_embs

        # ================================================================
        # Step 2: 計算相似度矩陣
        # ================================================================
        # 由於輸入已 L2 正規化，dot product = cosine similarity
        # 相似度矩陣 shape: (B, 2B) 或 (B, B)
        # similarities[i][j] = cosine_sim(query_i, doc_j)
        similarities = torch.mm(query_embs, all_doc_embs.T)

        # ================================================================
        # Step 3: 溫度縮放
        # ================================================================
        # 除以 temperature 等效於乘以 scale (scale = 1 / temperature)
        # 縮放目的: 使 softmax 分布更尖銳，讓模型更清楚地區分正負例
        # fp16 注意: temperature=0.05 時，similarities 最大值除以 0.05 = 20.0
        # 這個值在 fp16 範圍內 (max fp16 ≈ 65504)，安全。
        scaled_similarities = similarities / self.temperature

        # ================================================================
        # Step 4: 建立 labels (每個 query 的正確答案是 index i)
        # ================================================================
        # query 0 → positive 0 (index 0)
        # query 1 → positive 1 (index 1)
        # ...
        # query B-1 → positive B-1 (index B-1)
        labels = torch.arange(batch_size, device=query_embs.device, dtype=torch.long)

        # ================================================================
        # Step 5: Cross-Entropy Loss
        # ================================================================
        # F.cross_entropy 內部包含 softmax + log + NLLLoss
        # 等效於: -log(exp(sim[i][i]/t) / Σ_j exp(sim[i][j]/t))
        loss = F.cross_entropy(scaled_similarities, labels)

        # ================================================================
        # 輔助: 計算 batch 準確率 (不參與反向傳播)
        # ================================================================
        # 準確率反映「query 的最高相似度文件是否是其 positive」
        # 訓練初期 accuracy ≈ 1/batch_size (隨機猜測)
        # 訓練後期理想值 → 1.0
        with torch.no_grad():
            predictions = scaled_similarities.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

        return loss, accuracy

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


class SymmetricMNRLoss(nn.Module):
    """
    對稱版 MNRL: 同時計算 query→doc 和 doc→query 兩個方向的損失。

    對稱損失的優點:
    - 雙向梯度讓 positive embedding 也得到更新
    - 在 Bi-Encoder 訓練中有助於提升 recall
    - 代價: 計算量增加約 1.5 倍 (但在 VRAM 和時間上影響不大)

    使用時機:
    - 若 train 資料量較少，對稱損失有助於充分利用每個樣本
    - 若計算時間是瓶頸，可改用單向的 MultipleNegativesRankingLoss

    ★ 本專案預設使用單向 MNRL (上面的類別)，
      此類別保留作為進階選項，
      可透過 run_training.py 的 --symmetric 旗標啟用。
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.mnrl = MultipleNegativesRankingLoss(temperature)

    def forward(
        self,
        query_embs: torch.Tensor,
        positive_embs: torch.Tensor,
        negative_embs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 方向 1: query → positive (標準 MNRL)
        loss_q2p, acc_q2p = self.mnrl(query_embs, positive_embs, negative_embs)

        # 方向 2: positive → query (doc 找 query)
        # 注意: 此方向不加入 hard_negatives，避免梯度訊號過於複雜
        loss_p2q, acc_p2q = self.mnrl(positive_embs, query_embs)

        # 對稱平均
        total_loss = (loss_q2p + loss_p2q) / 2
        avg_accuracy = (acc_q2p + acc_p2q) / 2

        return total_loss, avg_accuracy
