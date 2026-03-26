# ============================================================
# evaluation/metrics.py — IR 評估指標 (獨立純函數)
#
# 提供 MRR@k 與 NDCG@k 的計算函數，與 training/lora_trainer.py
# 中使用的版本完全一致，確保訓練監控指標與最終評估指標可比。
#
# ★ 為何獨立為一個模組?
#   1. 供 evaluator.py 呼叫，不需要 import 整個訓練器
#   2. 可在 tests/test_metrics.py 中單獨測試正確性
#   3. 未來若需要更換指標 (如 MAP, Recall@k)，只需修改此檔案
#
# MRR@k (Mean Reciprocal Rank):
#   對排名靠前的相關文件非常敏感，適合衡量「第一個相關結果」的品質。
#   範圍: [0, 1]。若第一名就命中，MRR=1；前 k 名都未命中，MRR=0。
#
# NDCG@k (Normalized Discounted Cumulative Gain):
#   對所有在前 k 名內的相關文件都有貢獻，排名越前貢獻越大 (log 折扣)。
#   範圍: [0, 1]。本實作使用二元相關性 (relevant=1, not relevant=0)。
#   對多個相關文件的 query 比 MRR 更全面。
# ============================================================

import math
from typing import Dict, List, Optional, Set

import numpy as np


def compute_mrr_at_k(
    ranked_doc_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """
    計算單一 query 的 MRR@k。

    MRR = 1 / rank_first_relevant
    若前 k 名內無相關文件，MRR = 0。

    Args:
        ranked_doc_ids: 按相似度降序排列的文件 ID 列表 (最相關在前)
        relevant_ids:   此 query 的相關文件 ID 集合
        k:              評估截斷點

    Returns:
        float: 0.0 到 1.0 之間的 MRR 值
    """
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    ranked_doc_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """
    計算單一 query 的 NDCG@k (Normalized Discounted Cumulative Gain)。

    DCG@k  = Σ_{i=1}^{k} rel_i / log2(i + 1)    (rel_i ∈ {0, 1})
    IDCG@k = 理想情況下的 DCG (所有相關文件排在最前面)
    NDCG@k = DCG@k / IDCG@k

    Args:
        ranked_doc_ids: 按相似度降序排列的文件 ID 列表
        relevant_ids:   此 query 的相關文件 ID 集合
        k:              評估截斷點

    Returns:
        float: 0.0 到 1.0 之間的 NDCG 值
    """
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # IDCG: 假設所有相關文件都排在最前
    # num_relevant = min(實際相關數, k)，因為超過 k 的相關文件不計入
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking_results(
    query_ids: List[str],
    ranked_results: List[List[str]],          # ranked_results[i] = query_ids[i] 的排名文件列表
    relevant_docs: Dict[str, Set[str]],       # {query_id: Set[doc_id]}
    k: int = 10,
) -> Dict[str, float]:
    """
    對批次排名結果計算平均 MRR@k 與 NDCG@k。

    Args:
        query_ids:      query ID 列表
        ranked_results: 對應每個 query 的文件排名列表
                        ranked_results[i] 對應 query_ids[i]
        relevant_docs:  每個 query 的相關文件集合
        k:              評估截斷點

    Returns:
        {
            "mrr@{k}":     float,  # 平均 MRR
            "ndcg@{k}":    float,  # 平均 NDCG
            "num_queries": int,    # query 數量
            "hit_rate@{k}": float, # 前 k 名有命中的 query 比例
        }
    """
    mrr_scores  = []
    ndcg_scores = []
    hit_counts  = []

    for q_id, ranked_docs in zip(query_ids, ranked_results):
        relevant = relevant_docs.get(q_id, set())
        if not relevant:
            continue  # 跳過無 ground truth 的 query

        mrr  = compute_mrr_at_k(ranked_docs, relevant, k)
        ndcg = compute_ndcg_at_k(ranked_docs, relevant, k)
        hit  = int(any(d in relevant for d in ranked_docs[:k]))

        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        hit_counts.append(hit)

    n = len(mrr_scores)
    return {
        f"mrr@{k}":      float(np.mean(mrr_scores))  if n > 0 else 0.0,
        f"ndcg@{k}":     float(np.mean(ndcg_scores)) if n > 0 else 0.0,
        f"hit_rate@{k}": float(np.mean(hit_counts))  if n > 0 else 0.0,
        "num_queries":   n,
    }
