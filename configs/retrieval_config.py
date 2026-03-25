# ============================================================
# configs/retrieval_config.py — 混合檢索設定
# Dense (GPU) + BM25 (CPU) 雙軌檢索與 RRF 融合參數
# ============================================================

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalConfig:
    """
    混合檢索管線的完整設定。
    Dense 與 BM25 各自獨立檢索後，透過 RRF 演算法融合排序。
    """

    # ---- Dense Retriever (GPU) ----
    # 編碼時的 batch size (推論階段，VRAM 壓力遠小於訓練)
    dense_encode_batch_size: int = 64
    # Dense 檢索回傳的候選數量 (取 top_k 的 2 倍，供 RRF 融合使用)
    dense_top_k: int = 50

    # ---- BM25 Retriever (純 CPU) ----
    # BM25 演算法參數 (Okapi BM25)
    # k1: 詞頻飽和度控制，越大表示高頻詞的影響越線性
    bm25_k1: float = 1.5
    # b: 文件長度正規化，0=不正規化，1=完全正規化
    bm25_b: float = 0.75
    # BM25 檢索回傳的候選數量
    bm25_top_k: int = 50

    # ---- jieba 斷詞設定 ----
    # 自定義辭典路徑 (用於補充繁體中文專業術語)
    # 格式: 每行一個詞，可選附詞頻與詞性，例如 "機器學習 5 n"
    jieba_user_dict_path: Optional[str] = None
    # 停用詞檔案路徑 (過濾無意義的高頻詞如「的」「了」「在」)
    stopwords_path: Optional[str] = None

    # ---- RRF (Reciprocal Rank Fusion) ----
    # RRF 公式: score(d) = Σ 1/(k + rank_i(d))
    # k 參數控制排名靠前與靠後結果的分數差距
    # k=60 是原始論文的建議值，較大的 k 讓排名差距縮小 (更平滑)
    rrf_k: int = 60
    # 最終回傳的融合結果數量
    final_top_k: int = 10

    # ---- Hard Negative Mining 設定 ----
    # 挖掘 hard negatives 時，每個 query 取多少個負例
    hard_neg_per_query: int = 5
    # 相似度分數區間: 太高的是 false negative，太低的是 easy negative
    # 僅保留分數在此區間內的「困難但非正確」的段落
    hard_neg_score_min: float = 0.3
    hard_neg_score_max: float = 0.8
    # 挖掘時 embedding 編碼的 batch size
    mining_encode_batch_size: int = 32
    # 相似度矩陣分塊大小 (防止一次計算 NxN 矩陣導致 OOM)
    # 以 1000 為一塊，每次只計算 1000 x N 的子矩陣
    similarity_chunk_size: int = 1000
