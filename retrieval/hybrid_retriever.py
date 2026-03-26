# ============================================================
# retrieval/hybrid_retriever.py — RRF 混合排序檢索器
#
# 職責: 整合 DenseRetriever (GPU) 與 BM25Retriever (CPU)，
#       透過 Reciprocal Rank Fusion (RRF) 演算法融合雙軌結果，
#       輸出最終的排名列表。
#
# ★ RRF 演算法原理:
#   RRF 是 Cormack et al. (2009) 提出的無參數融合方法，
#   核心思想: 「多個系統都認為相關的文件，比任一單系統都認為最相關的文件更可信」。
#
#   公式: RRF_Score(d) = Σ_i  1 / (k + Rank_i(d))
#
#   其中:
#   - d: 候選文件
#   - i: 各個子檢索系統 (本專案為 Dense + BM25 兩個系統)
#   - Rank_i(d): 文件 d 在系統 i 的排名 (從 1 開始)
#   - k: 平滑常數 (預設 60，來自原始論文)
#
#   k 參數的直覺:
#   - 小 k (例如 10): 排名靠前的文件獲得巨大加成，拉開差距
#   - 大 k (例如 60): 分數更平滑，排名中段的文件也有較公平的機會
#   - k=60 是原始 TREC 實驗中在多數任務上表現最佳的值
#
#   RRF 的優點:
#   1. 無需訓練: 不需要為 Dense/BM25 分數的比例學習權重
#   2. 分數尺度無關: Dense 的餘弦相似度 [-1, 1] 與 BM25 的 TF-IDF 分數 [0, +∞]
#      具有完全不同的尺度，直接加權平均意義不明；RRF 只使用「排名」，不使用分數
#   3. 實踐有效: 多項 BEIR benchmark 研究顯示 RRF 的表現通常優於或持平於學習式融合
#
#   RRF 的已知局限:
#   - 若兩個系統結果完全不重疊，融合效果接近於各取一半
#   - 若其中一個系統對某個 query 表現極差 (如 BM25 對語意類 query)，
#     其 noise 排名可能干擾整體融合結果
#   - k=60 是語料庫無關的通用值，針對特定領域可能有更佳的 k 值
# ============================================================

from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from configs.retrieval_config import RetrievalConfig
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever


# ============================================================
# 純函數: RRF 融合演算法
# 提取為獨立函數方便測試 (tests/test_metrics.py)
# ============================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) 多路融合排序。

    Args:
        ranked_lists: 多個已排序的結果列表，每個元素為 (doc_id, score)。
                      分數只用於已知排名輸入時的視覺化，融合本身只依賴排名位置。
                      列表中位置靠前 = 排名靠前 (rank=1 表示最佳)
        k: RRF 平滑常數 (原始論文建議 60)
        top_k: 回傳前幾名。None 表示回傳所有有分數的文件。

    Returns:
        [(doc_id, rrf_score), ...] 依 RRF 分數從高到低排序
        rrf_score 的值域約 (0, len(ranked_lists)/k]，值越高表示越相關

    Example:
        >>> dense_results = [("doc_A", 0.9), ("doc_B", 0.8), ("doc_C", 0.5)]
        >>> bm25_results  = [("doc_B", 12.0), ("doc_A", 8.5), ("doc_D", 3.2)]
        >>> rrf_results = reciprocal_rank_fusion([dense_results, bm25_results], k=60)
        # doc_B: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252  (兩系統都認為很相關)
        # doc_A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252  (doc_A 與 doc_B 同分)
        # doc_C: 1/(60+3) = 0.01587                                    (只有 Dense 認為相關)
        # doc_D: 1/(60+3) = 0.01587                                    (只有 BM25 認為相關)
    """
    # 累積每個文件的 RRF 分數
    # key: doc_id, value: 累積的 RRF 分數
    rrf_scores: Dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _original_score) in enumerate(ranked_list, start=1):
            # ★ RRF 核心公式: 1 / (k + rank)
            # rank 從 1 開始 (第一名得 1/(k+1))，確保所有文件都有正分數
            contribution = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

    # 依 RRF 分數從高到低排序
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if top_k is not None:
        sorted_results = sorted_results[:top_k]

    return sorted_results


# ============================================================
# HybridRetriever — 主要對外介面
# ============================================================

class HybridRetriever:
    """
    Dense (GPU) + BM25 (CPU) 混合檢索器，透過 RRF 融合排序。

    這是 Step 5 評估腳本的主要介面。設計原則:
    - 單一入口: retriever.search(query, top_k) 即可取得融合結果
    - 靈活切換: use_dense / use_sparse 參數支援單軌或雙軌模式
    - 乾淨封裝: 呼叫端不需要知道 RRF 的實作細節

    使用範例:
        # 初始化
        hybrid = HybridRetriever(dense_retriever, bm25_retriever, retrieval_cfg)
        hybrid.build_index(corpus)

        # 混合檢索 (預設)
        results = hybrid.search("什麼是深度學習？", top_k=10)

        # 單獨 Dense (供 ablation 比較)
        results = hybrid.search(query, top_k=10, use_sparse=False)

        # 單獨 BM25 (供 ablation 比較)
        results = hybrid.search(query, top_k=10, use_dense=False)
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        retrieval_cfg: RetrievalConfig,
    ):
        """
        Args:
            dense_retriever: 已初始化的 DenseRetriever (含已載入的模型)
            bm25_retriever: 已初始化的 BM25Retriever (含已載入的 tokenizer)
            retrieval_cfg: 檢索設定 (rrf_k、top_k 等)
        """
        self.dense = dense_retriever
        self.sparse = bm25_retriever
        self.cfg = retrieval_cfg

        # 從設定讀取各路的候選數量
        # ★ 各路多取 5× 的候選數量，給 RRF 更多的融合空間
        # 例如: final_top_k=10，則 Dense 和 BM25 各自先取 50 個候選
        self.dense_top_k: int = retrieval_cfg.dense_top_k   # 預設 50
        self.bm25_top_k:  int = retrieval_cfg.bm25_top_k    # 預設 50
        self.rrf_k:       int = retrieval_cfg.rrf_k          # 預設 60
        self.final_top_k: int = retrieval_cfg.final_top_k   # 預設 10

        logger.debug(
            f"HybridRetriever 初始化: "
            f"dense_top_k={self.dense_top_k}, bm25_top_k={self.bm25_top_k}, "
            f"rrf_k={self.rrf_k}, final_top_k={self.final_top_k}"
        )

    # ----------------------------------------------------------------
    # 索引建立
    # ----------------------------------------------------------------

    def build_index(self, corpus: Dict[str, str]) -> None:
        """
        同時為 Dense 和 BM25 建立語料庫索引。

        Args:
            corpus: {doc_id: doc_text} 字典

        ★ 注意: 此函數同時觸發:
          1. DenseRetriever.build_corpus_index() — GPU 批次編碼，約 10-60 秒
          2. BM25Retriever.build_index() — jieba 斷詞 + BM25 建索引，約 10-30 秒
        """
        logger.info(f"開始建立混合檢索索引: {len(corpus)} 份文件")

        # Dense 索引 (GPU 編碼，耗時較長)
        logger.info("[1/2] 建立 Dense 索引...")
        self.dense.build_corpus_index(corpus)

        # BM25 索引 (CPU jieba 斷詞，純 CPU)
        logger.info("[2/2] 建立 BM25 索引...")
        self.sparse.build_index(corpus)

        logger.info("混合檢索索引建立完成 ✓")

    # ----------------------------------------------------------------
    # 核心檢索介面
    # ----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        混合向量檢索，回傳 RRF 融合後的排名列表。

        Args:
            query: 查詢文字 (原始中文，函數內部分別處理 Dense 指令前綴和 BM25 斷詞)
            top_k: 回傳結果數量。None 時使用設定值 (retrieval_cfg.final_top_k)
            use_dense: 是否使用 Dense (向量語意) 路線
            use_sparse: 是否使用 Sparse (BM25 關鍵詞) 路線

        Returns:
            [(doc_id, score), ...] 依分數從高到低排序

            分數含義:
            - use_dense=True, use_sparse=True  → RRF 分數 (範圍約 0~0.033)
            - use_dense=True, use_sparse=False → Dense 餘弦相似度 ([-1, 1])
            - use_dense=False, use_sparse=True → BM25 分數 (非負，無上限)

        ★ 各模式的適用場景:
            混合模式: 多數情況下最佳，兼顧語意理解與精確詞彙匹配
            僅 Dense: 語意類 query (「解釋量子糾纏的原理」)
            僅 BM25:  精確詞彙 query (特定法條號、產品型號)
        """
        if not use_dense and not use_sparse:
            raise ValueError("use_dense 與 use_sparse 不能同時為 False")

        if top_k is None:
            top_k = self.final_top_k

        # ----------------------------------------------------------------
        # 單軌模式: 直接回傳對應系統的結果，不做 RRF
        # ----------------------------------------------------------------
        if use_dense and not use_sparse:
            return self.dense.search(query, top_k=top_k)

        if use_sparse and not use_dense:
            return self.sparse.search(query, top_k=top_k)

        # ----------------------------------------------------------------
        # 雙軌 RRF 模式
        # ----------------------------------------------------------------
        # 各路先取比 final_top_k 更多的候選，給 RRF 融合足夠的材料
        dense_results = self.dense.search(query, top_k=self.dense_top_k)
        sparse_results = self.sparse.search(query, top_k=self.bm25_top_k)

        logger.debug(
            f"[RRF 融合前] Dense: {len(dense_results)} 個候選 | "
            f"BM25: {len(sparse_results)} 個候選"
        )

        # ★ RRF 融合: 使用排名而非原始分數
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=self.rrf_k,
            top_k=top_k,
        )

        logger.debug(f"[RRF 融合後] 最終候選: {len(fused_results)} 個")

        return fused_results

    def search_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> List[List[Tuple[str, float]]]:
        """
        批次查詢介面，供 Step 5 評估腳本大規模迴圈使用。

        Args:
            queries: 查詢文字列表
            top_k: 每個 query 的回傳結果數量
            use_dense: 是否使用 Dense 路線
            use_sparse: 是否使用 Sparse 路線

        Returns:
            每個 query 對應的結果列表 (與輸入 queries 順序一致)

        ★ 效能說明:
            Dense 查詢每次需要一次 GPU forward pass (query 編碼)，
            目前實作為順序查詢。若查詢量 > 1000，可考慮批次編碼所有 query
            後再統一做矩陣乘法 (進一步降低 GPU overhead)。
            對於 Step 5 的評估規模 (通常 < 500 queries)，此實作已足夠。
        """
        if top_k is None:
            top_k = self.final_top_k

        all_results = []
        total = len(queries)

        for i, query in enumerate(queries):
            results = self.search(query, top_k=top_k, use_dense=use_dense, use_sparse=use_sparse)
            all_results.append(results)

            # 每 50 個 query 記錄一次進度
            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.debug(f"批次查詢進度: {i+1}/{total}")

        return all_results

    # ----------------------------------------------------------------
    # LoRA 切換 (委派給 DenseRetriever)
    # ----------------------------------------------------------------

    def enable_lora(self, rebuild_index: bool = True, corpus: Optional[Dict[str, str]] = None) -> None:
        """
        啟用 LoRA Adapter 並可選地重建語料庫索引。

        Args:
            rebuild_index: 是否在切換後自動重建 Dense 索引 (強烈建議 True)
            corpus: rebuild_index=True 時必須提供
        """
        self.dense.enable_lora()

        if rebuild_index:
            if corpus is None:
                logger.warning(
                    "啟用 LoRA 後未重建索引！\n"
                    "請手動呼叫 build_index(corpus) 以更新語料庫 embedding。\n"
                    "使用舊 embedding 進行查詢的結果不可信。"
                )
            else:
                logger.info("LoRA 啟用後重建 Dense 索引...")
                self.dense.build_corpus_index(corpus)

    def disable_lora(self, rebuild_index: bool = True, corpus: Optional[Dict[str, str]] = None) -> None:
        """
        停用 LoRA Adapter 並可選地重建語料庫索引。

        Args:
            rebuild_index: 是否在切換後自動重建 Dense 索引 (強烈建議 True)
            corpus: rebuild_index=True 時必須提供
        """
        self.dense.disable_lora()

        if rebuild_index:
            if corpus is None:
                logger.warning(
                    "停用 LoRA 後未重建索引！請手動呼叫 build_index(corpus)。"
                )
            else:
                logger.info("LoRA 停用後重建 Dense 索引...")
                self.dense.build_corpus_index(corpus)

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"dense={self.dense.model_variant.short_name}, "
            f"rrf_k={self.rrf_k}, "
            f"dense_top_k={self.dense_top_k}, "
            f"bm25_top_k={self.bm25_top_k}, "
            f"final_top_k={self.final_top_k})"
        )
