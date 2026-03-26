# ============================================================
# evaluation/evaluator.py — 四象限消融實驗評估器
#
# 負責協調「五種檢索條件」的自動化評估，依序完成:
#   A. bge-small Baseline (LoRA Off)
#   B. bge-small + LoRA   (LoRA On)
#   C. bge-base Baseline  (LoRA Off)
#   D. bge-base + LoRA    (LoRA On)
#   E. 最佳 LoRA 模型 + Hybrid RRF (Dense + BM25)
#
# ★ VRAM 管理策略:
#   five conditions 不能同時載入，必須按「小 → 大 → 釋放」的順序:
#
#   ┌─────────────────────────────────────────────────────────┐
#   │ 1. 載入 small model → 評估 A, B → 釋放                  │
#   │ 2. 載入 base model  → 評估 C, D → 保留 (E 要用)         │
#   │ 3. 建立 BM25 索引 (CPU，不佔 VRAM) → 評估 E → 釋放      │
#   └─────────────────────────────────────────────────────────┘
#
#   若 small 在 B 的表現優於 base 在 D 的表現，才改用 small 做 E。
#   但為了避免重新載入，E 預設使用 base (最後釋放的模型)。
#   使用者可透過 best_model_variant 參數覆寫此行為。
#
# 評估流程細節:
#   Dense 評估: 批次編碼 corpus + queries → 相似度矩陣 → MRR/NDCG
#   Hybrid 評估: HybridRetriever.search_batch() → 排名列表 → MRR/NDCG
# ============================================================

import gc
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger

from configs.base_config import BaseConfig
from configs.model_config import ModelConfig, ModelVariant
from configs.retrieval_config import RetrievalConfig
from configs.vram_config import VRAMConfig

from evaluation.metrics import evaluate_ranking_results
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.jieba_tokenizer import JiebaTokenizer
from training.dataset import ValDataset
from training.memory_utils import log_vram


# ============================================================
# 條件標籤常數 (供視覺化與報告使用)
# ============================================================
CONDITION_LABELS = {
    "small_baseline": "Small\nBaseline",
    "small_lora":     "Small\n+LoRA",
    "base_baseline":  "Base\nBaseline",
    "base_lora":      "Base\n+LoRA",
    "hybrid":         "Best LoRA\n+Hybrid",
}

CONDITION_ORDER = [
    "small_baseline",
    "small_lora",
    "base_baseline",
    "base_lora",
    "hybrid",
]


# ============================================================
# AblationEvaluator
# ============================================================

class AblationEvaluator:
    """
    自動化執行五種檢索條件的評估器。

    使用範例:
        evaluator = AblationEvaluator(val_dataset, base_cfg, model_cfg, retrieval_cfg, vram_cfg)
        results = evaluator.run_all_conditions()
        # results: {"small_baseline": {"mrr@10": ..., "ndcg@10": ...}, ...}
    """

    def __init__(
        self,
        val_dataset: ValDataset,
        base_cfg: BaseConfig,
        model_cfg: ModelConfig,
        retrieval_cfg: RetrievalConfig,
        vram_cfg: VRAMConfig,
        k: int = 10,
        jieba_user_dict_path: Optional[str] = None,
        jieba_stopwords_path: Optional[str] = None,
    ):
        """
        Args:
            val_dataset: ValDataset 物件 (含 queries, corpus, relevant_docs)
            base_cfg: 全域設定
            model_cfg: 模型設定 (small/base 變體)
            retrieval_cfg: 檢索設定 (batch_size, top_k, rrf_k 等)
            vram_cfg: VRAM 預算設定
            k: 評估截斷點 (預設 10 → MRR@10, NDCG@10)
            jieba_user_dict_path: jieba 自定義辭典 (可選)
            jieba_stopwords_path: 停用詞清單 (可選)
        """
        self.val_dataset    = val_dataset
        self.base_cfg       = base_cfg
        self.model_cfg      = model_cfg
        self.retrieval_cfg  = retrieval_cfg
        self.vram_cfg       = vram_cfg
        self.k              = k

        # jieba tokenizer (BM25 使用，評估 E 條件)
        self._jieba_user_dict  = jieba_user_dict_path
        self._jieba_stopwords  = jieba_stopwords_path

        # 快取預先提取的 query_ids / corpus_ids (avoid repeated dict operations)
        self._query_ids   = list(val_dataset.queries.keys())
        self._corpus_ids  = list(val_dataset.corpus.keys())
        self._query_texts = [val_dataset.queries[qid] for qid in self._query_ids]
        self._corpus_texts = [val_dataset.corpus[cid] for cid in self._corpus_ids]

        logger.info(
            f"AblationEvaluator 初始化:\n"
            f"  queries: {len(self._query_ids)}\n"
            f"  corpus:  {len(self._corpus_ids)}\n"
            f"  k:       {k}"
        )

    # ----------------------------------------------------------------
    # 內部: Dense 評估 (不依賴 DenseRetriever 的 corpus 索引狀態)
    # ----------------------------------------------------------------

    def _evaluate_dense(
        self,
        dense_retriever: DenseRetriever,
        label: str,
    ) -> Dict[str, float]:
        """
        使用給定的 DenseRetriever 評估 Dense 檢索效能。

        直接呼叫 dense_retriever.encode() 計算 embedding，
        不使用 build_corpus_index() / search()，以避免修改 retriever 的內部狀態，
        讓 E 條件的 Hybrid 評估可以在同一個 retriever 上繼續進行。

        Args:
            dense_retriever: 已載入模型的 DenseRetriever
            label: 日誌標籤 (例如 "small_baseline")

        Returns:
            {"mrr@10": float, "ndcg@10": float, "num_queries": int, "hit_rate@10": float}
        """
        logger.info(f"[評估] {label}: 開始 Dense 評估...")
        eval_batch = self.retrieval_cfg.dense_encode_batch_size

        # 1. 批次編碼 corpus (不加 query 指令前綴)
        logger.debug(f"  編碼 corpus: {len(self._corpus_texts)} 份文件")
        corpus_embs = dense_retriever.encode(
            self._corpus_texts,
            batch_size=eval_batch,
            is_query=False,
            show_progress=False,
        )  # (N, D) numpy float32

        # 2. 批次編碼 queries (加 query 指令前綴)
        logger.debug(f"  編碼 queries: {len(self._query_texts)} 個")
        query_embs = dense_retriever.encode(
            self._query_texts,
            batch_size=eval_batch,
            is_query=True,
            show_progress=False,
        )  # (Q, D) numpy float32

        # 3. 計算相似度矩陣
        # (Q, D) × (D, N) → (Q, N)
        # 兩者皆已 L2 正規化 → 點積 = 餘弦相似度
        # ★ VRAM 防護: numpy dot product 在 CPU 執行，不佔 VRAM
        sims = np.dot(query_embs, corpus_embs.T)  # (Q, N)

        # 4. 對每個 query 按相似度排序，取 doc_id 列表
        ranked_results: List[List[str]] = []
        for i in range(len(self._query_ids)):
            ranked_indices = np.argsort(sims[i])[::-1]  # 從高到低
            ranked_docs = [self._corpus_ids[j] for j in ranked_indices]
            ranked_results.append(ranked_docs)

        # 5. 計算指標
        metrics = evaluate_ranking_results(
            query_ids=self._query_ids,
            ranked_results=ranked_results,
            relevant_docs=self.val_dataset.relevant_docs,
            k=self.k,
        )

        logger.info(
            f"[評估] {label}: "
            f"MRR@{self.k}={metrics[f'mrr@{self.k}']:.4f} | "
            f"NDCG@{self.k}={metrics[f'ndcg@{self.k}']:.4f}"
        )
        return metrics

    # ----------------------------------------------------------------
    # 內部: Hybrid 評估
    # ----------------------------------------------------------------

    def _evaluate_hybrid(
        self,
        hybrid_retriever: HybridRetriever,
        label: str = "hybrid",
    ) -> Dict[str, float]:
        """
        使用 HybridRetriever 評估 RRF 融合後的檢索效能。

        ★ 此函數假設 hybrid_retriever.build_index() 已被呼叫。

        Args:
            hybrid_retriever: 已建立索引的 HybridRetriever
            label: 日誌標籤

        Returns:
            {"mrr@10": float, "ndcg@10": float, ...}
        """
        logger.info(f"[評估] {label}: 開始 Hybrid 評估 (Dense + BM25 + RRF)...")

        # Hybrid 評估: 使用 search() 取得 RRF 融合排名
        # 取比 k 更多的候選 (50) 確保 MRR/NDCG 計算完整
        search_top_k = max(50, self.k * 5)
        ranked_results: List[List[str]] = []

        for i, (q_id, q_text) in enumerate(zip(self._query_ids, self._query_texts)):
            # search() 回傳 [(doc_id, rrf_score), ...]
            fused = hybrid_retriever.search(q_text, top_k=search_top_k)
            ranked_docs = [doc_id for doc_id, _ in fused]
            ranked_results.append(ranked_docs)

            if (i + 1) % 50 == 0:
                logger.debug(f"  Hybrid 查詢進度: {i+1}/{len(self._query_ids)}")

        metrics = evaluate_ranking_results(
            query_ids=self._query_ids,
            ranked_results=ranked_results,
            relevant_docs=self.val_dataset.relevant_docs,
            k=self.k,
        )

        logger.info(
            f"[評估] {label}: "
            f"MRR@{self.k}={metrics[f'mrr@{self.k}']:.4f} | "
            f"NDCG@{self.k}={metrics[f'ndcg@{self.k}']:.4f}"
        )
        return metrics

    # ----------------------------------------------------------------
    # 主評估流程
    # ----------------------------------------------------------------

    def run_all_conditions(self) -> Dict[str, Dict[str, float]]:
        """
        依序執行五種條件的評估，回傳完整結果字典。

        VRAM 使用流程:
          [載入 small] → 評估 A, B → [釋放 small]
          [載入 base]  → 評估 C, D → [保留 base for E]
          [建立 BM25]  → 評估 E → [釋放 base]

        Returns:
            {
                "small_baseline": {"mrr@10": float, "ndcg@10": float, ...},
                "small_lora":     {...},
                "base_baseline":  {...},
                "base_lora":      {...},
                "hybrid":         {...},
            }
        """
        results: Dict[str, Dict[str, float]] = {}

        # ================================================================
        # 條件 A & B: bge-small (Baseline vs LoRA)
        # ================================================================
        logger.info(f"\n{'='*60}\n[A/B] bge-small 評估\n{'='*60}")

        dense_small = DenseRetriever(
            model_variant=self.model_cfg.small,
            base_cfg=self.base_cfg,
            retrieval_cfg=self.retrieval_cfg,
            use_lora=True,  # 嘗試載入 LoRA，若不存在自動退回 baseline
        )
        dense_small.load_model()
        log_vram("small 模型載入後")

        # A. bge-small Baseline: 停用 LoRA adapter
        dense_small.disable_lora()
        results["small_baseline"] = self._evaluate_dense(dense_small, "small_baseline")

        # B. bge-small + LoRA: 啟用 LoRA adapter
        dense_small.enable_lora()
        results["small_lora"] = self._evaluate_dense(dense_small, "small_lora")

        # 釋放 small 模型 VRAM
        dense_small.unload_model()
        log_vram("small 模型釋放後")

        # ================================================================
        # 條件 C & D: bge-base (Baseline vs LoRA)
        # ================================================================
        logger.info(f"\n{'='*60}\n[C/D] bge-base 評估\n{'='*60}")

        dense_base = DenseRetriever(
            model_variant=self.model_cfg.base,
            base_cfg=self.base_cfg,
            retrieval_cfg=self.retrieval_cfg,
            use_lora=True,
        )
        dense_base.load_model()
        log_vram("base 模型載入後")

        # C. bge-base Baseline
        dense_base.disable_lora()
        results["base_baseline"] = self._evaluate_dense(dense_base, "base_baseline")

        # D. bge-base + LoRA
        dense_base.enable_lora()
        results["base_lora"] = self._evaluate_dense(dense_base, "base_lora")

        # ================================================================
        # 條件 E: 最佳 LoRA + Hybrid RRF
        # 使用目前仍在記憶體中的 base model (LoRA 啟用狀態)
        # ================================================================
        logger.info(f"\n{'='*60}\n[E] Hybrid RRF 評估\n{'='*60}")

        # ★ BM25 純 CPU，不佔 VRAM
        tokenizer = JiebaTokenizer(
            user_dict_path=self._jieba_user_dict,
            stopwords_path=self._jieba_stopwords,
        )
        bm25_retriever = BM25Retriever(
            tokenizer=tokenizer,
            k1=self.retrieval_cfg.bm25_k1,
            b=self.retrieval_cfg.bm25_b,
        )
        bm25_retriever.build_index(self.val_dataset.corpus)
        log_vram("BM25 索引建立後 (應與前相同，BM25 在 CPU)")

        # 決定 Hybrid 使用哪個模型 (base LoRA 已在記憶體中)
        # 若 small LoRA 表現優於 base LoRA，才需要換回 small
        small_lora_mrr = results["small_lora"].get(f"mrr@{self.k}", 0.0)
        base_lora_mrr  = results["base_lora"].get(f"mrr@{self.k}", 0.0)

        if small_lora_mrr > base_lora_mrr:
            logger.info(
                f"small LoRA MRR@{self.k} ({small_lora_mrr:.4f}) > "
                f"base LoRA MRR@{self.k} ({base_lora_mrr:.4f})\n"
                f"★ 切換至 small 模型進行 Hybrid 評估"
            )
            # 釋放 base，載入 small
            dense_base.unload_model()
            dense_hybrid = DenseRetriever(
                model_variant=self.model_cfg.small,
                base_cfg=self.base_cfg,
                retrieval_cfg=self.retrieval_cfg,
                use_lora=True,
            )
            dense_hybrid.load_model()
            dense_hybrid.enable_lora()
        else:
            logger.info(
                f"base LoRA MRR@{self.k} ({base_lora_mrr:.4f}) >= "
                f"small LoRA MRR@{self.k} ({small_lora_mrr:.4f})\n"
                f"★ 使用 base 模型 (LoRA 啟用) 進行 Hybrid 評估"
            )
            dense_hybrid = dense_base  # 直接沿用，LoRA 已啟用

        # 建立 Dense 語料庫索引 (Hybrid 評估需要)
        dense_hybrid.build_corpus_index(self.val_dataset.corpus)

        # 組裝 Hybrid Retriever
        hybrid = HybridRetriever(
            dense_retriever=dense_hybrid,
            bm25_retriever=bm25_retriever,
            retrieval_cfg=self.retrieval_cfg,
        )

        results["hybrid"] = self._evaluate_hybrid(hybrid, "hybrid")

        # 釋放最後一個模型
        dense_hybrid.unload_model()
        log_vram("所有評估完成，模型已釋放")

        # ================================================================
        # 最終摘要
        # ================================================================
        self._log_summary(results)
        return results

    def _log_summary(self, results: Dict[str, Dict[str, float]]) -> None:
        """在 logger 輸出五種條件的對比摘要。"""
        mrr_key  = f"mrr@{self.k}"
        ndcg_key = f"ndcg@{self.k}"

        logger.info(f"\n{'='*60}")
        logger.info("消融實驗結果摘要")
        logger.info(f"{'─'*60}")
        logger.info(f"{'條件':<22} | {'MRR@'+str(self.k):>8} | {'NDCG@'+str(self.k):>8}")
        logger.info(f"{'─'*60}")

        for cond in CONDITION_ORDER:
            if cond not in results:
                continue
            m = results[cond]
            label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
            logger.info(
                f"{label:<22} | {m.get(mrr_key, 0):.4f}   | {m.get(ndcg_key, 0):.4f}"
            )

        logger.info(f"{'='*60}\n")

    # ----------------------------------------------------------------
    # UMAP 用 embedding 擷取
    # ----------------------------------------------------------------

    def extract_umap_embeddings(
        self,
        model_variant: ModelVariant,
        adapter_path: Optional[Path] = None,
        max_samples: int = 300,
    ) -> Dict[str, np.ndarray]:
        """
        從 val_dataset 中取樣，分別以「Baseline」與「LoRA」模式編碼，
        回傳供 UMAP 可視化用的 embedding 字典。

        採樣策略: 最多取 max_samples 個三元組 (query, positive, hard_negative)。
        若 val_dataset 的三元組不足，取全部。

        ★ OOM 防護:
          所有 embedding 在 GPU 計算後立即 `.cpu().numpy()` 轉移到 CPU，
          UMAP 本身也在 CPU 上執行，不額外佔用 VRAM。

        Args:
            model_variant: 要使用的模型變體 (small 或 base)
            adapter_path:  LoRA adapter 路徑 (None = 自動推導)
            max_samples:   最多採樣的三元組數量

        Returns:
            {
                "baseline_queries":   (N, D),
                "baseline_positives": (N, D),
                "baseline_negatives": (N, D),
                "lora_queries":       (N, D),
                "lora_positives":     (N, D),
                "lora_negatives":     (N, D),
            }
            所有值均為 float32 numpy array，已 L2 正規化。
        """
        logger.info(
            f"提取 UMAP embedding: {model_variant.short_name} | "
            f"max_samples={max_samples}"
        )

        # ----------------------------------------------------------------
        # 從 val_dataset 重建三元組 (query, positive, hard_negative)
        # ----------------------------------------------------------------
        # ValDataset 只儲存 queries, corpus, relevant_docs，
        # 需要反向對應找出 positive 和 negative
        triplets = self._build_triplets(max_samples)
        if not triplets:
            logger.warning("無法建立三元組，UMAP 視覺化將被跳過")
            return {}

        queries   = [t[0] for t in triplets]
        positives = [t[1] for t in triplets]
        negatives = [t[2] for t in triplets]

        logger.info(f"  採樣三元組數: {len(triplets)}")

        # ----------------------------------------------------------------
        # 載入模型 (LoRA 模式，含切換開關)
        # ----------------------------------------------------------------
        dense = DenseRetriever(
            model_variant=model_variant,
            base_cfg=self.base_cfg,
            retrieval_cfg=self.retrieval_cfg,
            use_lora=True,
        )
        dense.load_model(adapter_path=adapter_path)

        eval_batch = min(32, self.retrieval_cfg.dense_encode_batch_size)

        # Baseline embeddings
        dense.disable_lora()
        baseline_q = dense.encode(queries,   batch_size=eval_batch, is_query=True,  show_progress=False)
        baseline_p = dense.encode(positives, batch_size=eval_batch, is_query=False, show_progress=False)
        baseline_n = dense.encode(negatives, batch_size=eval_batch, is_query=False, show_progress=False)

        # LoRA embeddings
        dense.enable_lora()
        lora_q = dense.encode(queries,   batch_size=eval_batch, is_query=True,  show_progress=False)
        lora_p = dense.encode(positives, batch_size=eval_batch, is_query=False, show_progress=False)
        lora_n = dense.encode(negatives, batch_size=eval_batch, is_query=False, show_progress=False)

        dense.unload_model()
        log_vram("UMAP embedding 提取完成，模型已釋放")

        return {
            "baseline_queries":   baseline_q,
            "baseline_positives": baseline_p,
            "baseline_negatives": baseline_n,
            "lora_queries":       lora_q,
            "lora_positives":     lora_p,
            "lora_negatives":     lora_n,
        }

    def _build_triplets(
        self,
        max_samples: int,
    ) -> List[Tuple[str, str, str]]:
        """
        從 ValDataset 重建 (query, positive, hard_negative) 三元組。

        ValDataset 儲存結構:
          queries:       {q_id: query_text}
          corpus:        {p_id/n_id: passage_text}
          relevant_docs: {q_id: {p_id}}  ← 每個 q_id 通常對應一個 positive

        重建邏輯:
          對每個 q_id，找出 relevant_docs[q_id] 的 positive，
          再從 corpus 中找一個不在 relevant_docs 的段落當 negative。
        """
        triplets = []
        corpus_id_list = list(self.val_dataset.corpus.keys())

        for q_id, q_text in self.val_dataset.queries.items():
            positive_ids = self.val_dataset.relevant_docs.get(q_id, set())
            if not positive_ids:
                continue

            p_id = next(iter(positive_ids))
            p_text = self.val_dataset.corpus.get(p_id)
            if not p_text:
                continue

            # 找一個 negative (不在 positive_ids 中的語料段落)
            neg_text = None
            for cid in corpus_id_list:
                if cid not in positive_ids:
                    neg_text = self.val_dataset.corpus[cid]
                    break

            if neg_text is None:
                continue

            triplets.append((q_text, p_text, neg_text))

            if len(triplets) >= max_samples:
                break

        return triplets
