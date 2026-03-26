# ============================================================
# retrieval/bm25_retriever.py — BM25 稀疏向量檢索器 (純 CPU)
#
# ★ 嚴禁規則: 本檔案內絕對不允許 import torch 或任何 GPU 相關函式庫。
#   理由: BM25 是稀疏模型，完全基於詞頻與逆文件頻率統計，
#         不需要也不應該佔用任何 GPU 資源。
#         在混合檢索管線中，BM25 與 Dense Retriever 在 CPU/GPU 上並行處理，
#         CPU-only 設計確保 BM25 永遠不會與 Dense Model 搶佔 VRAM。
#
# BM25 (Okapi BM25) 公式:
#   Score(D, Q) = Σ_i IDF(q_i) × f(q_i, D) × (k1 + 1)
#                                / (f(q_i, D) + k1 × (1 - b + b × |D| / avgdl))
#
#   其中:
#   - q_i: query 中的第 i 個詞
#   - f(q_i, D): q_i 在文件 D 中的詞頻
#   - |D|: 文件 D 的詞數
#   - avgdl: 語料庫的平均文件長度
#   - k1=1.5: 詞頻飽和度參數
#   - b=0.75: 文件長度正規化強度
#   - IDF(q_i) = log((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1)
#     N: 文件總數，n(q_i): 包含 q_i 的文件數
#
# 為何使用 BM25 而非 TF-IDF:
#   - BM25 的 k1 參數使詞頻的貢獻趨於飽和 (詞頻越高，邊際效益遞減)
#   - b 參數對長文件進行適當懲罰，避免長文件因詞頻加總而主導排名
#   - 實踐上 BM25 在 IR 任務上幾乎一致優於 TF-IDF
# ============================================================

# ★ 注意: 此處嚴禁 import torch、peft、transformers 等 GPU 函式庫
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.error("rank_bm25 未安裝，BM25 檢索功能無法使用。請執行: pip install rank-bm25")

from retrieval.jieba_tokenizer import JiebaTokenizer


class BM25Retriever:
    """
    基於 BM25Okapi 與 jieba 的中文稀疏向量檢索器。

    使用流程:
        retriever = BM25Retriever(tokenizer)
        retriever.build_index(corpus)            # 建立索引 (一次性)
        results = retriever.search(query, top_k) # 反覆查詢

    索引儲存與載入:
        retriever.save_index(path)   # 避免重複建立索引
        retriever.load_index(path)

    ★ BM25 索引建立的時間複雜度:
      O(N × L)，其中 N = 文件數，L = 平均文件長度 (詞數)
      10,000 份文件 × 平均 200 詞 = 約 2M 次斷詞操作
      jieba 精確模式速度約 500K 字/秒，建立索引約 10~30 秒

    ★ BM25 查詢的時間複雜度:
      O(|Q| × N) per query，其中 |Q| = query 詞數
      rank_bm25 使用 numpy 向量化，查詢速度很快 (~毫秒級)
    """

    def __init__(
        self,
        tokenizer: JiebaTokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Args:
            tokenizer: JiebaTokenizer 實例 (中文斷詞器)
            k1: BM25 詞頻飽和度參數。越大代表詞頻的影響更線性。
                原始論文建議 1.2 ~ 2.0，本專案預設 1.5
            b: BM25 文件長度正規化參數。
                0 = 完全不正規化，1 = 完全依文件長度正規化
                預設 0.75，多數 IR 任務的最佳值
        """
        if not BM25_AVAILABLE:
            raise ImportError(
                "rank_bm25 未安裝，請執行: pip install rank-bm25"
            )

        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b

        # 索引相關狀態 (build_index 後設定)
        self.bm25_index: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []      # 與 tokenized_corpus 平行對應的文件 ID 清單
        self.doc_count: int = 0

    def build_index(self, corpus: Dict[str, str]) -> None:
        """
        為語料庫建立 BM25 倒排索引。

        Args:
            corpus: {doc_id: doc_text} 字典

        ★ 效能說明:
            - jieba 斷詞是單執行緒的 (GIL 限制)，大語料可能需要數分鐘
            - 若要加速，可先用 jieba.enable_parallel(4) 啟用並行模式
              但並行模式在 Windows 下不穩定，建議 Linux/Docker 環境使用
            - 建立完成後用 save_index() 快取，避免重複建立
        """
        if not corpus:
            raise ValueError("語料庫為空，無法建立 BM25 索引")

        logger.info(f"開始建立 BM25 索引，文件數: {len(corpus)}")

        self.doc_ids = list(corpus.keys())

        # ★ 強制使用 jieba 斷詞，嚴禁使用 .split() 進行中文斷詞
        # .split() 對中文無效 (中文詞之間無空白分隔符)，會導致每個字元或整段文字
        # 被視為一個 token，完全破壞 BM25 的詞頻與 IDF 計算
        tokenized_corpus: List[List[str]] = []
        for doc_id in self.doc_ids:
            tokens = self.tokenizer.tokenize(corpus[doc_id])
            tokenized_corpus.append(tokens)

        # 建立 BM25 索引 (內部計算 IDF 與各文件的詞頻統計)
        self.bm25_index = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.doc_count = len(self.doc_ids)

        logger.info(
            f"BM25 索引建立完成: {self.doc_count} 份文件 | "
            f"k1={self.k1}, b={self.b} | "
            f"平均文件長度: {self.bm25_index.avgdl:.1f} 詞"
        )

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        BM25 稀疏向量檢索。

        Args:
            query: 查詢文字 (原始中文，函數內部自動斷詞)
            top_k: 回傳的候選文件數量

        Returns:
            [(doc_id, bm25_score), ...] 依分數從高到低排序
            若無相關文件 (分數皆為 0)，回傳空列表

        ★ BM25 分數的特性:
            - 非負數，無理論上限 (取決於語料庫的 IDF 分布)
            - 不同語料庫的分數不可直接比較 (IDF 值不同)
            - 因此 BM25 分數只用於排名，不用於跨系統比較
            - RRF 融合時也只使用排名 (rank)，不使用分數本身
        """
        if self.bm25_index is None:
            raise RuntimeError(
                "BM25 索引尚未建立，請先呼叫 build_index(corpus)"
            )

        # query 斷詞 (與建立索引時使用同一個 tokenizer，確保詞彙表一致)
        query_tokens = self.tokenizer.tokenize(query)

        if not query_tokens:
            logger.warning(f"query 斷詞後為空，可能是純停用詞或空字串: '{query}'")
            return []

        # BM25 對所有文件計算相關性分數
        # get_scores() 回傳 np.ndarray，shape = (doc_count,)
        scores: np.ndarray = self.bm25_index.get_scores(query_tokens)

        # 取 top_k (使用 argpartition 比 argsort 快，但結果需再排序)
        # 此處文件數通常 < 100K，argsort 的效能已足夠
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 只回傳分數 > 0 的結果 (分數為 0 表示 query 詞在文件中完全不出現)
        results = [
            (self.doc_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0.0
        ]

        return results

    def save_index(self, save_path: Path) -> None:
        """
        將 BM25 索引序列化儲存至磁碟。

        大語料庫建立索引耗時，序列化後可在下次啟動時直接載入，
        節省每次都需要重新斷詞和建立索引的時間。

        Args:
            save_path: 儲存路徑 (建議副檔名 .pkl)
        """
        if self.bm25_index is None:
            raise RuntimeError("BM25 索引尚未建立，無法儲存")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "bm25_index": self.bm25_index,
            "doc_ids":    self.doc_ids,
            "doc_count":  self.doc_count,
            "k1":         self.k1,
            "b":          self.b,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        size_mb = save_path.stat().st_size / (1024 ** 2)
        logger.info(f"BM25 索引已儲存: {save_path} ({size_mb:.1f} MB)")

    def load_index(self, load_path: Path) -> None:
        """
        從磁碟載入已序列化的 BM25 索引。

        Args:
            load_path: 先前 save_index() 儲存的路徑
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"BM25 索引檔案不存在: {load_path}")

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        self.bm25_index = state["bm25_index"]
        self.doc_ids    = state["doc_ids"]
        self.doc_count  = state["doc_count"]
        self.k1         = state["k1"]
        self.b          = state["b"]

        logger.info(
            f"BM25 索引載入完成: {self.doc_count} 份文件 ← {load_path}"
        )

    @property
    def is_built(self) -> bool:
        """回傳索引是否已建立。"""
        return self.bm25_index is not None

    def __repr__(self) -> str:
        status = f"{self.doc_count} docs" if self.is_built else "not built"
        return f"BM25Retriever(k1={self.k1}, b={self.b}, index={status})"
