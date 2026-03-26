# ============================================================
# retrieval/dense_retriever.py — Dense 向量檢索器 (GPU)
#
# 職責: 載入 BGE 模型 (+ 可選的 LoRA Adapter)，對語料庫預先編碼，
#       並對查詢進行即時向量檢索。
#
# ★ LoRA 動態掛載與 A/B 測試設計:
#   本模組的核心功能之一是支援 Baseline vs LoRA 的切換，
#   讓 Step 5 的評估腳本可以在同一個語料庫索引上比較兩者效能。
#
#   切換方式: 使用 peft 的 disable_adapter_layers() / enable_adapter_layers()
#   ┌─────────────────────────────────────────────────────────┐
#   │  retriever.disable_lora()  → 模型行為 = 純 Base Model   │
#   │  retriever.enable_lora()   → 模型行為 = Base + LoRA      │
#   └─────────────────────────────────────────────────────────┘
#
#   此設計的優點:
#   1. 不需要重新載入模型 (節省 5~30 秒的 IO 時間)
#   2. 不需要額外的 VRAM (兩套模型同時佔記憶體)
#   3. LoRA Adapter 的參數量極小 (~2-10 MB)，切換成本接近零
#
#   注意: 切換 LoRA 狀態後，必須重新呼叫 build_corpus_index()，
#         因為語料庫的 embedding 會因 LoRA 開啟與否而不同。
#
# ★ 推論階段的 VRAM 使用估算:
#   bge-small (fp16): ~48 MB 模型 + ~64 MB 工作空間 ≈ 112 MB
#   bge-base  (fp16): ~204 MB 模型 + ~128 MB 工作空間 ≈ 332 MB
#   語料庫 embedding (float32, 10K × 512): ~20 MB (儲存在 CPU RAM)
#   語料庫 embedding (float32, 10K × 768): ~30 MB (儲存在 CPU RAM)
#   ★ 語料庫 embedding 儲存在 CPU，避免佔用珍貴的 GPU VRAM
# ============================================================

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft 未安裝，LoRA 動態掛載功能不可用。請執行: pip install peft")

from configs.base_config import BaseConfig
from configs.model_config import ModelConfig, ModelVariant
from configs.retrieval_config import RetrievalConfig


# ============================================================
# Mean Pooling + L2 Normalization
# (與 training/lora_trainer.py 的 mean_pool_and_normalize 完全一致)
# ★ 訓練與推論必須使用相同的 pooling 邏輯，否則向量空間不匹配
# ============================================================

def mean_pool_and_normalize(
    model_output: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    對 Transformer 最後一層的 token embeddings 做 Masked Mean Pooling，
    再做 L2 正規化，使向量模長為 1。

    Masked Mean Pooling 原理:
      只對非填充 (non-padding) 的 token 做平均，避免 [PAD] token 汙染結果。
      步驟:
        1. 將 attention_mask 擴展至 (B, L, D) → 遮蔽 [PAD] token
        2. 對遮蔽後的 token embeddings 沿序列長度維度加總
        3. 除以有效 token 數 (clamp 至 1e-9 避免除以零)

    L2 正規化的意義:
      L2 正規化後，兩向量的點積 = 餘弦相似度 (值域 [-1, 1])，
      不需要再除以向量模長，簡化相似度計算，也讓分數有固定的物理意義。

    Args:
        model_output: AutoModel 輸出的 last_hidden_state, shape (B, L, D)
        attention_mask: tokenizer 輸出的 attention_mask, shape (B, L)
                        1 = 真實 token，0 = [PAD]

    Returns:
        shape (B, D) 的 L2 正規化 embedding 矩陣
    """
    # attention_mask: (B, L) → (B, L, 1) → broadcast 至 (B, L, D)
    mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()

    # 遮蔽 [PAD] token 並沿 L 維度加總: (B, D)
    sum_embeddings = torch.sum(model_output * mask_expanded, dim=1)

    # 每個樣本的有效 token 數: (B, 1)，clamp 避免除以零
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)

    # Masked mean pooling: (B, D)
    pooled = sum_embeddings / sum_mask

    # L2 正規化: 每個向量的模長 = 1 → 點積 = 餘弦相似度
    return F.normalize(pooled, p=2, dim=1)


# ============================================================
# Dense Retriever
# ============================================================

class DenseRetriever:
    """
    BGE 模型 + LoRA Adapter 的 Dense 向量檢索器。

    完整工作流程:
        retriever = DenseRetriever(variant, base_cfg, retrieval_cfg, use_lora=True)
        retriever.load_model(adapter_path)        # 載入模型
        retriever.build_corpus_index(corpus)      # 預編碼語料庫
        results = retriever.search(query, top_k)  # 查詢

    A/B 測試流程 (Baseline vs LoRA):
        retriever.disable_lora()
        retriever.build_corpus_index(corpus)      # 必須重新編碼！
        baseline_results = retriever.search(query)

        retriever.enable_lora()
        retriever.build_corpus_index(corpus)      # 必須重新編碼！
        lora_results = retriever.search(query)
    """

    def __init__(
        self,
        model_variant: ModelVariant,
        base_cfg: BaseConfig,
        retrieval_cfg: RetrievalConfig,
        use_lora: bool = True,
    ):
        """
        Args:
            model_variant: ModelVariant 實例 (small 或 base)
            base_cfg: 全域設定 (路徑、裝置等)
            retrieval_cfg: 檢索設定 (batch_size、top_k 等)
            use_lora: 是否載入 LoRA Adapter。
                      True  = 載入 Base Model + LoRA (Fine-tuned 版本)
                      False = 只載入 Base Model (Baseline 版本)
        """
        self.model_variant = model_variant
        self.base_cfg = base_cfg
        self.retrieval_cfg = retrieval_cfg
        self.use_lora = use_lora

        # 優先使用 GPU (若 VRAM 足夠)，否則退回 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型與 tokenizer (load_model 後設定)
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer = None
        self._lora_loaded: bool = False  # 是否成功載入了 LoRA adapter

        # 語料庫索引 (build_corpus_index 後設定)
        # 儲存在 CPU numpy array，避免佔用 GPU VRAM
        # 查詢時將 query embedding 送回 GPU 計算後，再拉回 CPU 排序
        self.corpus_embeddings: Optional[np.ndarray] = None  # (N, D) float32
        self.corpus_ids: List[str] = []

        logger.debug(
            f"DenseRetriever 初始化: {model_variant.short_name} | "
            f"use_lora={use_lora} | device={self.device}"
        )

    # ----------------------------------------------------------------
    # 模型載入
    # ----------------------------------------------------------------

    def load_model(self, adapter_path: Optional[Path] = None) -> None:
        """
        載入 Base Model，並可選地掛載 LoRA Adapter。

        Args:
            adapter_path: LoRA Adapter 的路徑 (Step 3 訓練輸出的 best/ 目錄)
                          若 use_lora=True 但 adapter_path=None，會自動推導路徑。
                          若 use_lora=False，此參數被忽略。

        ★ 載入順序:
          1. AutoModel.from_pretrained (Base Model, fp16)
          2. PeftModel.from_pretrained (掛載 LoRA Adapter)
          3. model.eval() 關閉 Dropout (推論模式)
          4. model.to(device) 移至 GPU
        """
        model_name = self.model_variant.model_name
        cache_dir = str(self.base_cfg.base_models_dir)

        logger.info(f"載入 Base Model: {model_name}")

        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        # 載入 Base Model (fp16 減少 VRAM 佔用)
        base_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        )

        # ----------------------------------------------------------------
        # LoRA Adapter 掛載
        # ----------------------------------------------------------------
        if self.use_lora:
            if not PEFT_AVAILABLE:
                logger.warning("peft 未安裝，退回純 Base Model (Baseline 模式)")
                self.model = base_model
                self._lora_loaded = False
            else:
                # 自動推導 adapter 路徑 (若未指定)
                if adapter_path is None:
                    adapter_path = (
                        self.base_cfg.lora_adapters_dir
                        / self.model_variant.lora_adapter_dirname
                        / "best"
                    )

                if not adapter_path.exists():
                    logger.warning(
                        f"LoRA Adapter 路徑不存在: {adapter_path}\n"
                        f"退回純 Base Model (Baseline 模式)。\n"
                        f"請先執行 LoRA 訓練: make train-{self.model_variant.short_name}"
                    )
                    self.model = base_model
                    self._lora_loaded = False
                else:
                    logger.info(f"掛載 LoRA Adapter: {adapter_path}")
                    # PeftModel.from_pretrained: 將 LoRA 參數以 delta 形式加載，
                    # 不改變 Base Model 權重 (仍可透過 disable/enable 切換)
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        str(adapter_path),
                        is_trainable=False,  # 推論模式: 凍結所有參數
                    )
                    self._lora_loaded = True
                    logger.info("LoRA Adapter 掛載成功")
        else:
            # Baseline 模式: 只使用 Base Model
            self.model = base_model
            self._lora_loaded = False

        # 推論模式 + 移至目標裝置
        self.model.eval()
        self.model = self.model.to(self.device)

        # 計算並記錄 VRAM 使用量
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(
                f"模型載入完成 | "
                f"LoRA={'on' if self._lora_loaded else 'off'} | "
                f"VRAM 已用: {vram_mb:.0f} MB"
            )

    # ----------------------------------------------------------------
    # LoRA 切換開關 (用於 A/B 測試)
    # ----------------------------------------------------------------

    def enable_lora(self) -> None:
        """
        啟用 LoRA Adapter (切換至 Fine-tuned 模式)。

        ★ 重要提醒:
          啟用/停用 LoRA 後，語料庫 embedding 會改變，
          必須重新呼叫 build_corpus_index() 重新編碼語料庫。
          否則 query embedding (LoRA on) 與 corpus embedding (LoRA off) 不在同一空間，
          相似度計算結果毫無意義。

        技術細節:
          peft 的 disable/enable_adapter_layers() 透過 hook 機制實現:
          停用時跳過 LoRA 的低秩矩陣計算 (A, B)，直接使用原始 W0 矩陣。
          啟用時恢復計算 W0 + alpha/r × B × A 的加法。
        """
        if not self._lora_loaded:
            logger.warning("未載入 LoRA Adapter，enable_lora() 無效")
            return
        if hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()
            logger.info("LoRA Adapter 已啟用 (Fine-tuned 模式)")
        else:
            logger.warning("模型不支援 enable_adapter_layers()")

    def disable_lora(self) -> None:
        """
        停用 LoRA Adapter (切換至 Baseline 模式)。

        停用後，模型行為等同於純 Base Model (未微調版本)。
        這讓我們可以在相同的語料庫上進行公平的 A/B 比較，
        確保唯一的差異來自 LoRA 微調，而非不同的模型或資料。

        ★ 提醒: 切換後必須重新 build_corpus_index()。
        """
        if not self._lora_loaded:
            logger.warning("未載入 LoRA Adapter，disable_lora() 無效")
            return
        if hasattr(self.model, "disable_adapter_layers"):
            self.model.disable_adapter_layers()
            logger.info("LoRA Adapter 已停用 (Baseline 模式)")
        else:
            logger.warning("模型不支援 disable_adapter_layers()")

    @property
    def lora_active(self) -> bool:
        """
        回傳 LoRA Adapter 目前是否啟用。
        用於日誌與斷言檢查。
        """
        if not self._lora_loaded:
            return False
        # peft PeftModel 的內部狀態: _peft_config 或 active_adapters
        if hasattr(self.model, "active_adapters"):
            return len(self.model.active_adapters) > 0
        # 退路: 若無法取得狀態，假設啟用中
        return True

    # ----------------------------------------------------------------
    # 編碼函數
    # ----------------------------------------------------------------

    def _encode_batch(
        self,
        texts: List[str],
        is_query: bool = False,
    ) -> np.ndarray:
        """
        對單一 batch 的文字進行編碼。

        Args:
            texts: 文字列表 (batch 內的所有文字)
            is_query: 若為 True，自動加入 BGE 的 query 指令前綴

        Returns:
            (batch_size, hidden_size) 的 float32 numpy array (已 L2 正規化)

        ★ 為何回傳 numpy 而非 torch.Tensor?
          語料庫 embedding 儲存在 CPU numpy array，避免長期佔用 VRAM。
          numpy 的矩陣乘法 (dot product) 對 10K-100K 規模的語料庫已足夠快。
          若語料庫超過 100K 文件，可考慮保留 GPU tensor 並改用 torch.mm。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型尚未載入，請先呼叫 load_model()")

        # BGE v1.5: query 需加前綴指令，corpus 不需要
        # 此指令幫助模型理解當前是「檢索查詢」任務而非「文章相似度」任務
        if is_query:
            model_cfg = ModelConfig()
            texts = [model_cfg.query_instruction + t for t in texts]

        # 動態 padding: 以 batch 內最長序列為基準，避免固定長度造成浪費
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,   # 與訓練時的 max_seq_length 保持一致
            return_tensors="pt",
        )

        # 將輸入移至 GPU
        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # 推論時不需要梯度計算，節省記憶體
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # last_hidden_state: (B, L, D)
            last_hidden = outputs.last_hidden_state

            # Mean pooling + L2 normalize → (B, D)
            embeddings = mean_pool_and_normalize(last_hidden, attention_mask)

        # 移回 CPU → 轉為 float32 numpy (節省 VRAM，避免長期持有 GPU tensor)
        return embeddings.cpu().float().numpy()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        is_query: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        批次編碼多段文字，回傳已 L2 正規化的 embedding 矩陣。

        Args:
            texts: 文字列表
            batch_size: 每次推論的批次大小 (推論階段不需要梯度，可比訓練時大)
            is_query: 是否為查詢文字 (影響是否加入 BGE 前綴)
            show_progress: 是否顯示進度條

        Returns:
            (len(texts), hidden_size) float32 numpy array
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx : batch_idx + batch_size]
            batch_embs = self._encode_batch(batch_texts, is_query=is_query)
            all_embeddings.append(batch_embs)

            if show_progress and total_batches > 5:
                current_batch = batch_idx // batch_size + 1
                if current_batch % max(1, total_batches // 10) == 0:
                    logger.debug(
                        f"  編碼進度: {current_batch}/{total_batches} batches "
                        f"({current_batch/total_batches:.0%})"
                    )

        return np.vstack(all_embeddings)  # (N, D)

    # ----------------------------------------------------------------
    # 語料庫索引建立
    # ----------------------------------------------------------------

    def build_corpus_index(self, corpus: Dict[str, str]) -> None:
        """
        預先編碼整個語料庫並建立向量索引。

        ★ 設計原則: O(N) 編碼，O(D) 查詢
          語料庫 embedding 預先計算一次 (O(N) forward passes)，
          每次查詢只需編碼 query (O(1) forward pass) + 矩陣乘法 (O(N×D))。
          對比「每次查詢重新編碼語料庫」的 O(N × Q) 設計，效率差距數量級。

        ★ 切換 LoRA 狀態後必須重新呼叫此函數:
          因為 LoRA on/off 會改變語料庫的 embedding，舊的 embedding 快取失效。

        Args:
            corpus: {doc_id: doc_text} 字典

        Note:
            語料庫 embedding 儲存在 self.corpus_embeddings (CPU numpy array)，
            不佔用 GPU VRAM，讓 GPU 可以保持空閒供查詢時的 encode 使用。
        """
        if not corpus:
            raise ValueError("語料庫為空，無法建立索引")

        logger.info(
            f"開始建立 Dense 語料庫索引: {len(corpus)} 份文件 | "
            f"LoRA={'on' if self.lora_active else 'off'}"
        )

        self.corpus_ids = list(corpus.keys())
        doc_texts = [corpus[did] for did in self.corpus_ids]

        # 語料庫文件不加 query_instruction
        self.corpus_embeddings = self.encode(
            doc_texts,
            batch_size=self.retrieval_cfg.dense_encode_batch_size,
            is_query=False,
            show_progress=True,
        )
        # corpus_embeddings shape: (N, D), dtype=float32, 已 L2 正規化

        logger.info(
            f"Dense 索引建立完成: shape={self.corpus_embeddings.shape} | "
            f"記憶體佔用: {self.corpus_embeddings.nbytes / (1024**2):.1f} MB (CPU RAM)"
        )

    # ----------------------------------------------------------------
    # 查詢
    # ----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Dense 向量相似度檢索。

        Args:
            query: 查詢文字 (函數內部自動加入 BGE 指令前綴)
            top_k: 回傳的候選文件數量

        Returns:
            [(doc_id, cosine_similarity), ...] 依相似度從高到低排序
            相似度值域: [-1, 1]，1 表示完全相同，-1 表示完全相反

        ★ 效能分析 (語料庫 10K 文件):
          - query 編碼: ~5-20 ms (單一 batch，GPU)
          - 相似度計算: ~1 ms (numpy dot product, CPU)
          - top-k 排序: ~0.5 ms (numpy argsort, CPU)
          - 總計: ~10-25 ms per query
        """
        if self.corpus_embeddings is None or len(self.corpus_ids) == 0:
            raise RuntimeError(
                "語料庫索引尚未建立，請先呼叫 build_corpus_index(corpus)"
            )

        # 編碼 query (加入 BGE 指令前綴)
        query_emb = self.encode(
            [query],
            batch_size=1,
            is_query=True,
            show_progress=False,
        )  # shape: (1, D)

        # 計算 query 與所有語料庫文件的餘弦相似度
        # query_emb 與 corpus_embeddings 皆已 L2 正規化
        # → 點積 = 餘弦相似度，值域 [-1, 1]
        # dot product: (1, D) × (D, N) → (1, N) → squeeze → (N,)
        similarities: np.ndarray = np.dot(query_emb, self.corpus_embeddings.T)[0]

        # 取 top_k (argsort 從小到大，[::-1] 翻轉為從大到小)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.corpus_ids[i], float(similarities[i]))
            for i in top_indices
        ]

        return results

    # ----------------------------------------------------------------
    # 資源釋放
    # ----------------------------------------------------------------

    def unload_model(self) -> None:
        """
        釋放 GPU 記憶體，在切換至下一個模型或完成所有推論後呼叫。

        ★ 三步釋放協議 (與 VRAMGuard 一致):
          1. del self.model → 移除 Python 參考
          2. gc.collect() → 清除循環引用
          3. torch.cuda.empty_cache() → 釋放 PyTorch CUDA 快取
        """
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"Dense Retriever 模型已釋放 | 剩餘 VRAM: {vram_mb:.0f} MB")

    def __repr__(self) -> str:
        model_status = "loaded" if self.model is not None else "not loaded"
        index_status = f"{len(self.corpus_ids)} docs" if self.corpus_embeddings is not None else "not built"
        return (
            f"DenseRetriever("
            f"model={self.model_variant.short_name}, "
            f"lora={self._lora_loaded}, "
            f"model_status={model_status}, "
            f"index={index_status})"
        )
