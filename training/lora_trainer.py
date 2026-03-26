# ============================================================
# training/lora_trainer.py — LoRA 微調訓練器 (核心)
#
# 本模組為整個 Step 3 的核心，實作以下完整流程:
#   1. 載入 AutoModel (HuggingFace) + AutoTokenizer
#   2. 套用 peft LoRA (僅 query + value attention 層)
#   3. 凍結 Base Model 參數，僅訓練 LoRA adapter
#   4. fp16 混合精度訓練 (GradScaler + autocast)
#   5. Gradient Accumulation (有效 batch = 真實 batch × accum_steps)
#   6. AdamW + 線性 Warmup + Cosine 學習率衰減
#   7. 每 epoch 後評估 MRR@10 / NDCG@10
#   8. Early Stopping (consecutive patience 機制)
#   9. ★ 僅儲存 LoRA Adapter 權重 (~2-10 MB)，絕不儲存 Base Model
#
# VRAM 設計決策:
#   - 三路 (query/positive/negative) 共用同一個模型的單次 forward pass
#     若分三次 forward，啟動值可獨立釋放，VRAM 峰值較低
#     若合併為一次 forward (batch × 3)，VRAM 峰值較高但速度快
#   - 本實作採用分開三次 forward 策略，犧牲速度換取 VRAM 安全餘量
#     (速度代價約 10-15%，但避免 base 模型 OOM 的風險)
#
# fp16 / GradScaler 設計說明:
#   torch.autocast: 將 forward pass 中的矩陣運算自動轉為 fp16
#   GradScaler: 在 backward 前將 loss 乘以大數 (scale factor)，
#               使梯度在 fp16 精度下不會因過小而消失 (underflow)
#               optimizer.step 前會自動除回去
#   兩者需要配合使用，缺一不可
# ============================================================

import math
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

# peft: 僅修改 attention 層的 query/value，使用低秩分解節省 VRAM
from peft import LoraConfig, TaskType, get_peft_model

from configs.base_config import BaseConfig
from configs.model_config import ModelConfig, ModelVariant
from configs.training_config import LoRAConfig, TrainingConfig
from configs.vram_config import VRAMConfig

from training.dataset import ValDataset, create_train_dataloader
from training.loss_functions import MultipleNegativesRankingLoss
from training.memory_utils import (
    TrainingMemoryMonitor,
    count_trainable_parameters,
    handle_oom_error,
    log_vram,
)


# ============================================================
# 評估工具函數
# ============================================================

def compute_mrr_at_k(
    ranked_doc_ids: list,
    relevant_ids: set,
    k: int = 10,
) -> float:
    """
    計算單一 query 的 MRR@k。

    MRR = 1 / rank_of_first_relevant
    若前 k 名內無相關文件，MRR = 0。

    Args:
        ranked_doc_ids: 按相似度降序排列的文件 ID 列表
        relevant_ids: 此 query 的相關文件 ID 集合
        k: 評估的截斷點

    Returns:
        float: MRR 值 (0 到 1 之間)
    """
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    ranked_doc_ids: list,
    relevant_ids: set,
    k: int = 10,
) -> float:
    """
    計算單一 query 的 NDCG@k (Normalized Discounted Cumulative Gain)。

    DCG@k  = Σ_{i=1}^{k} rel_i / log2(i + 1)
    IDCG@k = DCG 的理想值 (所有相關文件排在最前面)
    NDCG@k = DCG@k / IDCG@k

    本實作使用二元相關性 (rel_i ∈ {0, 1})。

    Args:
        ranked_doc_ids: 按相似度降序排列的文件 ID 列表
        relevant_ids: 此 query 的相關文件 ID 集合
        k: 評估的截斷點

    Returns:
        float: NDCG 值 (0 到 1 之間)
    """
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # IDCG: 假設所有相關文件都排在最前面
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# Mean Pooling + L2 Normalize
# ============================================================

def mean_pool_and_normalize(
    token_embeddings: torch.Tensor,  # (B, L, D)
    attention_mask: torch.Tensor,    # (B, L)
) -> torch.Tensor:
    """
    對 token-level embeddings 做 mean pooling，再 L2 正規化。

    mean pooling 方法:
    - 僅對非 padding 位置的 token 取平均 (attention_mask=1)
    - 比 CLS token 更穩定，對短文本表現更好

    L2 正規化後:
    - 所有 embedding 向量的長度均為 1
    - dot product = cosine similarity，便於在 MNRL 計算相似度

    Args:
        token_embeddings: (B, L, D) 模型 last_hidden_state 輸出
        attention_mask:   (B, L) tokenizer 輸出的 attention mask

    Returns:
        (B, D) 已 L2 正規化的 sentence embeddings
    """
    # 擴展 mask 維度以便廣播乘法: (B, L) → (B, L, 1)
    input_mask_expanded = attention_mask.unsqueeze(-1).float()

    # Masked sum: 對所有有效 token 位置的 embedding 求和
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)  # (B, D)

    # Masked count: 每個樣本的有效 token 數量
    token_counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B, 1) 防止除零

    # 均值 embedding
    pooled = sum_embeddings / token_counts  # (B, D)

    # L2 正規化: 使 dot product 等效於 cosine similarity
    normalized = F.normalize(pooled, p=2, dim=1)

    return normalized


# ============================================================
# LoRA Trainer 主類別
# ============================================================

class LoRATrainer:
    """
    LoRA Adapter 微調訓練器。

    支援 bge-small 與 bge-base 的消融實驗切換，
    所有 VRAM 限制相關的超參數均從 configs/ 讀取，
    不在此類別內 hardcode。
    """

    def __init__(
        self,
        model_variant: ModelVariant,
        base_cfg: BaseConfig,
        training_cfg: TrainingConfig,
        vram_cfg: VRAMConfig,
        symmetric_loss: bool = False,
    ):
        """
        Args:
            model_variant: ModelVariant 物件 (small 或 base)，含 HuggingFace 模型名稱
            base_cfg:      全域基礎設定 (路徑、裝置等)
            training_cfg:  訓練超參數
            vram_cfg:      VRAM 預算設定 (用於監控閾值)
            symmetric_loss: 是否使用對稱 MNRL (雙向損失)
        """
        self.variant = model_variant
        self.base_cfg = base_cfg
        self.training_cfg = training_cfg
        self.vram_cfg = vram_cfg
        self.device = torch.device(base_cfg.device if torch.cuda.is_available() else "cpu")

        # 依模型變體取得對應的 batch size 與 gradient accumulation steps
        self.batch_size = training_cfg.get_batch_size(model_variant.short_name)
        self.accum_steps = training_cfg.get_gradient_accumulation_steps(model_variant.short_name)

        # 儲存路徑: models/lora_adapters/{bge-small-zh-lora 或 bge-base-zh-lora}
        self.adapter_save_dir = base_cfg.lora_adapters_dir / model_variant.lora_adapter_dirname
        self.adapter_save_dir.mkdir(parents=True, exist_ok=True)

        # 訓練狀態追蹤
        self.global_step = 0         # gradient update 計數
        self.best_metric = 0.0       # 驗證集最佳指標 (MRR@10)
        self.best_epoch = 0
        self.patience_counter = 0    # 連續未改善的 epoch 計數

        # 模型和優化器 (setup() 後初始化)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        # 損失函數
        self.criterion = MultipleNegativesRankingLoss(temperature=0.05)

        # VRAM 監控器
        self.memory_monitor = TrainingMemoryMonitor(
            monitor_interval=vram_cfg.monitor_interval_steps,
            threshold_ratio=vram_cfg.vram_usage_threshold_ratio,
        )

        logger.info(
            f"\n{'='*55}\n"
            f"LoRATrainer 初始化\n"
            f"  模型:      {model_variant.model_name}\n"
            f"  裝置:      {self.device}\n"
            f"  batch:     {self.batch_size} × accum {self.accum_steps} = {self.batch_size * self.accum_steps} (有效)\n"
            f"  fp16:      {training_cfg.fp16}\n"
            f"  adapter →  {self.adapter_save_dir}\n"
            f"{'='*55}"
        )

    # ----------------------------------------------------------------
    # 模型初始化
    # ----------------------------------------------------------------

    def setup(self, num_training_steps: int) -> None:
        """
        載入模型、套用 LoRA、設定優化器與 Scheduler。

        Args:
            num_training_steps: 全局訓練步數 (用於 Scheduler 計算)
        """
        self._load_model_with_lora()
        self._setup_optimizer(num_training_steps)

        if self.training_cfg.fp16 and self.device.type == "cuda":
            # GradScaler: 防止 fp16 梯度 underflow
            # init_scale=2**16 是標準起始值，若發現梯度 inf/nan 會自動縮減
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2**16)
            logger.info("GradScaler 已啟用 (fp16 訓練)")
        else:
            self.scaler = None
            if not self.training_cfg.fp16:
                logger.info("fp16 未啟用，以 fp32 訓練")
            else:
                logger.info("非 CUDA 裝置，fp16 不可用，以 fp32 訓練")

    def _load_model_with_lora(self) -> None:
        """
        載入 Base Model 並套用 peft LoRA。

        ★ LoRA 套用的三個步驟:
        1. AutoModel.from_pretrained() — 載入完整 Base Model
        2. LoraConfig() — 定義 adapter 的超參數
        3. get_peft_model() — 凍結 Base Model，加入可訓練的 LoRA 矩陣

        ★ target_modules=["query", "value"] 的含義:
        peft 使用「後綴匹配」找到目標模組。
        BGE (BERT-based) 的 attention 模組路徑:
          bert.encoder.layer.{n}.attention.self.query  ← 命中 "query"
          bert.encoder.layer.{n}.attention.self.value  ← 命中 "value"
        每個命中的模組會被替換為: W_original + BA (B∈R^{d×r}, A∈R^{r×d})
        可訓練參數量 = 2 × r × d × (num_layers × 2)
          bge-small: 2 × 8 × 512 × (6 × 2) = 98,304
          bge-base:  2 × 8 × 768 × (12 × 2) = 294,912

        ★ 儲存行為:
        peft_model.save_pretrained(path) 「只儲存 LoRA 矩陣」，不儲存 Base Model。
        輸出: adapter_config.json + adapter_model.safetensors (~2-10 MB)
        """
        lora_cfg = self.training_cfg.lora
        model_name = self.variant.model_name
        cache_dir = str(self.base_cfg.base_models_dir)

        logger.info(f"載入 Base Model: {model_name}")
        log_vram("載入模型前")

        # 載入 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        # 載入 Base Model (fp16 推論節省 VRAM)
        # 注意: 若 GPU 記憶體不足以 fp16 載入，可改用 torch_dtype=torch.float32
        torch_dtype = torch.float16 if (
            self.training_cfg.fp16 and self.device.type == "cuda"
        ) else torch.float32

        base_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        )
        base_model = base_model.to(self.device)

        log_vram("Base Model 載入後 (LoRA 套用前)")

        # 建立 LoRA 設定
        peft_config = LoraConfig(
            # task_type=FEATURE_EXTRACTION: 告知 peft 這是 embedding 模型，
            # 不是分類或生成任務，影響 peft 如何初始化和儲存 adapter
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_cfg.r,                          # LoRA 秩 (低秩分解維度)
            lora_alpha=lora_cfg.lora_alpha,        # 縮放因子: 實際縮放 = alpha/r = 2.0
            lora_dropout=lora_cfg.lora_dropout,    # Dropout 正則化
            target_modules=lora_cfg.target_modules, # ["query", "value"]
            bias=lora_cfg.bias,                    # "none": 不訓練 bias 項
            inference_mode=False,                  # 訓練模式 (允許梯度流動)
        )

        # 套用 LoRA: 凍結 base model，加入可訓練的低秩矩陣
        self.model = get_peft_model(base_model, peft_config)

        log_vram("LoRA 套用後")

        # 輸出可訓練參數統計 (驗證 LoRA 正確套用)
        param_stats = count_trainable_parameters(self.model)
        expected_max_ratio = 0.05  # LoRA 可訓練參數不應超過總參數的 5%
        if param_stats["trainable_ratio"] > expected_max_ratio:
            raise RuntimeError(
                f"可訓練參數比例 ({param_stats['trainable_ratio']:.2%}) 異常！\n"
                f"LoRA 設定可能有誤，Base Model 可能未被正確凍結。"
            )

    def _setup_optimizer(self, num_training_steps: int) -> None:
        """
        設定 AdamW 優化器與帶 Warmup 的 Cosine 學習率 Scheduler。

        ★ 只優化 LoRA 參數:
        get_peft_model() 已將 Base Model 的 requires_grad 設為 False。
        AdamW 只會接收 requires_grad=True 的參數 (即 LoRA 矩陣)。
        這不只是正確性問題，也是效能問題:
        若誤將 Base Model 參數也傳入 AdamW，optimizer state 會大幅增加 VRAM。

        ★ 學習率策略:
        線性 Warmup (前 warmup_ratio 步從 0 線性升至 lr)
        + Cosine 衰減 (後續步數從 lr 衰減至 0)
        此策略在 Sentence-BERT 類型的微調中是業界標準。

        Args:
            num_training_steps: 全局訓練步數 (不含 gradient accumulation)
        """
        # 只訓練 LoRA 參數 (requires_grad=True)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        lora_param_count = sum(p.numel() for p in lora_params)
        logger.info(f"AdamW 優化器將訓練 {lora_param_count:,} 個 LoRA 參數")

        # AdamW: Adam with decoupled weight decay
        # weight_decay 應用於非 bias 與非 LayerNorm 的參數
        # 此處為簡化，統一套用 weight_decay (對 LoRA 的小型矩陣影響很小)
        self.optimizer = AdamW(
            lora_params,
            lr=self.training_cfg.learning_rate,
            weight_decay=self.training_cfg.weight_decay,
            betas=(0.9, 0.999),  # Adam 標準 beta 值
            eps=1e-8,
        )

        # 計算 warmup 步數
        num_warmup_steps = int(num_training_steps * self.training_cfg.warmup_ratio)
        logger.info(
            f"Scheduler: 線性 Warmup {num_warmup_steps} 步 "
            f"+ Cosine 衰減至 {num_training_steps} 步"
        )

        # 線性 Warmup + Cosine 衰減 lambda 函數
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                # 線性從 0 升至 1.0
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine 從 1.0 衰減至 0.0
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    # ----------------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------------

    def _encode(
        self,
        texts: list,
        no_grad: bool = False,
    ) -> torch.Tensor:
        """
        Tokenize 並 encode 文字，回傳 L2 正規化的 embedding。

        ★ 注意 no_grad 參數:
        - 訓練時: no_grad=False，梯度圖保留，用於 backward
        - 評估時: no_grad=True，節省記憶體

        ★ autocast 的使用:
        autocast context 由外部 (training step) 管理，
        此函數不自行包 autocast，確保與 GradScaler 正確配合。

        Args:
            texts: 文字列表
            no_grad: 是否在 torch.no_grad() 下執行

        Returns:
            (B, D) L2 正規化的 sentence embeddings
        """
        # Tokenize (CPU → GPU 由 to(self.device) 完成)
        encoded = self.tokenizer(
            texts,
            max_length=self.training_cfg.max_seq_length,
            padding=True,          # 批次中動態 pad 到最長序列
            truncation=True,       # 超過 max_seq_length 的文字截斷
                                   # ★ 妥協: 截斷會丟失長文本的尾部語意
                                   # 建議語料段落長度控制在 200 字以內
            return_tensors="pt",
        )

        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        @contextmanager
        def _maybe_no_grad():
            if no_grad:
                with torch.no_grad():
                    yield
            else:
                yield

        with _maybe_no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        token_embeddings = outputs.last_hidden_state  # (B, L, D)
        embeddings = mean_pool_and_normalize(token_embeddings, attention_mask)

        return embeddings

    # ----------------------------------------------------------------
    # 訓練步驟
    # ----------------------------------------------------------------

    def _train_step(
        self,
        queries: list,
        positives: list,
        hard_negatives: list,
    ) -> Tuple[float, float]:
        """
        單個 forward + backward 步驟。
        此函數對應一次 DataLoader batch，不是一次 gradient update。

        ★ 三路分開 forward 的設計理由:
        若將 queries + positives + negatives 合併為一個大 batch (B×3, L) forward，
        VRAM 峰值 = (B×3) × L × D × num_layers 的啟動值，比分開 forward 高 3 倍。
        分開 forward (每次 B × L) 讓啟動值在每次 forward 後可以被 GC 回收，
        代價是 3 次 forward 有約 10-15% 的 kernel launch overhead。
        在 RTX 4060 8GB 的 VRAM 限制下，優先選擇記憶體安全的方案。

        ★ fp16 autocast 範圍:
        autocast 只應包覆 forward pass，不應包覆 optimizer.step()。
        GradScaler 會在 optimizer.step() 前自動 unscale 梯度。

        Returns:
            (loss_value, accuracy_value) 浮點數
        """
        use_amp = (
            self.scaler is not None
            and self.training_cfg.fp16
            and self.device.type == "cuda"
        )

        if use_amp:
            # fp16 混合精度: forward 在半精度下執行
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                q_embs = self._encode(queries,        no_grad=False)
                p_embs = self._encode(positives,      no_grad=False)
                n_embs = self._encode(hard_negatives, no_grad=False)
                loss, accuracy = self.criterion(q_embs, p_embs, n_embs)
                # ★ Gradient Accumulation: loss 除以累積步數
                # 使多步累積的等效梯度 = 單步大 batch 的梯度
                loss = loss / self.accum_steps

            # GradScaler backward: 自動放大 loss 以防梯度 underflow
            self.scaler.scale(loss).backward()
        else:
            # fp32 路徑 (CPU 或未啟用 fp16)
            q_embs = self._encode(queries,        no_grad=False)
            p_embs = self._encode(positives,      no_grad=False)
            n_embs = self._encode(hard_negatives, no_grad=False)
            loss, accuracy = self.criterion(q_embs, p_embs, n_embs)
            loss = loss / self.accum_steps
            loss.backward()

        return loss.item() * self.accum_steps, accuracy.item()

    def _optimizer_step(self) -> None:
        """
        執行一次 optimizer update。
        在每 accum_steps 個 forward/backward 後呼叫一次。

        步驟:
        1. unscale_ : 將 GradScaler 放大過的梯度還原回真實值
        2. clip_grad_norm_: 梯度裁剪 (防止梯度爆炸，fp16 特別重要)
        3. scaler.step(): optimizer.step() (若梯度有 inf/nan 則自動跳過)
        4. scaler.update(): 調整 scale factor (若本次有 inf/nan 則縮小)
        5. scheduler.step(): 更新學習率
        6. zero_grad(): 清空梯度，準備下一個 accumulation 週期
        """
        use_amp = self.scaler is not None

        if use_amp:
            # Step 1: unscale 梯度 (必須在 clip_grad_norm_ 之前)
            self.scaler.unscale_(self.optimizer)

        # Step 2: 梯度裁剪
        # 只裁剪 LoRA 參數的梯度，因為 base model 梯度為 None
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(lora_params, self.training_cfg.max_grad_norm)

        if use_amp:
            # Step 3+4: optimizer step + scale update
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Step 5: 學習率更新
        self.scheduler.step()

        # Step 6: 清空梯度 (set_to_none=True 比 zero_grad() 更節省記憶體)
        self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1

    # ----------------------------------------------------------------
    # 訓練 Epoch
    # ----------------------------------------------------------------

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        執行一個 training epoch。

        Args:
            train_loader: 訓練資料的 DataLoader
            epoch: 當前 epoch 號 (從 1 開始)

        Returns:
            dict: {"loss": avg_loss, "accuracy": avg_accuracy, "lr": current_lr}
        """
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        log_vram(f"epoch {epoch} 開始")

        for batch_idx, (queries, positives, hard_negatives) in enumerate(train_loader):
            try:
                loss, accuracy = self._train_step(queries, positives, hard_negatives)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    handle_oom_error(self.batch_size)
                    raise  # OOM 後終止，讓使用者調整設定重試
                raise

            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

            # 每累積 accum_steps 個 batch，執行一次 optimizer update
            if (batch_idx + 1) % self.accum_steps == 0:
                self._optimizer_step()
                self.memory_monitor.step(
                    self.global_step,
                    extra_label=f"epoch={epoch}"
                )

                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss = total_loss / num_batches
                avg_acc  = total_accuracy / num_batches

                if self.global_step % 20 == 0:
                    logger.info(
                        f"epoch={epoch} | step={self.global_step} | "
                        f"loss={avg_loss:.4f} | acc={avg_acc:.3f} | "
                        f"lr={current_lr:.2e}"
                    )

        # 處理最後一個不完整的 accumulation 週期
        # (使用 drop_last=True 的 DataLoader 時此情況不應出現，保險起見保留)
        remaining = num_batches % self.accum_steps
        if remaining > 0 and num_batches > 0:
            self._optimizer_step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc  = total_accuracy / max(num_batches, 1)
        current_lr = self.scheduler.get_last_lr()[0]

        logger.info(
            f"\n[Epoch {epoch} 訓練完成]\n"
            f"  avg loss:     {avg_loss:.4f}\n"
            f"  avg accuracy: {avg_acc:.3f}\n"
            f"  lr:           {current_lr:.2e}\n"
            f"  global steps: {self.global_step}"
        )

        return {"loss": avg_loss, "accuracy": avg_acc, "lr": current_lr}

    # ----------------------------------------------------------------
    # 評估
    # ----------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, val_dataset: ValDataset) -> Dict[str, float]:
        """
        在驗證集上計算 MRR@10 與 NDCG@10。

        評估流程:
        1. 批次編碼所有 corpus 段落 → corpus_embeddings (N, D)
        2. 批次編碼所有 queries      → query_embeddings  (Q, D)
        3. 計算相似度矩陣            → sims (Q, N)
        4. 對每個 query，按相似度排序 corpus，計算 MRR@10 / NDCG@10
        5. 回傳平均值

        評估策略說明:
        此評估使用的 corpus 僅包含 val.jsonl 中的段落 (positives + hard_negatives)，
        並非完整的 domain corpus。MRR/NDCG 數值偏高，僅用於追蹤「相對改善」趨勢。

        Args:
            val_dataset: ValDataset 物件 (含 queries, corpus, relevant_docs)

        Returns:
            dict: {"mrr@10": ..., "ndcg@10": ..., "num_queries": ...}
        """
        self.model.eval()
        log_vram("評估開始")

        eval_batch_size = min(64, self.batch_size * 4)  # 評估時可用較大 batch (無梯度)

        # ----------------------------------------------------------------
        # 1. 編碼 corpus
        # ----------------------------------------------------------------
        corpus_ids   = list(val_dataset.corpus.keys())
        corpus_texts = [val_dataset.corpus[cid] for cid in corpus_ids]

        corpus_embeddings_list = []
        for i in range(0, len(corpus_texts), eval_batch_size):
            batch = corpus_texts[i: i + eval_batch_size]
            embs = self._encode(batch, no_grad=True)
            corpus_embeddings_list.append(embs.cpu().float())

        corpus_embeddings = torch.cat(corpus_embeddings_list, dim=0)  # (N, D)
        logger.debug(f"Corpus embeddings: {corpus_embeddings.shape}")

        # ----------------------------------------------------------------
        # 2. 編碼 queries
        # ----------------------------------------------------------------
        query_ids   = list(val_dataset.queries.keys())
        query_texts = [val_dataset.queries[qid] for qid in query_ids]

        query_embeddings_list = []
        for i in range(0, len(query_texts), eval_batch_size):
            batch = query_texts[i: i + eval_batch_size]
            embs = self._encode(batch, no_grad=True)
            query_embeddings_list.append(embs.cpu().float())

        query_embeddings = torch.cat(query_embeddings_list, dim=0)  # (Q, D)

        # ----------------------------------------------------------------
        # 3+4. 計算相似度 + MRR/NDCG
        # ----------------------------------------------------------------
        # (Q, N) 相似度矩陣 (在 CPU 上計算，避免大矩陣佔用 GPU VRAM)
        sim_matrix = torch.mm(query_embeddings, corpus_embeddings.T).numpy()

        mrr_scores  = []
        ndcg_scores = []

        for q_idx, q_id in enumerate(query_ids):
            relevant_ids = val_dataset.relevant_docs.get(q_id, set())
            if not relevant_ids:
                continue

            # 按相似度降序排列 corpus (numpy argsort 預設升序，用 [::-1] 反轉)
            sims = sim_matrix[q_idx]
            ranked_indices = np.argsort(sims)[::-1]
            ranked_doc_ids = [corpus_ids[i] for i in ranked_indices]

            mrr_scores.append(compute_mrr_at_k(ranked_doc_ids, relevant_ids, k=10))
            ndcg_scores.append(compute_ndcg_at_k(ranked_doc_ids, relevant_ids, k=10))

        if not mrr_scores:
            logger.warning("評估資料集無有效的 query-relevant 配對，無法計算指標")
            return {"mrr@10": 0.0, "ndcg@10": 0.0, "num_queries": 0}

        avg_mrr  = float(np.mean(mrr_scores))
        avg_ndcg = float(np.mean(ndcg_scores))

        logger.info(
            f"\n[評估結果] step={self.global_step}\n"
            f"  MRR@10:  {avg_mrr:.4f}\n"
            f"  NDCG@10: {avg_ndcg:.4f}\n"
            f"  評估 queries: {len(mrr_scores)}"
        )

        return {
            "mrr@10":      avg_mrr,
            "ndcg@10":     avg_ndcg,
            "num_queries": len(mrr_scores),
        }

    # ----------------------------------------------------------------
    # 儲存 LoRA Adapter
    # ----------------------------------------------------------------

    def save_adapter(self, label: str = "") -> Path:
        """
        ★ 儲存 LoRA Adapter 權重 (嚴格禁止儲存 Base Model)。

        peft 的 save_pretrained() 行為:
        - 儲存 adapter_config.json: 記錄 LoRA 超參數 (r, lora_alpha, target_modules...)
        - 儲存 adapter_model.safetensors: LoRA 矩陣 A 與 B 的權重
        - 「不儲存」Base Model 的權重 (這正是 LoRA 的空間優勢)
        - 輸出大小: ~2 MB (small) 到 ~10 MB (base)
          對比全參數微調: ~48 MB (small) 到 ~204 MB (base)

        推論時的載入方式:
            from transformers import AutoModel
            from peft import PeftModel
            base = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
            model = PeftModel.from_pretrained(base, adapter_path)
            # 可選: merged = model.merge_and_unload()  ← 合併後速度更快

        Args:
            label: 子目錄標籤 (例如 "best", "epoch-3", "final")

        Returns:
            儲存路徑
        """
        if label:
            save_path = self.adapter_save_dir / label
        else:
            save_path = self.adapter_save_dir

        save_path.mkdir(parents=True, exist_ok=True)

        # 儲存 LoRA adapter (不儲存 base model)
        # peft 的 get_peft_model() 包裝後，save_pretrained 預設行為就是只存 adapter
        self.model.save_pretrained(str(save_path))

        # 同時儲存 tokenizer (推論時需要)
        self.tokenizer.save_pretrained(str(save_path))

        # 驗證輸出大小 (應遠小於完整模型)
        total_size_mb = sum(
            f.stat().st_size for f in save_path.rglob("*") if f.is_file()
        ) / (1024 ** 2)

        logger.info(
            f"★ LoRA Adapter 已儲存: {save_path}\n"
            f"  大小: {total_size_mb:.1f} MB  "
            f"(完整模型大小: ~{self.variant.estimated_vram_mb} MB)\n"
            f"  空間節省: {(1 - total_size_mb / self.variant.estimated_vram_mb) * 100:.0f}%"
        )

        return save_path

    # ----------------------------------------------------------------
    # 主訓練迴圈
    # ----------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_dataset: ValDataset,
    ) -> Dict[str, float]:
        """
        完整訓練流程: 多 Epoch 訓練 + Early Stopping。

        Early Stopping 邏輯:
        - 主要監控指標: MRR@10 (on val set)
        - 若連續 early_stopping_patience 個 epoch 無改善，停止訓練
        - 最佳 adapter 會被複製到 adapter_save_dir/best/

        Args:
            train_loader: 訓練 DataLoader
            val_dataset:  驗證用 ValDataset

        Returns:
            最終評估結果 dict
        """
        logger.info(
            f"\n{'='*55}\n"
            f"開始訓練: {self.variant.short_name} ({self.variant.model_name})\n"
            f"  訓練步數/epoch: {len(train_loader)}\n"
            f"  有效 batch:     {self.batch_size * self.accum_steps}\n"
            f"  總 epochs:      {self.training_cfg.num_epochs}\n"
            f"{'='*55}"
        )

        best_result: Dict[str, float] = {}

        for epoch in range(1, self.training_cfg.num_epochs + 1):
            # ---- 訓練 ----
            train_metrics = self.train_epoch(train_loader, epoch)

            # ---- 評估 ----
            eval_metrics = self.evaluate(val_dataset)
            current_mrr = eval_metrics["mrr@10"]

            # ---- 儲存當前 epoch 的 checkpoint ----
            self.save_adapter(label=f"epoch-{epoch}")

            # ---- 判斷是否為最佳模型 ----
            if current_mrr > self.best_metric:
                self.best_metric = current_mrr
                self.best_epoch  = epoch
                self.patience_counter = 0
                best_result = eval_metrics

                # 複製到 best/ 子目錄
                best_dir = self.adapter_save_dir / "best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(
                    str(self.adapter_save_dir / f"epoch-{epoch}"),
                    str(best_dir),
                )
                logger.info(
                    f"★ 新最佳模型！epoch={epoch} | MRR@10={current_mrr:.4f}"
                )
            else:
                self.patience_counter += 1
                logger.info(
                    f"  未改善 ({self.patience_counter}/{self.training_cfg.early_stopping_patience})"
                    f" | 最佳 MRR@10={self.best_metric:.4f} @ epoch {self.best_epoch}"
                )

            # ---- Early Stopping 判斷 ----
            if self.patience_counter >= self.training_cfg.early_stopping_patience:
                logger.info(
                    f"\n[Early Stopping] 連續 {self.patience_counter} 個 epoch 無改善，"
                    f"在 epoch {epoch} 停止訓練。\n"
                    f"最佳結果: epoch {self.best_epoch} | MRR@10={self.best_metric:.4f}"
                )
                break

        # ---- 最終報告 ----
        logger.info(
            f"\n{'='*55}\n"
            f"訓練完成: {self.variant.short_name}\n"
            f"  最佳 epoch:    {self.best_epoch}\n"
            f"  最佳 MRR@10:  {self.best_metric:.4f}\n"
            f"  最佳 NDCG@10: {best_result.get('ndcg@10', 0):.4f}\n"
            f"  Adapter 路徑: {self.adapter_save_dir / 'best'}\n"
            f"{'='*55}"
        )

        self.memory_monitor.report_peak()

        return best_result
