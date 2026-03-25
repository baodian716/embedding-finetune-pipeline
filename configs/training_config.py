# ============================================================
# configs/training_config.py — LoRA 微調訓練設定
# 所有超參數皆針對 RTX 4060 8GB VRAM 調校
# 嚴禁全參數微調 — 僅透過 peft LoRA adapter 更新部分權重
# ============================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """
    LoRA (Low-Rank Adaptation) 超參數。
    這些參數控制 adapter 的規模與正則化強度。
    """
    # LoRA 秩 (rank): 低秩分解的維度
    # r=8 是在 adapter 體積 (~2-10MB) 與表達能力之間的平衡點
    # 增大 r 可提升表達能力，但同時增加 VRAM 佔用與過擬合風險
    r: int = 8

    # LoRA 縮放因子: 實際縮放比為 lora_alpha / r
    # alpha=16, r=8 → 縮放比 2.0，這是 BGE 微調的常見設定
    lora_alpha: int = 16

    # Dropout: LoRA 層的隨機失活率，用於防止過擬合
    lora_dropout: float = 0.1

    # 目標模組: 只在這些模組上掛載 LoRA adapter
    # "query" 和 "value" 是 Attention 機制中最影響語意表徵的投影層
    # 不掛載 "key" 和 "output" 是為了進一步節省 VRAM
    target_modules: List[str] = field(
        default_factory=lambda: ["query", "value"]
    )

    # 偏置項處理策略: "none" 表示不訓練任何 bias
    bias: str = "none"


@dataclass
class TrainingConfig:
    """
    訓練迴圈設定。
    針對 bge-small 和 bge-base 分別定義安全的 batch size。
    """

    # ---- LoRA 超參數 ----
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # ---- 學習率 ----
    # 2e-5 是 Sentence-BERT 微調的標準起點
    # 過高 (>1e-4) 在 fp16 下容易導致梯度爆炸
    learning_rate: float = 2e-5

    # ---- 訓練輪數 ----
    # 3-5 epochs 通常足以讓 LoRA adapter 收斂
    # 過多 epochs 在小資料集上容易過擬合
    num_epochs: int = 3

    # ---- Batch Size 設定 (防 OOM 核心) ----
    # bge-small: 24M 參數，VRAM 餘裕充足，可用較大 batch
    batch_size_small: int = 8
    gradient_accumulation_steps_small: int = 4
    # 有效 batch size = 8 × 4 = 32

    # bge-base: 102M 參數，VRAM 較緊，必須縮小 batch
    batch_size_base: int = 4
    gradient_accumulation_steps_base: int = 8
    # 有效 batch size = 4 × 8 = 32
    # 妥協: 真實 batch 僅 4，in-batch negatives 數量受限
    #        對比學習品質低於大 VRAM 環境，但透過 hard negatives 補償

    # ---- 序列長度 ----
    max_seq_length: int = 256

    # ---- 混合精度 ----
    # fp16=True 將模型權重與梯度以半精度儲存，VRAM 佔用減半
    # 需搭配 GradScaler 防止梯度下溢 (underflow)
    fp16: bool = True

    # ---- 梯度裁剪 ----
    # 限制梯度範數上限，防止訓練不穩定
    max_grad_norm: float = 1.0

    # ---- Warmup ----
    # 前 10% 的步數使用線性 warmup，避免初始學習率過高
    warmup_ratio: float = 0.1

    # ---- 權重衰減 ----
    weight_decay: float = 0.01

    # ---- DataLoader 工作執行緒 ----
    # Windows 環境下 num_workers > 0 可能導致問題，
    # Docker (Linux) 環境建議設為 2
    num_workers: int = 2

    # ---- 驗證與儲存策略 ----
    # 每 N 步執行一次驗證
    eval_steps: int = 100
    # 每 N 步儲存一次 checkpoint
    save_steps: int = 100
    # 最多保留的 checkpoint 數量 (節省磁碟空間)
    save_total_limit: int = 2

    # ---- 早停 (Early Stopping) ----
    # 連續 N 次驗證無改善則停止訓練
    early_stopping_patience: int = 3

    def get_batch_size(self, model_variant: str) -> int:
        """依模型變體回傳安全的 batch size。"""
        if model_variant == "small":
            return self.batch_size_small
        elif model_variant == "base":
            return self.batch_size_base
        else:
            raise ValueError(f"未知模型變體: {model_variant}")

    def get_gradient_accumulation_steps(self, model_variant: str) -> int:
        """依模型變體回傳梯度累積步數。"""
        if model_variant == "small":
            return self.gradient_accumulation_steps_small
        elif model_variant == "base":
            return self.gradient_accumulation_steps_base
        else:
            raise ValueError(f"未知模型變體: {model_variant}")
