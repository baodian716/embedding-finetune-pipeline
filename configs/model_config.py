# ============================================================
# configs/model_config.py — 模型設定
# 定義 bge-small-zh-v1.5 與 bge-base-zh-v1.5 的參數
# 用於 Ablation Study (消融實驗) 的雙模型對比
# ============================================================

from dataclasses import dataclass


@dataclass
class ModelVariant:
    """
    單一模型變體的完整參數。
    每個欄位皆標註其對 VRAM 的影響，供 VRAMGuard 參考。
    """
    # HuggingFace 模型識別碼
    model_name: str
    # 簡短標籤 (用於圖表、日誌、檔案命名)
    short_name: str
    # 模型參數量 (百萬)，用於估算 VRAM 佔用
    param_count_million: float
    # 隱藏層維度 (embedding 輸出維度)
    hidden_size: int
    # Transformer 層數
    num_layers: int
    # 該模型 fp16 推論時的預估 VRAM (MB)
    estimated_vram_mb: int
    # LoRA adapter 儲存子目錄名稱
    lora_adapter_dirname: str


@dataclass
class ModelConfig:
    """
    雙模型對比實驗設定。
    small 與 base 兩個變體的參數差異是消融實驗的核心觀察對象。
    """

    # ---- bge-small-zh-v1.5 ----
    # 24M 參數，512 維 embedding，6 層 Transformer
    # 優勢: VRAM 佔用極低 (~48MB fp16)，訓練快
    # 劣勢: 表達能力受限於較小的隱藏層與層數
    small: ModelVariant = None

    # ---- bge-base-zh-v1.5 ----
    # 102M 參數，768 維 embedding，12 層 Transformer
    # 優勢: 表達能力較強，通常在 benchmark 上勝出
    # 劣勢: VRAM 佔用約為 small 的 4 倍，訓練較慢
    base: ModelVariant = None

    # ---- 共用設定 ----
    # 最大序列長度 (超過此長度的輸入將被截斷)
    # 256 是在 8GB VRAM 下的安全上限；原始 BGE 訓練使用 512
    # 妥協: 較長的段落可能因截斷而遺失尾部語意資訊
    max_seq_length: int = 256

    # Tokenizer 是否添加 [CLS] 前綴指令
    # BGE v1.5 系列在 query 編碼時需加上 "为这个句子生成表示以用于检索相关文章："
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    def __post_init__(self):
        if self.small is None:
            self.small = ModelVariant(
                model_name="BAAI/bge-small-zh-v1.5",
                short_name="small",
                param_count_million=24.0,
                hidden_size=512,
                num_layers=6,
                estimated_vram_mb=48,
                lora_adapter_dirname="bge-small-zh-lora",
            )
        if self.base is None:
            self.base = ModelVariant(
                model_name="BAAI/bge-base-zh-v1.5",
                short_name="base",
                param_count_million=102.0,
                hidden_size=768,
                num_layers=12,
                estimated_vram_mb=204,
                lora_adapter_dirname="bge-base-zh-lora",
            )

    def get_variant(self, name: str) -> ModelVariant:
        """依名稱取得模型變體，支援 'small' 或 'base'。"""
        variants = {"small": self.small, "base": self.base}
        if name not in variants:
            raise ValueError(f"未知的模型變體: {name}，僅支援 'small' 或 'base'")
        return variants[name]
