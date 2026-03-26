# ============================================================
# configs/base_config.py — 全域基礎設定
# 路徑、裝置、隨機種子等不隨實驗改變的固定參數
# ============================================================

import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數 (若存在)
load_dotenv()


@dataclass
class BaseConfig:
    """
    全域基礎設定。
    所有路徑皆以專案根目錄為基準，透過 Path 物件管理以相容不同作業系統。
    """

    # ---- 專案根目錄 ----
    # 自動偵測: 此檔案位於 configs/，上一層即為專案根目錄
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    # ---- 資料路徑 ----
    data_raw_dir: Path = field(init=False)
    data_synthetic_dir: Path = field(init=False)
    data_hard_negatives_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)

    # ---- 模型路徑 ----
    base_models_dir: Path = field(init=False)
    lora_adapters_dir: Path = field(init=False)

    # ---- 輸出路徑 ----
    output_charts_dir: Path = field(init=False)
    output_embeddings_dir: Path = field(init=False)
    output_reports_dir: Path = field(init=False)

    # ---- 裝置設定 ----
    device: str = field(
        default_factory=lambda: "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    )

    # ---- 隨機種子 (確保實驗可重現) ----
    seed: int = field(
        default_factory=lambda: int(os.getenv("GLOBAL_SEED", "42"))
    )

    # ---- Ollama 設定 ----
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    # llama3.2 (3B, q4_K_M) 約佔 2.0 GB VRAM；
    # 原 qwen2.5:7b 約佔 4.5 GB，對 8GB 裝置在資料管線與訓練階段餘量不足
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )

    # ---- 日誌等級 ----
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def __post_init__(self):
        """根據 project_root 自動推導所有子路徑。"""
        self.data_raw_dir = self.project_root / "data" / "raw"
        self.data_synthetic_dir = self.project_root / "data" / "synthetic"
        self.data_hard_negatives_dir = self.project_root / "data" / "hard_negatives"
        self.data_processed_dir = self.project_root / "data" / "processed"

        self.base_models_dir = self.project_root / "models" / "base_models"
        self.lora_adapters_dir = self.project_root / "models" / "lora_adapters"

        self.output_charts_dir = self.project_root / "outputs" / "charts"
        self.output_embeddings_dir = self.project_root / "outputs" / "embeddings"
        self.output_reports_dir = self.project_root / "outputs" / "reports"
