# ============================================================
# scripts/run_training.py — LoRA 微調入口腳本
#
# 功能: 依 --model 參數對 bge-small 或 bge-base 執行 LoRA 微調，
#       讀取 Step 2 產出的 train.jsonl + val.jsonl，
#       訓練結束後僅儲存 LoRA Adapter (~2-10 MB)。
#
# 消融實驗 (Ablation Study) 執行方式:
#   python scripts/run_training.py --model small  # 訓練 bge-small-zh-lora
#   python scripts/run_training.py --model base   # 訓練 bge-base-zh-lora
#   # 或透過 Makefile:
#   make train-small
#   make train-base
#
# 記憶體使用估算 (RTX 4060 Ti 8GB):
#   bge-small: ~308 MB  (遠低於 2000 MB 預算)
#   bge-base:  ~644 MB  (遠低於 3500 MB 預算)
#   若 OOM，請先確認 Ollama 服務已停止 (make stop-ollama)
# ============================================================

import argparse
import math
import random
import sys
from pathlib import Path

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from loguru import logger

from configs.base_config import BaseConfig
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from configs.vram_config import VRAMConfig

from data_pipeline.vram_guard import VRAMGuard
from training.dataset import ValDataset, create_train_dataloader
from training.lora_trainer import LoRATrainer
from training.memory_utils import log_vram


def setup_logger(log_level: str, log_file: Path) -> None:
    """設定 loguru 日誌 (stdout + 檔案)。"""
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        rotation="100 MB",
        retention=3,
    )


def set_seeds(seed: int) -> None:
    """固定所有隨機種子，確保實驗可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN deterministic 模式: 速度稍慢但保證可重現
    # 若優先考慮速度，可設為 False (但結果可能有細微差異)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain-Adaptive Retrieval System — LoRA 微調",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
執行範例:
  # 訓練 bge-small (快，~15 分鐘)
  python scripts/run_training.py --model small

  # 訓練 bge-base (較慢，~30 分鐘)
  python scripts/run_training.py --model base

  # 快速 smoke test (只跑 2 epoch，小資料)
  python scripts/run_training.py --model small --max-samples 100 --num-epochs 2

  # 使用對稱 MNRL 損失 (雙向梯度)
  python scripts/run_training.py --model small --symmetric-loss
        """,
    )

    parser.add_argument(
        "--model",
        choices=["small", "base"],
        required=True,
        help="要微調的模型變體: small=bge-small-zh-v1.5 | base=bge-base-zh-v1.5",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="train.jsonl 路徑 (預設: data/processed/train.jsonl)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="val.jsonl 路徑 (預設: data/processed/val.jsonl)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多載入的訓練樣本數 (開發測試用，None = 全部)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="訓練 epoch 數 (覆蓋 configs/training_config.py 的設定)",
    )
    parser.add_argument(
        "--symmetric-loss",
        action="store_true",
        help="啟用對稱 MNRL (雙向梯度，對少量資料有幫助)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="停用 fp16，以 fp32 訓練 (除錯用，VRAM 佔用加倍)",
    )
    parser.add_argument(
        "--skip-vram-check",
        action="store_true",
        help="跳過 VRAM 安全斷言 (若 Ollama 無法停止時的臨時措施，不建議)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日誌等級",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ----------------------------------------------------------------
    # 1. 載入設定
    # ----------------------------------------------------------------
    base_cfg     = BaseConfig()
    model_cfg    = ModelConfig()
    training_cfg = TrainingConfig()
    vram_cfg     = VRAMConfig()

    # 命令列覆蓋設定
    if args.num_epochs is not None:
        training_cfg.num_epochs = args.num_epochs
    if args.no_fp16:
        training_cfg.fp16 = False

    # 設定日誌
    log_level = args.log_level or base_cfg.log_level
    log_file = base_cfg.project_root / "outputs" / f"training_{args.model}.log"
    setup_logger(log_level, log_file)

    # 固定隨機種子
    set_seeds(base_cfg.seed)

    # 取得模型變體
    model_variant = model_cfg.get_variant(args.model)

    logger.info(
        f"\n{'='*60}\n"
        f"Domain-Adaptive Retrieval System — LoRA 微調\n"
        f"  模型變體:  {model_variant.short_name} ({model_variant.model_name})\n"
        f"  參數量:    {model_variant.param_count_million}M\n"
        f"  裝置:      {base_cfg.device}\n"
        f"  batch:     {training_cfg.get_batch_size(args.model)} (真實) × "
        f"{training_cfg.get_gradient_accumulation_steps(args.model)} (累積) = "
        f"{training_cfg.get_batch_size(args.model) * training_cfg.get_gradient_accumulation_steps(args.model)} (有效)\n"
        f"  fp16:      {training_cfg.fp16}\n"
        f"  epochs:    {training_cfg.num_epochs}\n"
        f"  seed:      {base_cfg.seed}\n"
        f"{'='*60}"
    )

    # ----------------------------------------------------------------
    # 2. VRAM 安全斷言: 確認 Ollama 已停止，VRAM 已清空
    # ----------------------------------------------------------------
    if not args.skip_vram_check:
        guard = VRAMGuard(post_release_max_mb=vram_cfg.post_release_max_mb)
        phase_key = f"training_{args.model}"
        budget_mb = vram_cfg.get_phase_budget(phase_key)
        logger.info(f"VRAM 預算 (phase {phase_key}): {budget_mb} MB")

        # 若 GPU 上殘留超過 200 MB，很可能 Ollama 或 Mining 模型未釋放
        guard.assert_safe_to_proceed(f"LoRA Training ({args.model})")
    else:
        logger.warning("已跳過 VRAM 安全斷言 (--skip-vram-check)")

    log_vram("訓練開始前")

    # ----------------------------------------------------------------
    # 3. 資料路徑
    # ----------------------------------------------------------------
    train_path = Path(args.train_data) if args.train_data else (
        base_cfg.data_processed_dir / "train.jsonl"
    )
    val_path = Path(args.val_data) if args.val_data else (
        base_cfg.data_processed_dir / "val.jsonl"
    )

    for path in [train_path, val_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"資料檔案不存在: {path}\n"
                f"請先執行: make data-process"
            )

    # ----------------------------------------------------------------
    # 4. 建立 DataLoader
    # ----------------------------------------------------------------
    batch_size = training_cfg.get_batch_size(args.model)
    accum_steps = training_cfg.get_gradient_accumulation_steps(args.model)

    train_loader = create_train_dataloader(
        train_path=train_path,
        query_instruction=model_cfg.query_instruction,
        batch_size=batch_size,
        num_workers=training_cfg.num_workers,
        seed=base_cfg.seed,
        max_samples=args.max_samples,
    )

    val_dataset = ValDataset(val_path=val_path)

    # ----------------------------------------------------------------
    # 5. 計算總訓練步數 (用於 Scheduler)
    # ----------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    # gradient update 次數 per epoch = steps / accum_steps (取整)
    updates_per_epoch = math.ceil(steps_per_epoch / accum_steps)
    total_updates = updates_per_epoch * training_cfg.num_epochs
    num_warmup = int(total_updates * training_cfg.warmup_ratio)

    logger.info(
        f"\n訓練步數計算:\n"
        f"  DataLoader steps/epoch: {steps_per_epoch}\n"
        f"  Gradient updates/epoch: {updates_per_epoch}\n"
        f"  總 gradient updates:    {total_updates}\n"
        f"  Warmup steps:           {num_warmup} ({training_cfg.warmup_ratio:.0%})"
    )

    # ----------------------------------------------------------------
    # 6. 初始化 Trainer
    # ----------------------------------------------------------------
    trainer = LoRATrainer(
        model_variant=model_variant,
        base_cfg=base_cfg,
        training_cfg=training_cfg,
        vram_cfg=vram_cfg,
        symmetric_loss=args.symmetric_loss,
    )
    trainer.setup(num_training_steps=total_updates)

    log_vram("模型載入 + LoRA 套用後")

    # ----------------------------------------------------------------
    # 7. 訓練
    # ----------------------------------------------------------------
    try:
        best_metrics = trainer.train(
            train_loader=train_loader,
            val_dataset=val_dataset,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                f"\nCUDA OOM 終止訓練！\n"
                f"請嘗試以下措施:\n"
                f"  1. 確認 Ollama 服務已停止: make stop-ollama\n"
                f"  2. 降低 batch_size: 修改 configs/training_config.py\n"
                f"  3. 縮短序列長度: max_seq_length 從 256 → 128\n"
                f"  4. 使用 --no-fp16 後重試 (診斷用，非解決方案)"
            )
        raise

    # ----------------------------------------------------------------
    # 8. 最終摘要
    # ----------------------------------------------------------------
    best_adapter_path = base_cfg.lora_adapters_dir / model_variant.lora_adapter_dirname / "best"
    logger.info(
        f"\n{'='*60}\n"
        f"[{args.model}] 訓練完成\n"
        f"  MRR@10:          {best_metrics.get('mrr@10', 0):.4f}\n"
        f"  NDCG@10:         {best_metrics.get('ndcg@10', 0):.4f}\n"
        f"  最佳 Adapter:    {best_adapter_path}\n"
        f"\n  ★ 下一步: 使用 make retrieve 執行混合檢索\n"
        f"            或 make train-base 訓練 base 模型\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
