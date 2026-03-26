# ============================================================
# scripts/run_data_pipeline.py — 資料管線入口腳本
#
# 此腳本是整個 data_pipeline/ 模組的統一執行入口。
# 支援三個子命令 (--phase 參數):
#
#   synthetic  — 階段 A: 呼叫 Ollama LLM 生成合成 query-passage 配對
#   mining     — 階段 B: Hard Negative Mining (需 Ollama 已停止)
#   process    — 後處理: 清洗、去重、切分 train/val
#   all        — 依序執行上述三個階段 (pipeline 完整版)
#
# VRAM 時序說明:
#   [A] Ollama 佔 GPU → Python 零 VRAM
#   [停止 Ollama: make stop-ollama]
#   [B] Embedding 模型佔 GPU → mining 完成後立即釋放
#   [後處理] 純 CPU，零 VRAM
#
# 使用範例:
#   # 在容器內執行
#   python scripts/run_data_pipeline.py --phase synthetic
#   make stop-ollama  # 在 Host 上執行 (切換到 Makefile 的 target)
#   python scripts/run_data_pipeline.py --phase mining
#   python scripts/run_data_pipeline.py --phase process
#
#   # 或直接使用 Makefile 快捷指令 (建議)
#   make data-synthetic
#   make stop-ollama
#   make data-mining
#   make data-process
# ============================================================

import argparse
import sys
from pathlib import Path

# 確保專案根目錄在 sys.path 中 (Docker 環境中 PYTHONPATH 已設定，本地開發需要)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger

from configs.base_config import BaseConfig
from configs.model_config import ModelConfig
from configs.retrieval_config import RetrievalConfig
from configs.vram_config import VRAMConfig

from data_pipeline.synthetic_generator import run_synthetic_generation
from data_pipeline.hard_negative_miner import run_hard_negative_mining
from data_pipeline.data_processor import run_data_processing
from data_pipeline.vram_guard import VRAMGuard


def setup_logger(log_level: str) -> None:
    """設定 loguru 日誌格式。"""
    logger.remove()  # 移除預設 handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    # 同時輸出到檔案 (方便後續審查)
    log_file = project_root / "outputs" / "data_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_file),
        level="DEBUG",  # 檔案記錄 DEBUG 以上的所有訊息
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        rotation="50 MB",  # 超過 50MB 自動分割
        retention=3,       # 保留最近 3 個日誌檔案
    )


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="Domain-Adaptive Retrieval System — 資料管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
執行範例:
  # 階段 A: LLM 合成資料 (Ollama 需已啟動)
  python scripts/run_data_pipeline.py --phase synthetic

  # 停止 Ollama (在 Host 上執行)
  make stop-ollama

  # 階段 B: Hard Negative Mining (Ollama 必須已停止)
  python scripts/run_data_pipeline.py --phase mining

  # 後處理
  python scripts/run_data_pipeline.py --phase process

  # 開發測試: 只處理前 50 個段落
  python scripts/run_data_pipeline.py --phase synthetic --max-passages 50
        """,
    )

    parser.add_argument(
        "--phase",
        choices=["synthetic", "mining", "process", "all"],
        required=True,
        help="執行的管線階段: synthetic (A) | mining (B) | process | all (全流程)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="corpus.txt 的路徑 (預設: data/raw/corpus.txt)",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="最多處理的段落數量 (開發測試用，None = 全部)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hard Negative Mining 使用的 embedding 模型名稱 "
             "(預設: BAAI/bge-small-zh-v1.5，small 速度較快)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用斷點續傳，從頭重新生成 (synthetic 階段適用)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="驗證集佔比 (預設: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日誌等級 (預設從 .env LOG_LEVEL 讀取，或 INFO)",
    )

    return parser.parse_args()


def main() -> None:
    """主函數: 依據 --phase 參數分派到對應的子流程。"""

    args = parse_args()

    # 載入設定
    base_cfg = BaseConfig()
    model_cfg = ModelConfig()
    retrieval_cfg = RetrievalConfig()
    vram_cfg = VRAMConfig()

    # 設定日誌
    log_level = args.log_level or base_cfg.log_level
    setup_logger(log_level)

    logger.info(f"{'='*60}")
    logger.info(f"Domain-Adaptive Retrieval System — 資料管線")
    logger.info(f"  Phase:        {args.phase}")
    logger.info(f"  Project root: {base_cfg.project_root}")
    logger.info(f"  Device:       {base_cfg.device}")
    logger.info(f"  Seed:         {base_cfg.seed}")
    logger.info(f"{'='*60}")

    # corpus 路徑: 優先使用命令列參數，否則使用設定檔的預設路徑
    corpus_path = Path(args.corpus) if args.corpus else (
        base_cfg.data_raw_dir / "corpus.txt"
    )

    # embedding model: 優先使用命令列參數，否則使用 small 模型 (速度較快)
    mining_model = args.model or model_cfg.small.model_name

    # ----------------------------------------------------------------
    # 階段路由
    # ----------------------------------------------------------------

    if args.phase in ("synthetic", "all"):
        _run_phase_synthetic(
            base_cfg=base_cfg,
            corpus_path=corpus_path,
            max_passages=args.max_passages,
            resume=not args.no_resume,
        )

        # all 模式下，階段 A 結束後需要停止 Ollama 才能繼續
        if args.phase == "all":
            logger.warning(
                "\n" + "!"*60 + "\n"
                "[all 模式] 階段 A 完成後，需要手動停止 Ollama 服務！\n"
                "請在 Host 端執行: make stop-ollama\n"
                "然後重新執行: python scripts/run_data_pipeline.py --phase mining\n"
                "!"*60 + "\n"
                "注意: all 模式無法自動停止 Ollama (Ollama 運行在獨立容器中)，\n"
                "      VRAM 切換必須透過 docker compose stop 在容器外部執行。\n"
                "      這是 VRAM 時間分時架構的設計限制，非程式錯誤。"
            )
            # 在 --phase all 模式下，階段 A 後必須人工介入，故此處提前結束
            # 完整自動化應透過 Makefile 的 pipeline target 串接
            logger.info("請完成 Ollama 停止後，重新執行 --phase mining")
            return

    if args.phase in ("mining", "all"):
        _run_phase_mining(
            base_cfg=base_cfg,
            corpus_path=corpus_path,
            retrieval_cfg=retrieval_cfg,
            vram_cfg=vram_cfg,
            mining_model=mining_model,
        )

    if args.phase in ("process", "all"):
        _run_phase_process(
            base_cfg=base_cfg,
            val_ratio=args.val_ratio,
            seed=base_cfg.seed,
        )

    logger.info(f"{'='*60}")
    logger.info(f"✓ 階段 [{args.phase}] 全部完成")
    logger.info(f"{'='*60}")


# ----------------------------------------------------------------
# 子階段函數
# ----------------------------------------------------------------

def _run_phase_synthetic(
    base_cfg: BaseConfig,
    corpus_path: Path,
    max_passages: int,
    resume: bool,
) -> None:
    """
    執行階段 A: LLM 合成資料生成。

    VRAM 備忘: 此函數執行期間 Python 進程 VRAM = 0。
    Ollama 服務自行管理 GPU，佔用約 4-6 GB。
    """
    logger.info("=" * 55)
    logger.info("[階段 A] LLM 合成資料生成")
    logger.info("=" * 55)

    output_path = base_cfg.data_synthetic_dir / "synthetic_pairs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = run_synthetic_generation(
        corpus_path=corpus_path,
        output_path=output_path,
        ollama_host=base_cfg.ollama_host,
        ollama_model=base_cfg.ollama_model,
        max_passages=max_passages,
        resume=resume,
    )

    logger.info(f"[階段 A] 輸出: {output_path} ({count} 筆配對)")
    logger.info(
        "[階段 A] ★ 必要後續步驟: 停止 Ollama 釋放 VRAM\n"
        "          執行: make stop-ollama"
    )


def _run_phase_mining(
    base_cfg: BaseConfig,
    corpus_path: Path,
    retrieval_cfg: RetrievalConfig,
    vram_cfg: VRAMConfig,
    mining_model: str,
) -> None:
    """
    執行階段 B: Hard Negative Mining。

    前置條件: Ollama 服務必須已停止，否則 VRAMGuard 的安全斷言會失敗。
    """
    logger.info("=" * 55)
    logger.info("[階段 B] Hard Negative Mining")
    logger.info("=" * 55)

    synthetic_path = base_cfg.data_synthetic_dir / "synthetic_pairs.jsonl"
    output_path = base_cfg.data_hard_negatives_dir / "triplets.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = run_hard_negative_mining(
        synthetic_data_path=synthetic_path,
        corpus_path=corpus_path,
        output_path=output_path,
        model_name=mining_model,
        device=base_cfg.device,
        cache_dir=str(base_cfg.base_models_dir),
        hard_neg_per_query=retrieval_cfg.hard_neg_per_query,
        score_min=retrieval_cfg.hard_neg_score_min,
        score_max=retrieval_cfg.hard_neg_score_max,
        encode_batch_size=retrieval_cfg.mining_encode_batch_size,
        similarity_chunk_size=retrieval_cfg.similarity_chunk_size,
        stopwords_path=retrieval_cfg.stopwords_path,
        jieba_user_dict_path=retrieval_cfg.jieba_user_dict_path,
        post_release_max_mb=vram_cfg.post_release_max_mb,
    )

    logger.info(f"[階段 B] 輸出: {output_path} ({count} 個 triplets)")


def _run_phase_process(
    base_cfg: BaseConfig,
    val_ratio: float,
    seed: int,
) -> None:
    """執行資料後處理: 清洗、去重、切分 train/val。"""
    logger.info("=" * 55)
    logger.info("[後處理] 資料清洗與切分")
    logger.info("=" * 55)

    input_path = base_cfg.data_hard_negatives_dir / "triplets.jsonl"
    train_path = base_cfg.data_processed_dir / "train.jsonl"
    val_path = base_cfg.data_processed_dir / "val.jsonl"

    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_count, val_count = run_data_processing(
        hard_negatives_path=input_path,
        train_output_path=train_path,
        val_output_path=val_path,
        val_ratio=val_ratio,
        seed=seed,
    )

    logger.info(
        f"[後處理] 完成\n"
        f"  Train: {train_count} 筆 → {train_path}\n"
        f"  Val:   {val_count} 筆 → {val_path}"
    )


if __name__ == "__main__":
    main()
