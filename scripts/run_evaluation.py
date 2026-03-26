# ============================================================
# scripts/run_evaluation.py — 完整評估入口腳本
#
# 執行四象限消融實驗並輸出報告，流程如下:
#   1. 載入 val.jsonl 建立 ValDataset
#   2. 依序評估五種條件 (A~E)，自動管理 VRAM 載入/釋放
#   3. 終端機輸出 Markdown 結果表格
#   4. 生成量化長條圖 (PNG)
#   5. 提取 embedding → UMAP 降維 → 散點對比圖 (PNG)
#   6. 生成完整 Markdown 評估報告
#   7. 儲存原始結果 JSON
#
# 執行方式:
#   # 完整評估 (所有五種條件 + UMAP 視覺化)
#   python scripts/run_evaluation.py
#
#   # 僅評估 small 模型 (跳過 base，速度較快)
#   python scripts/run_evaluation.py --models small
#
#   # 跳過 UMAP (umap-learn 未安裝時)
#   python scripts/run_evaluation.py --no-umap
#
#   # 使用自定義 val 資料與輸出目錄
#   python scripts/run_evaluation.py --val-data data/processed/val.jsonl \
#                                    --output-dir outputs/eval_20240101
#
# ★ VRAM 需求: ~200~350 MB (推論無需梯度，遠小於訓練)
#   若仍 OOM，請確認 Ollama 服務已停止: make stop-ollama
# ============================================================

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger

from configs.base_config import BaseConfig
from configs.model_config import ModelConfig
from configs.retrieval_config import RetrievalConfig
from configs.vram_config import VRAMConfig

from data_pipeline.vram_guard import VRAMGuard
from training.dataset import ValDataset
from training.memory_utils import log_vram

from evaluation.evaluator import AblationEvaluator
from evaluation.visualizer import generate_all_charts
from evaluation.report_generator import (
    generate_markdown_report,
    print_results_table,
    save_results_json,
)


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
        rotation="50 MB",
        retention=3,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain-Adaptive Retrieval System — 消融實驗評估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
執行範例:
  # 完整評估 (推薦)
  python scripts/run_evaluation.py

  # 只評估 small 模型
  python scripts/run_evaluation.py --models small

  # 跳過 UMAP 視覺化 (速度更快)
  python scripts/run_evaluation.py --no-umap

  # 使用自定義路徑
  python scripts/run_evaluation.py \\
    --val-data data/processed/val.jsonl \\
    --output-dir outputs/my_eval
        """,
    )

    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="val.jsonl 路徑 (預設: data/processed/val.jsonl)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["small", "base"],
        default=["small", "base"],
        help="要評估的模型變體 (預設: both small and base)",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="跳過 UMAP 視覺化 (umap-learn 未安裝時使用)",
    )
    parser.add_argument(
        "--umap-model",
        choices=["small", "base"],
        default="small",
        help="UMAP 視覺化使用哪個模型 (預設: small，速度較快)",
    )
    parser.add_argument(
        "--umap-samples",
        type=int,
        default=300,
        help="UMAP 採樣的三元組數量 (預設 300，過多會使 UMAP 計算變慢)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="評估輸出目錄 (預設: outputs/)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="評估截斷點 (預設: 10 → MRR@10, NDCG@10)",
    )
    parser.add_argument(
        "--skip-vram-check",
        action="store_true",
        help="跳過 VRAM 安全斷言",
    )
    parser.add_argument(
        "--jieba-dict",
        type=str,
        default=None,
        help="jieba 自定義辭典路徑",
    )
    parser.add_argument(
        "--stopwords",
        type=str,
        default=None,
        help="停用詞清單路徑",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--eval-note",
        type=str,
        default="",
        help="附加到報告的自定義備注文字",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ----------------------------------------------------------------
    # 1. 載入設定
    # ----------------------------------------------------------------
    base_cfg      = BaseConfig()
    model_cfg     = ModelConfig()
    retrieval_cfg = RetrievalConfig()
    vram_cfg      = VRAMConfig()

    # 設定輸出目錄
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_cfg.project_root / "outputs"

    charts_dir  = output_dir / "charts"
    reports_dir = output_dir / "reports"

    log_file = output_dir / "evaluation.log"
    setup_logger(args.log_level, log_file)

    logger.info(
        f"\n{'='*60}\n"
        f"Domain-Adaptive Retrieval System — 消融實驗評估\n"
        f"  評估模型: {args.models}\n"
        f"  截斷點:   k={args.k}\n"
        f"  UMAP:     {'停用' if args.no_umap else '啟用'}\n"
        f"  輸出目錄: {output_dir}\n"
        f"{'='*60}"
    )

    # ----------------------------------------------------------------
    # 2. VRAM 安全確認
    # ----------------------------------------------------------------
    if not args.skip_vram_check:
        guard = VRAMGuard(post_release_max_mb=vram_cfg.post_release_max_mb)
        guard.assert_safe_to_proceed("Evaluation")
    else:
        logger.warning("已跳過 VRAM 安全斷言 (--skip-vram-check)")

    log_vram("評估開始前")

    # ----------------------------------------------------------------
    # 3. 載入 val.jsonl
    # ----------------------------------------------------------------
    val_path = Path(args.val_data) if args.val_data else (
        base_cfg.data_processed_dir / "val.jsonl"
    )

    if not val_path.exists():
        logger.error(
            f"val.jsonl 不存在: {val_path}\n"
            f"請先執行資料管線: make data-process"
        )
        sys.exit(1)

    val_dataset = ValDataset(val_path=val_path)

    if len(val_dataset.queries) == 0:
        logger.error("val.jsonl 解析後無有效 query，請確認資料格式正確")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 4. 過濾要評估的模型變體
    # ----------------------------------------------------------------
    # 若用戶只指定 --models small，則跳過 base 相關條件
    # AblationEvaluator 目前設計為評估全部五個條件
    # 這裡透過提前告知 logger 即可，評估器內部仍執行全部條件
    # (部分條件若 adapter 不存在，會自動退回 baseline)
    logger.info(
        f"評估範圍: {args.models}\n"
        "  ★ 若指定了 --models small，base 條件仍會執行，"
        "但 base LoRA adapter 若不存在將退回 baseline"
    )

    # ----------------------------------------------------------------
    # 5. 執行消融實驗評估
    # ----------------------------------------------------------------
    evaluator = AblationEvaluator(
        val_dataset=val_dataset,
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        retrieval_cfg=retrieval_cfg,
        vram_cfg=vram_cfg,
        k=args.k,
        jieba_user_dict_path=args.jieba_dict,
        jieba_stopwords_path=args.stopwords,
    )

    logger.info("開始執行五種條件的評估 (預計需要 5~20 分鐘)...")
    results = evaluator.run_all_conditions()

    # ----------------------------------------------------------------
    # 6. 終端機表格輸出
    # ----------------------------------------------------------------
    print_results_table(results, k=args.k)
    log_vram("所有 Dense 評估完成")

    # ----------------------------------------------------------------
    # 7. UMAP Embedding 提取 (可選)
    # ----------------------------------------------------------------
    embedding_data = {}
    if not args.no_umap:
        logger.info(f"提取 UMAP embedding (模型: {args.umap_model}, 樣本: {args.umap_samples})...")
        try:
            umap_variant = model_cfg.get_variant(args.umap_model)
            embedding_data = evaluator.extract_umap_embeddings(
                model_variant=umap_variant,
                max_samples=args.umap_samples,
            )
        except Exception as e:
            logger.warning(f"UMAP embedding 提取失敗，跳過視覺化: {e}")
            embedding_data = {}

        log_vram("UMAP embedding 提取完成")
    else:
        logger.info("--no-umap: 跳過 UMAP 視覺化")

    # ----------------------------------------------------------------
    # 8. 生成圖表
    # ----------------------------------------------------------------
    logger.info("生成視覺化圖表...")
    chart_paths = generate_all_charts(
        results=results,
        embedding_data=embedding_data,
        charts_dir=charts_dir,
        model_short_name=args.umap_model,
        k=args.k,
    )

    # ----------------------------------------------------------------
    # 9. 生成 Markdown 報告
    # ----------------------------------------------------------------
    report_path = reports_dir / "evaluation_report.md"
    logger.info("生成 Markdown 評估報告...")
    generate_markdown_report(
        results=results,
        charts_dir=charts_dir,
        output_path=report_path,
        k=args.k,
        eval_note=args.eval_note,
    )

    # ----------------------------------------------------------------
    # 10. 儲存原始 JSON
    # ----------------------------------------------------------------
    json_path = reports_dir / "evaluation_results.json"
    save_results_json(results, json_path)

    # ----------------------------------------------------------------
    # 11. 最終摘要
    # ----------------------------------------------------------------
    mrr_key  = f"mrr@{args.k}"
    ndcg_key = f"ndcg@{args.k}"

    best_cond = max(
        (c for c in results if c != "hybrid"),
        key=lambda c: results[c].get(mrr_key, 0.0),
        default=None,
    )

    logger.info(
        f"\n{'='*60}\n"
        f"評估完成 ✓\n"
        f"\n  最佳 Dense 條件: {best_cond}\n"
        f"    MRR@{args.k}:  {results.get(best_cond, {}).get(mrr_key,  0):.4f}\n"
        f"    NDCG@{args.k}: {results.get(best_cond, {}).get(ndcg_key, 0):.4f}\n"
        f"\n  Hybrid 結果:\n"
        f"    MRR@{args.k}:  {results.get('hybrid', {}).get(mrr_key,  0):.4f}\n"
        f"    NDCG@{args.k}: {results.get('hybrid', {}).get(ndcg_key, 0):.4f}\n"
        f"\n  輸出檔案:\n"
        f"    長條圖: {chart_paths.get('bar_chart', '未生成')}\n"
        f"    UMAP:   {chart_paths.get('umap', '未生成')}\n"
        f"    報告:   {report_path}\n"
        f"    JSON:   {json_path}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
