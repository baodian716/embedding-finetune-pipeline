# ============================================================
# scripts/run_retrieval.py — 混合檢索示範入口腳本
#
# 功能:
#   1. 從 data/raw/corpus.txt 或 data/processed/ 載入語料庫
#   2. 建立 Dense (BGE + LoRA) + BM25 (jieba) 雙軌索引
#   3. 對使用者指定的 query 執行三種模式的對比查詢:
#      - Dense Only (語意向量)
#      - BM25 Only  (關鍵詞稀疏)
#      - Hybrid RRF (融合結果)
#   4. 輸出排名比較表，直觀顯示三種模式的差異
#
# 執行方式:
#   # 基本示範 (使用 bge-small + LoRA)
#   python scripts/run_retrieval.py --model small
#
#   # 使用 bge-base + LoRA
#   python scripts/run_retrieval.py --model base
#
#   # 停用 LoRA (純 Baseline 評估)
#   python scripts/run_retrieval.py --model small --no-lora
#
#   # 指定自定義 query
#   python scripts/run_retrieval.py --model small --query "您的查詢文字"
#
#   # 互動模式 (重複輸入 query)
#   python scripts/run_retrieval.py --model small --interactive
#
# ★ 此腳本的主要用途是「快速驗證」，非正式評估。
#   完整的 Baseline vs LoRA 對比評估請使用 scripts/run_evaluation.py。
# ============================================================

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger

from configs.base_config import BaseConfig
from configs.model_config import ModelConfig
from configs.retrieval_config import RetrievalConfig
from configs.vram_config import VRAMConfig

from data_pipeline.vram_guard import VRAMGuard
from retrieval.jieba_tokenizer import JiebaTokenizer
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from training.memory_utils import log_vram


# ============================================================
# 工具函數
# ============================================================

def setup_logger(log_level: str) -> None:
    """設定 loguru 日誌。"""
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


def load_corpus_from_file(corpus_path: Path) -> Dict[str, str]:
    """
    從 corpus.txt 載入語料庫。

    格式: 每行一段文字，行號作為 doc_id。
    空行與 # 開頭的行視為註解並跳過。

    Args:
        corpus_path: corpus.txt 的路徑

    Returns:
        {doc_id: doc_text} 字典
    """
    corpus: Dict[str, str] = {}

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"語料庫不存在: {corpus_path}\n"
            f"請將語料庫文字放入此路徑，每行一段文字。"
        )

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            doc_id = f"doc_{line_num:06d}"
            corpus[doc_id] = line

    logger.info(f"語料庫載入完成: {len(corpus)} 份文件 ← {corpus_path}")
    return corpus


def load_corpus_from_processed(processed_dir: Path) -> Dict[str, str]:
    """
    從 data/processed/val.jsonl 或 train.jsonl 提取語料庫段落。

    ★ 僅用於示範與快速測試。
      正式評估時，應使用完整的 domain corpus，而非只有訓練資料中的段落。
    """
    import json

    corpus: Dict[str, str] = {}
    seen_texts = set()  # 去重

    for filename in ["val.jsonl", "train.jsonl"]:
        jsonl_path = processed_dir / filename
        if not jsonl_path.exists():
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                texts = record.get("texts", [])
                # positive 段落加入語料庫
                if len(texts) >= 2 and texts[1] not in seen_texts:
                    doc_id = f"p_{record.get('id', len(corpus)):06s}" if isinstance(record.get('id'), str) else f"p_{len(corpus):06d}"
                    corpus[doc_id] = texts[1]
                    seen_texts.add(texts[1])
                # hard_negative 也加入語料庫 (增加檢索難度)
                if len(texts) >= 3 and texts[2] not in seen_texts:
                    neg_id = f"n_{len(corpus):06d}"
                    corpus[neg_id] = texts[2]
                    seen_texts.add(texts[2])

    logger.info(f"從 processed/ 提取語料庫: {len(corpus)} 份文件")
    return corpus


def print_results_table(
    query: str,
    results_by_mode: dict,
    corpus: Dict[str, str],
    top_k: int = 5,
) -> None:
    """
    以表格形式輸出三種模式的檢索結果對比。

    Args:
        query: 查詢文字
        results_by_mode: {"Dense": [...], "BM25": [...], "Hybrid": [...]}
        corpus: {doc_id: doc_text} 用於顯示文件內容
        top_k: 顯示前幾筆
    """
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")

    for mode_name, results in results_by_mode.items():
        print(f"\n[{mode_name}] Top-{min(top_k, len(results))} 結果:")
        print(f"{'-'*60}")

        if not results:
            print("  (無結果)")
            continue

        for rank, (doc_id, score) in enumerate(results[:top_k], start=1):
            doc_text = corpus.get(doc_id, "(文件不存在)")
            # 截斷過長的文件文字以利閱讀
            display_text = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
            print(f"  [{rank}] {doc_id} | score={score:.4f}")
            print(f"       {display_text}")

    print(f"\n{'='*70}\n")


# ============================================================
# 主程序
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain-Adaptive Retrieval System — 混合檢索示範",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
執行範例:
  # 使用 bge-small + LoRA 進行混合檢索
  python scripts/run_retrieval.py --model small

  # 使用 bge-base，停用 LoRA (Baseline 對比)
  python scripts/run_retrieval.py --model base --no-lora

  # 指定 query
  python scripts/run_retrieval.py --model small --query "機器學習的應用場景有哪些？"

  # 互動模式
  python scripts/run_retrieval.py --model small --interactive
        """,
    )

    parser.add_argument(
        "--model",
        choices=["small", "base"],
        required=True,
        help="使用的模型變體: small=bge-small-zh-v1.5 | base=bge-base-zh-v1.5",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="停用 LoRA Adapter (Baseline 模式)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA Adapter 路徑 (預設: models/lora_adapters/<variant>/best/)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="語料庫文字路徑 (預設: data/raw/corpus.txt；不存在則退回 processed/)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="查詢文字。若未指定，使用內建的示範 query",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="互動模式: 持續輸入 query 直到 q 退出",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="顯示前 K 筆結果 (預設: 5)",
    )
    parser.add_argument(
        "--skip-vram-check",
        action="store_true",
        help="跳過 VRAM 安全斷言",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--jieba-dict",
        type=str,
        default=None,
        help="jieba 自定義辭典路徑 (補充繁體中文專業術語)",
    )
    parser.add_argument(
        "--stopwords",
        type=str,
        default=None,
        help="停用詞清單路徑",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    # ----------------------------------------------------------------
    # 1. 載入設定
    # ----------------------------------------------------------------
    base_cfg      = BaseConfig()
    model_cfg     = ModelConfig()
    retrieval_cfg = RetrievalConfig()
    vram_cfg      = VRAMConfig()

    model_variant = model_cfg.get_variant(args.model)
    use_lora = not args.no_lora

    logger.info(
        f"\n{'='*60}\n"
        f"Domain-Adaptive Retrieval System — 混合檢索\n"
        f"  模型:   {model_variant.short_name} ({model_variant.model_name})\n"
        f"  LoRA:   {'啟用' if use_lora else '停用 (Baseline)'}\n"
        f"  模式:   Dense + BM25 + RRF (k={retrieval_cfg.rrf_k})\n"
        f"{'='*60}"
    )

    # ----------------------------------------------------------------
    # 2. VRAM 安全確認
    # ----------------------------------------------------------------
    if not args.skip_vram_check:
        guard = VRAMGuard(post_release_max_mb=vram_cfg.post_release_max_mb)
        guard.assert_safe_to_proceed(f"Retrieval ({args.model})")
    else:
        logger.warning("已跳過 VRAM 安全斷言")

    log_vram("檢索開始前")

    # ----------------------------------------------------------------
    # 3. 載入語料庫
    # ----------------------------------------------------------------
    if args.corpus:
        corpus_path = Path(args.corpus)
        corpus = load_corpus_from_file(corpus_path)
    else:
        # 優先使用 data/raw/corpus.txt，否則退回 processed/
        default_corpus = base_cfg.data_raw_dir / "corpus.txt"
        if default_corpus.exists():
            corpus = load_corpus_from_file(default_corpus)
        else:
            logger.warning(
                f"找不到 {default_corpus}，改從 data/processed/ 提取段落作為語料庫。\n"
                f"注意: 此語料庫僅包含訓練資料中的段落，並非完整的 domain corpus。"
            )
            corpus = load_corpus_from_processed(base_cfg.data_processed_dir)

    if not corpus:
        logger.error("語料庫為空，無法建立索引。請確認語料庫路徑正確。")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 4. 初始化各元件
    # ----------------------------------------------------------------

    # 4a. jieba 斷詞器 (BM25 使用)
    tokenizer = JiebaTokenizer(
        user_dict_path=args.jieba_dict,
        stopwords_path=args.stopwords,
    )

    # 4b. BM25 Retriever (純 CPU，不佔 VRAM)
    bm25_retriever = BM25Retriever(
        tokenizer=tokenizer,
        k1=retrieval_cfg.bm25_k1,
        b=retrieval_cfg.bm25_b,
    )

    # 4c. Dense Retriever (GPU)
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    dense_retriever = DenseRetriever(
        model_variant=model_variant,
        base_cfg=base_cfg,
        retrieval_cfg=retrieval_cfg,
        use_lora=use_lora,
    )
    dense_retriever.load_model(adapter_path=adapter_path)
    log_vram("模型載入後")

    # 4d. Hybrid Retriever (RRF 融合)
    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        retrieval_cfg=retrieval_cfg,
    )

    # ----------------------------------------------------------------
    # 5. 建立索引
    # ----------------------------------------------------------------
    hybrid.build_index(corpus)
    log_vram("索引建立後")

    # ----------------------------------------------------------------
    # 6. 準備示範 query
    # ----------------------------------------------------------------
    if args.query:
        demo_queries = [args.query]
    else:
        # 內建示範 query (適合一般領域語料庫，可依實際語料庫調整)
        demo_queries = [
            "機器學習模型如何進行訓練？",
            "自然語言處理的應用領域",
            "深度學習與傳統機器學習的差異",
        ]
        logger.info("未指定 --query，使用內建示範 query")

    # ----------------------------------------------------------------
    # 7. 執行查詢 (三種模式對比)
    # ----------------------------------------------------------------
    for query in demo_queries:
        results_by_mode = {
            "Dense Only":  hybrid.search(query, top_k=args.top_k, use_dense=True,  use_sparse=False),
            "BM25 Only":   hybrid.search(query, top_k=args.top_k, use_dense=False, use_sparse=True),
            "Hybrid (RRF)": hybrid.search(query, top_k=args.top_k, use_dense=True,  use_sparse=True),
        }
        print_results_table(query, results_by_mode, corpus, top_k=args.top_k)

    # ----------------------------------------------------------------
    # 8. 互動模式
    # ----------------------------------------------------------------
    if args.interactive:
        print("\n[互動模式] 請輸入查詢 (輸入 'q' 退出):")
        while True:
            try:
                query = input("\nQuery > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n退出")
                break

            if query.lower() in ("q", "quit", "exit", ""):
                print("退出互動模式")
                break

            results_by_mode = {
                "Dense Only":   hybrid.search(query, top_k=args.top_k, use_dense=True,  use_sparse=False),
                "BM25 Only":    hybrid.search(query, top_k=args.top_k, use_dense=False, use_sparse=True),
                "Hybrid (RRF)": hybrid.search(query, top_k=args.top_k, use_dense=True,  use_sparse=True),
            }
            print_results_table(query, results_by_mode, corpus, top_k=args.top_k)

    # ----------------------------------------------------------------
    # 9. 資源釋放
    # ----------------------------------------------------------------
    dense_retriever.unload_model()
    log_vram("資源釋放後")

    logger.info(
        f"\n{'='*60}\n"
        f"✓ 混合檢索示範完成\n"
        f"  ★ 下一步: 執行正式評估 → make eval\n"
        f"           或繼續調整 RRF 參數: configs/retrieval_config.py\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
