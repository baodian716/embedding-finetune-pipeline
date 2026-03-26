# ============================================================
# evaluation/report_generator.py — 評估報告生成器
#
# 產出兩種格式:
#   1. 終端機表格 (pandas DataFrame → Markdown)
#      讓開發者直接在 CLI 檢視結果，不需要打開文件
#
#   2. Markdown 報告 (outputs/reports/evaluation_report.md)
#      包含: 結果表格、圖表嵌入、效能分析、已知限制聲明
#
# ★ 誠實聲明原則:
#   本評估使用的 corpus 僅包含 val.jsonl 中的段落 (positives + hard_negatives)，
#   非完整的 domain corpus。此限制使 MRR/NDCG 數值偏高。
#   報告中必須清楚標注此限制，避免誤導讀者高估絕對效能。
# ============================================================

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas 未安裝，終端機表格輸出將降級為純文字。請執行: pip install pandas")

from evaluation.evaluator import CONDITION_LABELS, CONDITION_ORDER


# ============================================================
# 終端機表格輸出
# ============================================================

def print_results_table(
    results: Dict[str, Dict[str, float]],
    k: int = 10,
) -> None:
    """
    在終端機輸出 Markdown 格式的結果對比表。

    若 pandas 已安裝，使用 DataFrame.to_markdown() 格式化。
    否則退回純文字表格。

    Args:
        results: AblationEvaluator.run_all_conditions() 的輸出
        k:       評估截斷點
    """
    mrr_key  = f"mrr@{k}"
    ndcg_key = f"ndcg@{k}"
    hit_key  = f"hit_rate@{k}"

    conditions = [c for c in CONDITION_ORDER if c in results]

    if PANDAS_AVAILABLE:
        rows = []
        for cond in conditions:
            m = results[cond]
            label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
            rows.append({
                "條件":           label,
                f"MRR@{k}":      f"{m.get(mrr_key,  0.0):.4f}",
                f"NDCG@{k}":     f"{m.get(ndcg_key, 0.0):.4f}",
                f"Hit Rate@{k}": f"{m.get(hit_key,  0.0):.4f}",
                "Query 數":      m.get("num_queries", 0),
            })
        df = pd.DataFrame(rows)

        try:
            table_str = df.to_markdown(index=False)
        except Exception:
            # tabulate 可能未安裝 (to_markdown 的依賴)
            table_str = df.to_string(index=False)

        print(f"\n{'='*70}")
        print("消融實驗結果 (Ablation Study)")
        print(f"{'='*70}")
        print(table_str)
        print(f"{'='*70}\n")
    else:
        # 純文字退路
        print(f"\n{'='*70}")
        print(f"{'條件':<25} | {f'MRR@{k}':>8} | {f'NDCG@{k}':>8} | {f'HitRate@{k}':>10}")
        print(f"{'-'*70}")
        for cond in conditions:
            m = results[cond]
            label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
            print(
                f"{label:<25} | "
                f"{m.get(mrr_key, 0.0):.4f}   | "
                f"{m.get(ndcg_key, 0.0):.4f}   | "
                f"{m.get(hit_key, 0.0):.4f}"
            )
        print(f"{'='*70}\n")


def _compute_improvement(
    results: Dict[str, Dict[str, float]],
    metric_key: str,
) -> Dict[str, float]:
    """計算 LoRA 相對 Baseline 的改善幅度 (百分點與百分比)。"""
    improvements = {}
    pairs = [
        ("small_baseline", "small_lora",  "small"),
        ("base_baseline",  "base_lora",   "base"),
    ]
    for base_cond, lora_cond, prefix in pairs:
        if base_cond in results and lora_cond in results:
            base_val = results[base_cond].get(metric_key, 0.0)
            lora_val = results[lora_cond].get(metric_key, 0.0)
            delta_abs = lora_val - base_val
            delta_pct = (delta_abs / base_val * 100) if base_val > 0 else 0.0
            improvements[prefix] = {
                "baseline":  base_val,
                "lora":      lora_val,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
            }
    return improvements


# ============================================================
# Markdown 報告生成
# ============================================================

def generate_markdown_report(
    results: Dict[str, Dict[str, float]],
    charts_dir: Path,
    output_path: Path,
    k: int = 10,
    eval_note: str = "",
) -> Path:
    """
    生成完整的 Markdown 評估報告，包含:
    - 執行時間與設定摘要
    - 結果表格
    - 效能提升分析
    - 圖表引用 (長條圖 + UMAP)
    - 方法論說明與已知限制聲明

    Args:
        results:     AblationEvaluator.run_all_conditions() 的輸出
        charts_dir:  圖表目錄 (相對路徑將嵌入 Markdown)
        output_path: 報告輸出路徑 (.md)
        k:           評估截斷點
        eval_note:   自定義備注文字 (可選)

    Returns:
        報告的實際儲存路徑
    """
    mrr_key  = f"mrr@{k}"
    ndcg_key = f"ndcg@{k}"
    hit_key  = f"hit_rate@{k}"

    conditions = [c for c in CONDITION_ORDER if c in results]
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ----------------------------------------------------------------
    # 計算改善幅度
    # ----------------------------------------------------------------
    mrr_imp  = _compute_improvement(results, mrr_key)
    ndcg_imp = _compute_improvement(results, ndcg_key)

    # ----------------------------------------------------------------
    # 建構 Markdown 內容
    # ----------------------------------------------------------------
    lines: List[str] = []

    # 標題
    lines += [
        "# Domain-Adaptive Retrieval System — 評估報告",
        "",
        f"**生成時間:** {timestamp}  ",
        f"**評估截斷點:** k={k}  ",
        "",
        "---",
        "",
    ]

    # 摘要
    lines += [
        "## 執行摘要",
        "",
        "本報告呈現四象限消融實驗 (Ablation Study) 的量化評估結果，",
        "比較以下五種檢索條件:",
        "",
        "| 代號 | 條件描述 |",
        "|------|----------|",
        "| A    | `bge-small` Baseline (未微調) |",
        "| B    | `bge-small` + LoRA 微調 |",
        "| C    | `bge-base` Baseline (未微調) |",
        "| D    | `bge-base` + LoRA 微調 |",
        "| E    | 最佳 LoRA 模型 + BM25 Hybrid (RRF 融合) |",
        "",
        "---",
        "",
    ]

    # 結果表格
    lines += [
        "## 量化評估結果",
        "",
        "### 指標說明",
        "",
        "- **MRR@k** (Mean Reciprocal Rank): 衡量「第一個相關文件」的排名品質。",
        "  若第一名命中，MRR=1；前 k 名全未命中，MRR=0。",
        "- **NDCG@k** (Normalized Discounted Cumulative Gain): 對前 k 名所有相關文件計分，",
        "  排名越前分數越高 (log 折扣)。對多個相關文件的 query 比 MRR 更全面。",
        "- **Hit Rate@k**: 前 k 名內至少有一個相關文件的 query 比例。",
        "",
        "### 結果對比",
        "",
    ]

    # 表格 header
    lines.append(
        f"| 條件 | 描述 | MRR@{k} | NDCG@{k} | Hit Rate@{k} | Queries |"
    )
    lines.append("|------|------|---------|----------|-------------|---------|")

    cond_code = {"small_baseline": "A", "small_lora": "B",
                 "base_baseline": "C",  "base_lora": "D", "hybrid": "E"}

    for cond in conditions:
        m     = results[cond]
        code  = cond_code.get(cond, "?")
        label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
        lines.append(
            f"| **{code}** | {label} "
            f"| {m.get(mrr_key, 0):.4f} "
            f"| {m.get(ndcg_key, 0):.4f} "
            f"| {m.get(hit_key, 0):.4f} "
            f"| {m.get('num_queries', 0)} |"
        )

    lines += ["", "---", ""]

    # 改善幅度分析
    lines += [
        "## LoRA 微調效果分析",
        "",
    ]

    for prefix, label in [("small", "bge-small"), ("base", "bge-base")]:
        if prefix not in mrr_imp:
            continue
        mi = mrr_imp[prefix]
        ni = ndcg_imp.get(prefix, {})

        sign_mrr  = "+" if mi["delta_abs"] >= 0 else ""
        sign_ndcg = "+" if ni.get("delta_abs", 0) >= 0 else ""

        lines += [
            f"### {label}",
            "",
            f"| 指標 | Baseline | LoRA | 絕對改善 | 相對改善 |",
            "|------|----------|------|---------|---------|",
            f"| MRR@{k}  | {mi['baseline']:.4f} | {mi['lora']:.4f} "
            f"| {sign_mrr}{mi['delta_abs']:.4f} | {sign_mrr}{mi['delta_pct']:.2f}% |",
            f"| NDCG@{k} | {ni.get('baseline', 0):.4f} | {ni.get('lora', 0):.4f} "
            f"| {sign_ndcg}{ni.get('delta_abs', 0):.4f} "
            f"| {sign_ndcg}{ni.get('delta_pct', 0):.2f}% |",
            "",
        ]

    # Hybrid 效果
    if "hybrid" in results and ("base_lora" in results or "small_lora" in results):
        best_lora_cond  = "base_lora" if "base_lora" in results else "small_lora"
        hybrid_mrr      = results["hybrid"].get(mrr_key, 0)
        best_lora_mrr   = results[best_lora_cond].get(mrr_key, 0)
        hybrid_delta    = hybrid_mrr - best_lora_mrr
        hybrid_delta_pct = (hybrid_delta / best_lora_mrr * 100) if best_lora_mrr > 0 else 0

        sign = "+" if hybrid_delta >= 0 else ""
        lines += [
            "### Hybrid RRF 融合的附加價值",
            "",
            f"相較於最佳 Dense LoRA 單軌模式:",
            "",
            f"| 指標 | 最佳 LoRA (Dense) | Hybrid (Dense+BM25) | 改善 |",
            "|------|------------------|---------------------|------|",
            f"| MRR@{k} | {best_lora_mrr:.4f} | {hybrid_mrr:.4f} "
            f"| {sign}{hybrid_delta:.4f} ({sign}{hybrid_delta_pct:.2f}%) |",
            "",
            "> BM25 的關鍵詞匹配補充了 Dense 模型在「精確詞彙」上的不足，",
            "> RRF 融合通常能帶來 1~5% 的額外提升。",
            "",
            "---",
            "",
        ]

    # 圖表引用
    bar_chart_path  = charts_dir / "metrics_comparison.png"
    umap_chart_path = charts_dir / "umap_comparison_small.png"

    lines += [
        "## 視覺化圖表",
        "",
        "### 指標長條圖",
        "",
    ]

    if bar_chart_path.exists():
        rel_path = bar_chart_path.name
        lines += [
            f"![指標長條圖]({rel_path})",
            "",
            "*圖: 五種條件的 MRR@10 / NDCG@10 對比。虛線表示 Baseline 分數。"
            "長條右上方的百分比為相對 Baseline 的改善幅度。*",
            "",
        ]
    else:
        lines += ["*(圖表檔案尚未生成)*", ""]

    lines += ["### UMAP 高維拓撲視覺化", ""]

    if umap_chart_path.exists():
        rel_path = umap_chart_path.name
        lines += [
            f"![UMAP 對比圖]({rel_path})",
            "",
            "*圖: UMAP 2D 投影對比。左: Baseline，右: LoRA 微調後。",
            "理想情況下，微調後 Query (★) 應更靠近 Positive (●)，",
            "遠離 Hard Negative (✕)，呈現對比學習的物理效果。*",
            "",
        ]
    else:
        lines += ["*(UMAP 圖表尚未生成或 umap-learn 未安裝)*", ""]

    lines += ["---", ""]

    # 方法論說明
    lines += [
        "## 方法論說明",
        "",
        "### 評估語料庫的限制 ⚠️",
        "",
        "> **重要**: 本評估的 `corpus` 僅包含 `val.jsonl` 中的段落",
        "> (positives + hard_negatives)，而非完整的 domain corpus。",
        ">",
        "> **影響**: MRR 和 NDCG 數值偏高，因為每個 query 只需從",
        "> 幾百個候選中找到正確答案，而非從數萬個真實語料中搜尋。",
        ">",
        "> **正確的使用方式**: 這些數值應用於比較各條件的「相對表現」，",
        "> 而非代表真實 IR 任務的「絕對效能」。",
        "",
        "### 硬體限制與妥協",
        "",
        "- **小批次 in-batch negatives**: RTX 4060 8GB 限制 batch_size=4~8，",
        "  每步只有 3~7 個 in-batch negatives，遠少於原始 BGE 訓練的 256+。",
        "  預挖掘的 hard negatives 部分補償了此不足，但無法完全彌補。",
        "- **LoRA r=8 的容量限制**: 參數量極小 (~1% 的 base model)，",
        "  複雜的領域遷移可能需要更大的 r 值 (如 16 或 32)。",
        "- **max_seq_length=256**: 原始 BGE 訓練使用 512，截斷可能遺失尾部語意。",
        "",
        "---",
        "",
    ]

    if eval_note:
        lines += [
            "## 備注",
            "",
            eval_note,
            "",
            "---",
            "",
        ]

    # 原始資料 (JSON)
    lines += [
        "## 原始評估資料",
        "",
        "```json",
        json.dumps(results, indent=2, ensure_ascii=False),
        "```",
        "",
    ]

    # ----------------------------------------------------------------
    # 寫入文件
    # ----------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"評估報告已儲存: {output_path} ({size_kb:.0f} KB)")
    return output_path


def save_results_json(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> Path:
    """
    將評估結果儲存為 JSON，供後續分析或程式化讀取。

    Args:
        results:     評估結果字典
        output_path: JSON 輸出路徑

    Returns:
        實際儲存路徑
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "results":   results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"評估結果 JSON 已儲存: {output_path}")
    return output_path
