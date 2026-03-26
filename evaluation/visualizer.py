# ============================================================
# evaluation/visualizer.py — 量化長條圖 + UMAP 高維拓撲視覺化
#
# 輸出兩種圖表:
#
# 1. 量化指標長條圖 (Metrics Bar Chart)
#    - 五種條件的 MRR@10 / NDCG@10 並列比較
#    - Baseline (A/C) 以水平虛線標示，讓微調進步幅度一目了然
#    - 長條上方標示數值 (annotate)
#
# 2. UMAP 對比散點圖 (UMAP Scatter Plot)
#    - 左側: Baseline 模型的 embedding 分布
#    - 右側: LoRA 微調後的 embedding 分布
#    - 標記:
#        Query:          ★ 星形 (藍色)
#        Positive:       ● 圓形 (綠色)
#        Hard Negative:  ✕ 叉形 (紅色)
#    - 對比學習的物理效果:
#        理想情況下，LoRA 訓練後應可見:
#        → Query 與對應 Positive 的距離縮短
#        → Query 與 Hard Negative 的距離增大
#        → 同類型的點聚類更緊密
#
# ★ OOM 防護:
#    UMAP 的 fit_transform() 在 CPU numpy 上執行，
#    不使用 GPU，不佔用 VRAM。
#    若語料超過 5000 點，UMAP 本身的 CPU RAM 需求也較大，
#    建議透過 max_samples 限制採樣數 (預設 300 三元組 = 900 點)。
# ============================================================

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")  # 非互動後端，確保 headless server 也能生成圖片
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib 未安裝，無法生成圖表。請執行: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("umap-learn 未安裝，UMAP 視覺化將被跳過。請執行: pip install umap-learn")

from evaluation.evaluator import CONDITION_LABELS, CONDITION_ORDER


# ============================================================
# 圖表樣式設定
# ============================================================

# 顏色方案: 使用色盲友善配色
PALETTE = {
    "small_baseline": "#AECBFA",   # 淺藍 (baseline)
    "small_lora":     "#1A73E8",   # 深藍 (LoRA)
    "base_baseline":  "#FDD99B",   # 淺橙 (baseline)
    "base_lora":      "#F9A825",   # 深橙 (LoRA)
    "hybrid":         "#34A853",   # 綠色 (Hybrid, 最佳)
}

HATCH_BASELINE = "///"  # Baseline 長條加斜線紋理，進一步區分

# UMAP 點的樣式設定
UMAP_STYLES = {
    "query":    {"marker": "*", "color": "#1A73E8", "s": 120, "label": "Query",         "zorder": 5},
    "positive": {"marker": "o", "color": "#34A853", "s": 50,  "label": "Positive",      "zorder": 4},
    "negative": {"marker": "X", "color": "#EA4335", "s": 50,  "label": "Hard Negative", "zorder": 4},
}


def _check_matplotlib() -> bool:
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib 未安裝，無法生成圖表")
        return False
    return True


# ============================================================
# 圖表 1: 量化指標長條圖
# ============================================================

def plot_metrics_bar_chart(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    k: int = 10,
    figsize: Tuple[int, int] = (14, 7),
    dpi: int = 150,
) -> Optional[Path]:
    """
    繪製五種條件的 MRR@k / NDCG@k 並列長條圖，並儲存為 PNG。

    設計說明:
    - 左子圖: MRR@k，右子圖: NDCG@k
    - Baseline 條 (A, C) 加斜線紋理
    - Baseline 分數以水平虛線在整張圖中標示，使改善幅度一目了然
    - 長條上方標示精確數值 (4 位小數)

    Args:
        results:     AblationEvaluator.run_all_conditions() 的輸出
        output_path: PNG 儲存路徑
        k:           評估截斷點
        figsize:     圖表尺寸
        dpi:         解析度 (150 dpi 足夠報告使用，300 dpi 用於印刷)

    Returns:
        實際儲存的路徑，若生成失敗則回傳 None
    """
    if not _check_matplotlib():
        return None

    if SEABORN_AVAILABLE:
        sns.set_theme(style="whitegrid", font_scale=1.1)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    mrr_key  = f"mrr@{k}"
    ndcg_key = f"ndcg@{k}"

    # 準備資料 (依 CONDITION_ORDER 排列)
    conditions = [c for c in CONDITION_ORDER if c in results]
    labels     = [CONDITION_LABELS.get(c, c) for c in conditions]
    mrr_vals   = [results[c].get(mrr_key, 0.0)  for c in conditions]
    ndcg_vals  = [results[c].get(ndcg_key, 0.0) for c in conditions]
    colors     = [PALETTE.get(c, "#9E9E9E")      for c in conditions]
    hatches    = [HATCH_BASELINE if "baseline" in c else "" for c in conditions]

    x = np.arange(len(conditions))
    bar_width = 0.55

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.suptitle(
        "Ablation Study: Four-Quadrant Model Comparison (Dense + Hybrid)\n"
        "Domain-Adaptive Retrieval System",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, vals, metric_name in [
        (axes[0], mrr_vals,  f"MRR@{k}"),
        (axes[1], ndcg_vals, f"NDCG@{k}"),
    ]:
        bars = ax.bar(
            x, vals,
            width=bar_width,
            color=colors,
            hatch=hatches,
            edgecolor="white",
            linewidth=0.8,
        )

        # ★ Baseline 水平參考虛線
        # small baseline (條件 A) 對應 "small_baseline"
        # base baseline  (條件 C) 對應 "base_baseline"
        vals_map = dict(zip(conditions, vals))
        for ref_cond, ref_label, ref_color in [
            ("small_baseline", "Small Baseline", PALETTE["small_baseline"]),
            ("base_baseline",  "Base Baseline",  PALETTE["base_baseline"]),
        ]:
            if ref_cond in vals_map:
                ref_val = vals_map[ref_cond]
                ax.axhline(
                    y=ref_val,
                    color=ref_color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"{ref_label}: {ref_val:.4f}",
                )

        # 長條頂端數值標籤
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color="#333333",
            )

        ax.set_title(metric_name, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, min(1.0, max(vals) * 1.25))
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")

        # 微調 LoRA 改善幅度標注
        _annotate_improvement(ax, conditions, vals_map, x, bars, k)

    # 手動圖例: 顏色說明
    legend_patches = [
        mpatches.Patch(facecolor=PALETTE["small_baseline"], hatch=HATCH_BASELINE,
                       edgecolor="gray", label="Small Baseline"),
        mpatches.Patch(facecolor=PALETTE["small_lora"],
                       edgecolor="gray", label="Small +LoRA"),
        mpatches.Patch(facecolor=PALETTE["base_baseline"], hatch=HATCH_BASELINE,
                       edgecolor="gray", label="Base Baseline"),
        mpatches.Patch(facecolor=PALETTE["base_lora"],
                       edgecolor="gray", label="Base +LoRA"),
        mpatches.Patch(facecolor=PALETTE["hybrid"],
                       edgecolor="gray", label="Best LoRA +Hybrid"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"指標長條圖已儲存: {output_path} ({size_kb:.0f} KB)")
    return output_path


def _annotate_improvement(ax, conditions, vals_map, x, bars, k):
    """
    在 LoRA 的長條上方標注相對於對應 Baseline 的改善幅度 (Δ%)。
    """
    improvements = {
        "small_lora": ("small_baseline", PALETTE["small_lora"]),
        "base_lora":  ("base_baseline",  PALETTE["base_lora"]),
    }
    cond_to_x = {c: i for i, c in enumerate(conditions)}

    for lora_cond, (base_cond, color) in improvements.items():
        if lora_cond not in vals_map or base_cond not in vals_map:
            continue
        lora_val = vals_map[lora_cond]
        base_val = vals_map[base_cond]
        if base_val == 0:
            continue
        delta_pct = (lora_val - base_val) / base_val * 100

        xi = cond_to_x.get(lora_cond)
        if xi is None:
            continue

        bar = bars[xi]
        sign = "+" if delta_pct >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.025,
            f"({sign}{delta_pct:.1f}%)",
            ha="center", va="bottom",
            fontsize=8, color="#555555", style="italic",
        )


# ============================================================
# 圖表 2: UMAP 對比散點圖
# ============================================================

def plot_umap_comparison(
    embedding_data: Dict[str, np.ndarray],
    output_path: Path,
    model_short_name: str = "model",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    figsize: Tuple[int, int] = (16, 7),
    dpi: int = 150,
    max_points_per_group: int = 300,
) -> Optional[Path]:
    """
    使用 UMAP 將高維 embedding 投影至 2D，繪製 Baseline vs LoRA 對比圖。

    ★ OOM 防護說明:
      embedding_data 中的 numpy array 已在 CPU 上 (DenseRetriever.encode()
      回傳 CPU numpy)，UMAP 的 fit_transform() 也在 CPU 執行。
      整個函數不需要 GPU，不佔用 VRAM。

    ★ 超參數說明:
      n_neighbors=15: 控制局部 vs 全局結構的平衡。15 偏局部結構。
      min_dist=0.1:   控制點的緊湊程度。越小越緊湊。
      metric="cosine": 使用餘弦距離 (與訓練/推論的相似度計算一致)。

    Args:
        embedding_data:       extract_umap_embeddings() 的輸出
        output_path:          PNG 儲存路徑
        model_short_name:     模型名稱 (圖標題用)
        umap_n_neighbors:     UMAP 鄰居數
        umap_min_dist:        UMAP 最小距離
        umap_metric:          UMAP 距離計算方式
        figsize:              圖表尺寸
        dpi:                  解析度
        max_points_per_group: 每類型最多繪製幾個點 (避免圖片過密)

    Returns:
        實際儲存的路徑，若生成失敗則回傳 None
    """
    if not _check_matplotlib():
        return None

    if not UMAP_AVAILABLE:
        logger.warning("umap-learn 未安裝，跳過 UMAP 視覺化")
        return None

    required_keys = [
        "baseline_queries", "baseline_positives", "baseline_negatives",
        "lora_queries",     "lora_positives",     "lora_negatives",
    ]
    if not all(k in embedding_data for k in required_keys):
        logger.warning("embedding_data 欄位不完整，跳過 UMAP 視覺化")
        return None

    # ----------------------------------------------------------------
    # 資料準備: 截取前 max_points_per_group 個點
    # ----------------------------------------------------------------
    def _trim(arr: np.ndarray, n: int) -> np.ndarray:
        return arr[:n] if len(arr) > n else arr

    n = max_points_per_group

    b_q = _trim(embedding_data["baseline_queries"],   n)
    b_p = _trim(embedding_data["baseline_positives"], n)
    b_n = _trim(embedding_data["baseline_negatives"], n)

    l_q = _trim(embedding_data["lora_queries"],       n)
    l_p = _trim(embedding_data["lora_positives"],     n)
    l_n = _trim(embedding_data["lora_negatives"],     n)

    # ----------------------------------------------------------------
    # UMAP 降維
    # ★ 對 Baseline 和 LoRA 分別 fit_transform，使得各自的內部距離可比
    # 注意: 兩張圖的座標系不同，不能直接比較絕對位置，
    #        但同一張圖內的點間距離是有意義的
    # ----------------------------------------------------------------
    logger.info("UMAP 降維計算中 (CPU)...")

    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        n_components=2,
        random_state=42,
        verbose=False,
    )

    # Baseline: 合併三組後一起 fit，確保三組在同一 2D 空間
    baseline_all = np.vstack([b_q, b_p, b_n])  # (3N, D)
    baseline_2d  = reducer.fit_transform(baseline_all)

    n_b_q = len(b_q)
    n_b_p = len(b_p)
    baseline_q_2d = baseline_2d[:n_b_q]
    baseline_p_2d = baseline_2d[n_b_q : n_b_q + n_b_p]
    baseline_n_2d = baseline_2d[n_b_q + n_b_p:]

    # LoRA: 重新 fit (不共用 reducer，保持各自內部結構清晰)
    reducer_lora = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        n_components=2,
        random_state=42,
        verbose=False,
    )
    lora_all = np.vstack([l_q, l_p, l_n])  # (3N, D)
    lora_2d  = reducer_lora.fit_transform(lora_all)

    n_l_q = len(l_q)
    n_l_p = len(l_p)
    lora_q_2d = lora_2d[:n_l_q]
    lora_p_2d = lora_2d[n_l_q : n_l_q + n_l_p]
    lora_n_2d = lora_2d[n_l_q + n_l_p:]

    logger.info("UMAP 降維完成")

    # ----------------------------------------------------------------
    # 繪圖
    # ----------------------------------------------------------------
    if SEABORN_AVAILABLE:
        sns.set_theme(style="dark", font_scale=1.0)
    else:
        plt.style.use("dark_background")

    fig, (ax_base, ax_lora) = plt.subplots(1, 2, figsize=figsize)

    fig.suptitle(
        f"UMAP Embedding Topology: {model_short_name.upper()}\n"
        "Contrastive Learning Effect: Post-LoRA Queries should cluster closer to Positives, farther from Hard Negatives",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, (q_2d, p_2d, n_2d), title in [
        (ax_base, (baseline_q_2d, baseline_p_2d, baseline_n_2d), "Baseline (LoRA Off)"),
        (ax_lora, (lora_q_2d,     lora_p_2d,     lora_n_2d),     "After Fine-Tuning (LoRA On)"),
    ]:
        # 先畫 Positive 和 Negative (在底層)，再畫 Query (在頂層)
        ax.scatter(
            p_2d[:, 0], p_2d[:, 1],
            **{k: v for k, v in UMAP_STYLES["positive"].items() if k != "label"},
            alpha=0.65,
            label=UMAP_STYLES["positive"]["label"],
        )
        ax.scatter(
            n_2d[:, 0], n_2d[:, 1],
            **{k: v for k, v in UMAP_STYLES["negative"].items() if k != "label"},
            alpha=0.65,
            label=UMAP_STYLES["negative"]["label"],
        )
        ax.scatter(
            q_2d[:, 0], q_2d[:, 1],
            **{k: v for k, v in UMAP_STYLES["query"].items() if k != "label"},
            alpha=0.85,
            label=UMAP_STYLES["query"]["label"],
        )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("UMAP Dimension 1", fontsize=10)
        ax.set_ylabel("UMAP Dimension 2", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")

        # 移除刻度 (UMAP 座標無物理意義)
        ax.set_xticks([])
        ax.set_yticks([])

    # 共用說明文字
    fig.text(
        0.5, -0.04,
        f"每組各 {n} 個樣本 | UMAP: n_neighbors={umap_n_neighbors}, "
        f"min_dist={umap_min_dist}, metric={umap_metric}\n"
        "★ 注意: 兩圖座標系獨立，絕對位置不可直接比較，請比較同圖內各點的相對距離",
        ha="center", va="top",
        fontsize=8, style="italic", color="#888888"
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"UMAP 散點圖已儲存: {output_path} ({size_kb:.0f} KB)")
    return output_path


# ============================================================
# 輔助: 合併輸出所有圖表
# ============================================================

def generate_all_charts(
    results: Dict[str, Dict[str, float]],
    embedding_data: Dict[str, np.ndarray],
    charts_dir: Path,
    model_short_name: str = "model",
    k: int = 10,
) -> Dict[str, Optional[Path]]:
    """
    一次生成所有圖表，回傳各圖表的儲存路徑。

    Args:
        results:          AblationEvaluator.run_all_conditions() 的輸出
        embedding_data:   AblationEvaluator.extract_umap_embeddings() 的輸出
        charts_dir:       圖表輸出目錄
        model_short_name: 模型名稱 (UMAP 標題用)
        k:                評估截斷點

    Returns:
        {"bar_chart": Path, "umap": Path}，失敗的圖表對應 None
    """
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # 長條圖
    bar_path = charts_dir / "metrics_comparison.png"
    saved["bar_chart"] = plot_metrics_bar_chart(results, bar_path, k=k)

    # UMAP 散點圖
    if embedding_data:
        umap_path = charts_dir / f"umap_comparison_{model_short_name}.png"
        saved["umap"] = plot_umap_comparison(
            embedding_data,
            umap_path,
            model_short_name=model_short_name,
        )
    else:
        saved["umap"] = None
        logger.info("embedding_data 為空，跳過 UMAP 圖表")

    return saved
