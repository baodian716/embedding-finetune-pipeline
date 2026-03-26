# evaluation/ — 評估與視覺化模組
# 包含 MRR@10、NDCG@10 計算、長條圖、UMAP 散點圖

from evaluation.metrics import compute_mrr_at_k, compute_ndcg_at_k, evaluate_ranking_results
from evaluation.evaluator import AblationEvaluator, CONDITION_LABELS, CONDITION_ORDER
from evaluation.visualizer import plot_metrics_bar_chart, plot_umap_comparison, generate_all_charts
from evaluation.report_generator import (
    print_results_table,
    generate_markdown_report,
    save_results_json,
)

__all__ = [
    # metrics
    "compute_mrr_at_k",
    "compute_ndcg_at_k",
    "evaluate_ranking_results",
    # evaluator
    "AblationEvaluator",
    "CONDITION_LABELS",
    "CONDITION_ORDER",
    # visualizer
    "plot_metrics_bar_chart",
    "plot_umap_comparison",
    "generate_all_charts",
    # report
    "print_results_table",
    "generate_markdown_report",
    "save_results_json",
]
