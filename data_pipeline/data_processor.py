# ============================================================
# data_pipeline/data_processor.py — 資料後處理與格式轉換
#
# 功能: 讀取 Hard Negative Mining 的輸出 (triplets JSONL)，
#       執行清洗、去重、格式轉換，最終產出 train.jsonl 與 val.jsonl，
#       供 Step 3 (training/) 模組直接使用。
#
# 處理流程:
#   1. 載入 triplets JSONL (階段 B 輸出)
#   2. 清洗: 過濾長度異常、移除重複 query
#   3. 格式轉換: 轉換為 sentence-transformers 訓練格式
#   4. 切分: 按 val_ratio 切分 train/val 集合 (以 query 為單位分層切分)
#   5. 統計: 輸出資料分布摘要供人工審查
#
# 輸出格式 (sentence-transformers InputExample 對應的 JSONL):
#   {"texts": ["query", "positive", "hard_negative"]}
#   每行一個訓練樣本，可直接用 json.loads() 讀取後建立 InputExample
# ============================================================

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ============================================================
# 資料清洗規則
# ============================================================

def is_valid_text(text: str, min_len: int = 5, max_len: int = 2000) -> bool:
    """
    檢查文字是否符合最低品質要求。

    Args:
        text: 待驗證的文字
        min_len: 最短字元數 (過短 = 幾乎沒有語意)
        max_len: 最長字元數 (超長 = 超過模型 max_seq_length 會被大量截斷)

    Returns:
        True = 通過檢查，False = 應過濾
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text:
        return False
    # 長度檢查: 過短或過長
    char_len = len(text)
    if char_len < min_len or char_len > max_len:
        return False
    # 空白比例檢查: 若超過 50% 都是空白，可能是格式錯誤的資料
    non_space_ratio = len(text.replace(" ", "").replace("\n", "")) / char_len
    if non_space_ratio < 0.5:
        return False
    return True


def deduplicate_by_query(
    triplets: List[Dict],
    max_per_query: int = 10,
) -> List[Dict]:
    """
    以 query 為鍵去重，並限制每個 query 最多保留 max_per_query 個 triplets。

    目的:
    - 避免同一個 query 在訓練集中出現過多次，導致模型對特定 query 過度擬合
    - 確保訓練集的 query 多樣性

    Args:
        triplets: 原始 triplet 列表
        max_per_query: 每個 query 最多保留幾個 triplets

    Returns:
        去重後的 triplet 列表
    """
    query_to_triplets: Dict[str, List[Dict]] = defaultdict(list)
    for t in triplets:
        query = t.get("query", "")
        if len(query_to_triplets[query]) < max_per_query:
            query_to_triplets[query].append(t)

    result = []
    for group in query_to_triplets.values():
        result.extend(group)

    logger.info(
        f"去重結果: {len(triplets)} → {len(result)} 個 triplets "
        f"(唯一 query 數: {len(query_to_triplets)})"
    )
    return result


# ============================================================
# 資料切分
# ============================================================

def split_train_val(
    triplets: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
    min_val_size: int = 50,
) -> Tuple[List[Dict], List[Dict]]:
    """
    以 query 為單位分層切分 train/val 集合。

    設計原則:
    「以 query 為單位」切分，確保同一個 query 的所有 triplets 要麼全在 train，
    要麼全在 val，避免 data leakage。
    若按 triplet 隨機切分，可能導致同一個 query 的部分樣本在 train、部分在 val，
    讓模型在 val 上有「見過此 query」的不公平優勢。

    Args:
        triplets: 所有 triplets
        val_ratio: val 集合佔比 (以 query 數量計算，非 triplet 數量)
        seed: 隨機種子 (確保可重現)
        min_val_size: val 集合的最小 triplet 數量 (語料太少時保護用)

    Returns:
        (train_triplets, val_triplets) 元組
    """
    # 按 query 分組
    query_to_triplets: Dict[str, List[Dict]] = defaultdict(list)
    for t in triplets:
        query_to_triplets[t["query"]].append(t)

    unique_queries = list(query_to_triplets.keys())
    random.seed(seed)
    random.shuffle(unique_queries)

    # 計算 val 要取幾個 query
    num_val_queries = max(
        int(len(unique_queries) * val_ratio),
        1  # 至少 1 個 query 放進 val
    )
    val_queries = set(unique_queries[:num_val_queries])
    train_queries = set(unique_queries[num_val_queries:])

    train_triplets = []
    val_triplets = []
    for query, group in query_to_triplets.items():
        if query in val_queries:
            val_triplets.extend(group)
        else:
            train_triplets.extend(group)

    # 若 val 太小，給出警告 (不強制修改，保留人工決策空間)
    if len(val_triplets) < min_val_size:
        logger.warning(
            f"val 集合只有 {len(val_triplets)} 個 triplets，"
            f"建議至少 {min_val_size} 個以獲得可靠的評估指標。\n"
            f"可透過增加 corpus 語料或降低 val_ratio 調整。"
        )

    logger.info(
        f"資料切分完成:\n"
        f"  Train: {len(train_triplets)} triplets ({len(train_queries)} unique queries)\n"
        f"  Val:   {len(val_triplets)} triplets ({len(val_queries)} unique queries)"
    )
    return train_triplets, val_triplets


# ============================================================
# 格式轉換
# ============================================================

def convert_to_training_format(triplet: Dict) -> Dict:
    """
    將 triplet 轉換為 sentence-transformers 的訓練格式。

    sentence-transformers 的 MultipleNegativesRankingLoss 接受:
        InputExample(texts=["query", "positive", "negative"])

    輸出 JSON 格式:
        {"texts": ["query_text", "positive_text", "hard_negative_text"]}

    為何選擇此格式:
    - 與 sentence-transformers 原生 API 直接對應
    - 可直接用 json.loads() 反序列化後建立 InputExample
    - 格式簡單，便於人工抽樣審查

    Args:
        triplet: 含有 "query", "positive", "hard_negative" 欄位的字典

    Returns:
        訓練格式字典，或 None (若資料不完整)
    """
    query = triplet.get("query", "")
    positive = triplet.get("positive", "")
    hard_negative = triplet.get("hard_negative", "")

    if not all([query, positive, hard_negative]):
        return None

    return {
        "texts": [query, positive, hard_negative],
        # 保留原始 id 方便追溯
        "id": triplet.get("id", ""),
    }


# ============================================================
# 統計摘要
# ============================================================

def compute_statistics(triplets: List[Dict], label: str = "") -> Dict:
    """
    計算資料集的基本統計資訊，輸出到日誌供人工審查。

    Args:
        triplets: triplet 列表
        label: 資料集標籤 (例如 "train", "val")

    Returns:
        統計資訊字典
    """
    if not triplets:
        logger.warning(f"[{label}] 空資料集，無法計算統計")
        return {}

    query_lengths = []
    positive_lengths = []
    neg_lengths = []

    for t in triplets:
        texts = t.get("texts", [])
        if len(texts) >= 3:
            query_lengths.append(len(texts[0]))
            positive_lengths.append(len(texts[1]))
            neg_lengths.append(len(texts[2]))

    def stats(lengths: List[int]) -> Dict:
        if not lengths:
            return {}
        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
        }

    stats_info = {
        "count": len(triplets),
        "query_char_length": stats(query_lengths),
        "positive_char_length": stats(positive_lengths),
        "negative_char_length": stats(neg_lengths),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"[{label}] 資料集統計:")
    logger.info(f"  Triplet 數量: {stats_info['count']}")
    if query_lengths:
        ql = stats_info["query_char_length"]
        logger.info(
            f"  Query 字元長度: min={ql['min']} | "
            f"mean={ql['mean']:.0f} | max={ql['max']}"
        )
    if positive_lengths:
        pl = stats_info["positive_char_length"]
        logger.info(
            f"  Positive 字元長度: min={pl['min']} | "
            f"mean={pl['mean']:.0f} | max={pl['max']}"
        )
    logger.info(f"{'='*50}\n")

    return stats_info


# ============================================================
# 主函數
# ============================================================

def run_data_processing(
    hard_negatives_path: Path,
    train_output_path: Path,
    val_output_path: Path,
    val_ratio: float = 0.1,
    max_per_query: int = 10,
    min_query_len: int = 5,
    max_query_len: int = 150,
    min_passage_len: int = 10,
    max_passage_len: int = 1000,
    seed: int = 42,
) -> Tuple[int, int]:
    """
    資料後處理的主入口函數。

    Args:
        hard_negatives_path: Hard Negative Mining 輸出的 JSONL 路徑 (階段 B 輸出)
        train_output_path: 訓練集輸出路徑 (JSONL)
        val_output_path: 驗證集輸出路徑 (JSONL)
        val_ratio: 驗證集佔比 (以 unique query 數量計算)
        max_per_query: 每個 query 最多保留幾個 triplets
        min_query_len: Query 最短字元數
        max_query_len: Query 最長字元數
        min_passage_len: Passage (positive/negative) 最短字元數
        max_passage_len: Passage 最長字元數
        seed: 隨機種子

    Returns:
        (train_count, val_count) 元組
    """
    logger.info(f"[資料處理] 開始")
    logger.info(f"  input:  {hard_negatives_path}")
    logger.info(f"  train:  {train_output_path}")
    logger.info(f"  val:    {val_output_path}")

    # ----------------------------------------------------------------
    # 1. 載入 triplets
    # ----------------------------------------------------------------
    if not hard_negatives_path.exists():
        raise FileNotFoundError(
            f"Hard negatives 資料不存在: {hard_negatives_path}\n"
            f"請先執行階段 B: make data-mining"
        )

    raw_triplets: List[Dict] = []
    with open(hard_negatives_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                raw_triplets.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失敗: {e}")

    logger.info(f"載入原始 triplets: {len(raw_triplets)} 筆")

    # ----------------------------------------------------------------
    # 2. 資料清洗
    # ----------------------------------------------------------------
    cleaned: List[Dict] = []
    filter_stats = defaultdict(int)

    for t in raw_triplets:
        query = t.get("query", "")
        positive = t.get("positive", "")
        hard_neg = t.get("hard_negative", "")

        # 過濾條件 (任一不符合即捨棄)
        if not is_valid_text(query, min_query_len, max_query_len):
            filter_stats["query_invalid"] += 1
            continue

        if not is_valid_text(positive, min_passage_len, max_passage_len):
            filter_stats["positive_invalid"] += 1
            continue

        if not is_valid_text(hard_neg, min_passage_len, max_passage_len):
            filter_stats["hard_neg_invalid"] += 1
            continue

        # 確保三者不完全相同 (hard_negative 不應等同於 positive)
        if hard_neg == positive:
            filter_stats["neg_equals_positive"] += 1
            continue

        cleaned.append(t)

    logger.info(
        f"清洗後: {len(cleaned)} 筆 (過濾掉 {len(raw_triplets) - len(cleaned)} 筆)\n"
        f"  過濾原因: {dict(filter_stats)}"
    )

    if len(cleaned) == 0:
        raise ValueError(
            "清洗後資料集為空！請檢查:\n"
            "  1. corpus.txt 的段落長度是否過短或過長\n"
            "  2. 合成的 query 品質是否過差\n"
            "  3. Hard Negative Mining 的 score_min/score_max 設定是否合理"
        )

    # ----------------------------------------------------------------
    # 3. 去重
    # ----------------------------------------------------------------
    deduped = deduplicate_by_query(cleaned, max_per_query=max_per_query)

    # ----------------------------------------------------------------
    # 4. 切分
    # ----------------------------------------------------------------
    train_triplets, val_triplets = split_train_val(
        deduped, val_ratio=val_ratio, seed=seed
    )

    # ----------------------------------------------------------------
    # 5. 格式轉換並輸出
    # ----------------------------------------------------------------
    def write_jsonl(triplets: List[Dict], output_path: Path, label: str) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for t in triplets:
                formatted = convert_to_training_format(t)
                if formatted is None:
                    continue
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"輸出 {label}: {count} 筆 → {output_path}")
        compute_statistics(
            [json.loads(json.dumps(formatted)) for t in triplets
             if (formatted := convert_to_training_format(t)) is not None],
            label=label,
        )
        return count

    train_count = write_jsonl(train_triplets, train_output_path, "train")
    val_count = write_jsonl(val_triplets, val_output_path, "val")

    logger.info(
        f"[資料處理] 完成\n"
        f"  Train: {train_count} 筆\n"
        f"  Val:   {val_count} 筆\n"
        f"[資料處理] ★ 下一步: 執行 LoRA 微調\n"
        f"          執行: make train"
    )

    return train_count, val_count
