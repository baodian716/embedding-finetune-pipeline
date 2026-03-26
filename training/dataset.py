# ============================================================
# training/dataset.py — 訓練資料集與 DataLoader 定義
#
# 資料格式 (來自 Step 2 data_processor.py 的輸出):
#   {"texts": ["query", "positive", "hard_negative"], "id": "..."}
#
# TripletDataset: 訓練用三元組資料集
# ValDataset:     評估用資料集 (建構 queries / corpus / relevant_docs 三結構)
# ============================================================

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 訓練資料集
# ============================================================

class TripletDataset(Dataset):
    """
    讀取 train.jsonl，回傳 (query, positive, hard_negative) 三元組。

    DataLoader 使用說明:
    - 回傳的每個 item 是三個純文字字串 (而非已 tokenize 的 tensor)。
    - Tokenization 在 LoRATrainer 的訓練迴圈內部完成，
      這樣可以在 batch 維度上 dynamic padding，節省計算量。
    - 若在此 Dataset 預先 tokenize，則 tensor 大小因樣本而異，
      無法直接 stack，需要更複雜的 collate_fn。

    query_instruction:
    - BGE v1.5 系列在 retrieval 任務中，query 需要加入前綴指令
    - 前綴在此 Dataset 層加入，確保一致性
    - 段落 (positive/hard_negative) 不加前綴
    """

    def __init__(
        self,
        data_path: Path,
        query_instruction: str = "",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: train.jsonl 的路徑
            query_instruction: 加在每個 query 前面的 BGE 任務指令前綴
                               例如: "为这个句子生成表示以用于检索相关文章："
            max_samples: 最多載入幾筆資料 (開發測試用，None = 全部)
        """
        self.query_instruction = query_instruction
        self.samples: List[Dict[str, str]] = []

        if not data_path.exists():
            raise FileNotFoundError(
                f"訓練資料不存在: {data_path}\n"
                f"請先執行 Step 2 資料管線: make data-process"
            )

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                texts = record.get("texts", [])
                if len(texts) < 3:
                    continue  # 格式不完整，跳過

                self.samples.append({
                    "query":         texts[0],
                    "positive":      texts[1],
                    "hard_negative": texts[2],
                    "id":            record.get("id", ""),
                })

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        logger.info(
            f"TripletDataset 載入完成: {len(self.samples)} 筆 ← {data_path}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        """
        回傳 (query, positive, hard_negative) 純文字三元組。
        query 已加上 BGE 任務前綴 (若有設定)。
        """
        s = self.samples[idx]
        query = self.query_instruction + s["query"] if self.query_instruction else s["query"]
        return query, s["positive"], s["hard_negative"]


def triplet_collate_fn(
    batch: List[Tuple[str, str, str]]
) -> Tuple[List[str], List[str], List[str]]:
    """
    DataLoader 的 collate 函數。

    將 [(q1, p1, n1), (q2, p2, n2), ...] 轉換為
    ([q1, q2, ...], [p1, p2, ...], [n1, n2, ...])

    回傳純文字 List 而非 Tensor，Tokenization 交由 Trainer 批次處理。
    這樣的好處: DataLoader worker 不需要 GPU，資料讀取與 GPU 計算可以並行。
    """
    queries, positives, hard_negatives = zip(*batch)
    return list(queries), list(positives), list(hard_negatives)


def create_train_dataloader(
    train_path: Path,
    query_instruction: str,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    建立訓練用 DataLoader。

    Args:
        train_path: train.jsonl 的路徑
        query_instruction: BGE 任務前綴指令
        batch_size: 真實 batch size (非有效 batch size)
                    對比學習的品質受此影響: batch 越大，in-batch negatives 越多
        num_workers: DataLoader 的工作執行緒數
                     Windows 環境建議設為 0 或 2，Docker Linux 可設為 4
        seed: 資料 shuffle 的隨機種子
        max_samples: 最多載入的樣本數 (None = 全部)

    Returns:
        DataLoader 物件
    """
    import torch

    dataset = TripletDataset(
        data_path=train_path,
        query_instruction=query_instruction,
        max_samples=max_samples,
    )

    # generator 用於確保 shuffle 結果可重現
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,               # 每 epoch 重新 shuffle，避免固定的順序偏差
        collate_fn=triplet_collate_fn,
        num_workers=num_workers,
        pin_memory=True,            # 將資料 pin 在 CPU 記憶體，加速 GPU 傳輸
        drop_last=True,             # 丟棄最後一個不完整的 batch
                                    # 對比學習需要固定 batch size 以確保負例數量一致
        generator=generator,
    )

    logger.info(
        f"Train DataLoader: {len(dataset)} 筆 | "
        f"batch_size={batch_size} | steps/epoch={len(dataloader)}"
    )
    return dataloader


# ============================================================
# 驗證資料集 (供評估器使用)
# ============================================================

class ValDataset:
    """
    讀取 val.jsonl，建構 IR 評估所需的三種結構。

    InformationRetrievalEvaluator 需要:
    - queries:      {query_id: query_text}
    - corpus:       {doc_id: doc_text}
    - relevant_docs: {query_id: Set[doc_id]}  ← 該 query 的相關文件集合

    從 val.jsonl 的三元組建構方式:
    - 每筆記錄有一個 query，一個 positive (相關文件)，一個 hard_negative
    - positive 加入 corpus 並標記為 relevant
    - hard_negative 也加入 corpus (增加評估難度)
    - 多個不同 hard_negative 可能對應同一個 positive (de-dup 處理)

    評估侷限說明:
    此評估使用的 corpus 只包含 val 集合中的段落，而非完整的 domain corpus。
    因此 MRR@10 / NDCG@10 數值會比真實 IR 評估高 (corpus 更小，任務更易)。
    其主要作用是在訓練中監控相對改善趨勢，而非絕對效能指標。
    """

    def __init__(self, val_path: Path):
        self.queries: Dict[str, str] = {}
        self.corpus: Dict[str, str] = {}
        self.relevant_docs: Dict[str, Set[str]] = {}

        if not val_path.exists():
            raise FileNotFoundError(f"驗證資料不存在: {val_path}")

        # 追蹤已加入 corpus 的段落文字 (去重用)
        passage_text_to_id: Dict[str, str] = {}
        query_count = 0

        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                texts = record.get("texts", [])
                if len(texts) < 2:
                    continue

                rid = record.get("id", f"val_{query_count:06d}")
                q_text = texts[0]
                p_text = texts[1]

                # 每個 query 文字只對應一個 query_id
                # (多個 hard_negative 可能來自同一個 query，需去重)
                # 使用 query 文字做 key 確保唯一性
                if q_text not in [v for v in self.queries.values()]:
                    q_id = f"q_{rid}"
                    self.queries[q_id] = q_text
                    self.relevant_docs[q_id] = set()
                else:
                    # 找到對應的 q_id
                    q_id = next(k for k, v in self.queries.items() if v == q_text)

                # 段落去重: 同一段落文字不重複加入 corpus
                if p_text not in passage_text_to_id:
                    p_id = f"p_{rid}"
                    self.corpus[p_id] = p_text
                    passage_text_to_id[p_text] = p_id
                else:
                    p_id = passage_text_to_id[p_text]

                self.relevant_docs[q_id].add(p_id)

                # hard_negative 也加入 corpus (使評估更具挑戰性)
                if len(texts) >= 3:
                    n_text = texts[2]
                    if n_text not in passage_text_to_id:
                        n_id = f"n_{rid}"
                        self.corpus[n_id] = n_text
                        passage_text_to_id[n_text] = n_id

                query_count += 1

        logger.info(
            f"ValDataset 載入完成: "
            f"{len(self.queries)} queries | "
            f"{len(self.corpus)} corpus | "
            f"{val_path}"
        )
