# ============================================================
# data_pipeline/hard_negative_miner.py — 困難負例挖掘器
#
# 功能: 對合成的 (query, positive) 配對，挖掘困難負例 (hard negatives)，
#       最終輸出 (query, positive, hard_negative) triplets。
#
# 雙軌挖掘策略:
#   Track 1: Semantic Mining  — 用 SentenceTransformer 計算 dense embedding 相似度
#   Track 2: BM25 Mining      — 用 BM25Okapi + jieba 計算詞彙匹配分數
#   兩軌挖掘結果取聯集並去重，提升 hard negative 的多樣性
#
# ★ VRAM 時間分時 (本模組最重要的工程約束):
#   - 進入本模組前，Ollama 服務必須已停止 (VRAM 已釋放)
#   - Embedding 模型載入後，相似度計算完畢即應立即釋放 (del + gc + empty_cache)
#   - 嚴禁 LLM 模型與 Embedding 模型同時存在於 GPU 上
#
# ★ 時間複雜度防禦 (O(N) 降級防禦):
#   Corpus 的 Dense Embedding 必須在外部迴圈「整批預先計算並快取」，
#   嚴禁在遍歷 Query 的內部迴圈中重複呼叫 model.encode(corpus)。
#   若 corpus 有 N 個段落、Q 個 query，正確做法的複雜度為:
#     - 編碼: O(N + Q) 次 forward pass
#     - 相似度: O(Q × N) 次乘法 (用矩陣運算批次完成)
#   錯誤做法 (在 query 迴圈內 encode corpus) 的複雜度為 O(Q × N) 次 forward pass，
#   在 N=1000, Q=200 的規模下，後者比前者慢 200 倍。
#
# ★ BM25 語言防禦:
#   BM25Okapi 需要 tokenized (分詞後) 的文字列表作為輸入。
#   中文文字嚴禁使用 .split() (只能切空白，對中文完全無效)，
#   必須使用 jieba 分詞器，並搭配停用詞過濾。
# ============================================================

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# ★ BM25 挖掘的語言防禦: 強制匯入 jieba，嚴禁退化為 .split()
# 若 jieba 未安裝，此處會直接 ImportError，明確告知問題所在
import jieba
import jieba.analyse

# rank-bm25 套件提供 BM25Okapi 實作
from rank_bm25 import BM25Okapi

# ★ VRAM 分時: sentence_transformers 匯入在此，但模型載入/卸載由呼叫方控制
from sentence_transformers import SentenceTransformer

# 引入 VRAM 守衛
from data_pipeline.vram_guard import VRAMGuard


# ============================================================
# jieba 分詞封裝
# ============================================================

def tokenize_chinese(
    text: str,
    stopwords: Optional[set] = None,
    use_paddle: bool = False,
) -> List[str]:
    """
    使用 jieba 對中文文字進行分詞，並過濾停用詞。

    ★ BM25 語言防禦核心:
    BM25Okapi 的 tokenized_corpus 參數期望每個文件是一個 token list。
    中文文字「我愛機器學習」用 .split() 只得到 ["我愛機器學習"]，
    完全無法讓 BM25 識別個別詞彙。
    必須用 jieba 分詞得到 ["我", "愛", "機器", "學習"]。

    繁體中文特別說明:
    jieba 預設辭典為簡體中文。繁體中文在沒有自定義辭典的情況下，
    分詞效果可能次優 (例如繁體「資訊」可能被切成「資」「訊」)。
    兩種解決方案:
    1. 提供 jieba_user_dict_path 載入繁體辭典
    2. 使用 opencc 將繁體轉簡體後分詞，再轉回繁體 (本專案目前未啟用)

    Args:
        text: 待分詞的中文文字
        stopwords: 停用詞集合 (若為 None 則不過濾)
        use_paddle: 是否使用 PaddlePaddle 精確模式 (需額外安裝 paddlepaddle)

    Returns:
        分詞後的 token 列表 (已過濾停用詞與空白)
    """
    if use_paddle:
        # PaddlePaddle 模式分詞精度更高，但需要額外安裝依賴
        # 本專案預設不啟用，避免增加 Docker 映像大小
        jieba.enable_paddle()
        tokens = list(jieba.cut(text, use_paddle=True))
    else:
        # 精確模式: 將句子最精確地切開 (適合文字分析)
        tokens = list(jieba.cut(text, cut_all=False))

    # 過濾: 去除空白 token 與停用詞
    tokens = [t.strip() for t in tokens if t.strip()]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    return tokens


def load_stopwords(stopwords_path: Optional[str]) -> set:
    """
    載入停用詞檔案。每行一個停用詞，#開頭為註解。

    若路徑為 None 或檔案不存在，回傳預設的基本中文停用詞集合。
    預設停用詞僅包含最高頻的虛詞，避免過度過濾影響 BM25 效果。
    """
    if stopwords_path and Path(stopwords_path).exists():
        with open(stopwords_path, "r", encoding="utf-8") as f:
            words = set()
            for line in f:
                word = line.strip()
                if word and not word.startswith("#"):
                    words.add(word)
        logger.info(f"載入停用詞: {len(words)} 個 (來源: {stopwords_path})")
        return words

    # 預設基本停用詞: 僅包含最常見的功能詞
    # 刻意保守 (少量停用詞)，因為 BM25 本身的 IDF 已能降低高頻詞的影響
    default_stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就",
        "不", "人", "都", "一", "一個", "上", "也", "很",
        "到", "說", "要", "去", "你", "會", "著", "沒有",
        "看", "好", "自己", "這", "那", "來", "時",
        # 標點符號
        "，", "。", "、", "；", "：", "？", "！", "…",
        "（", "）", "「", "」", "『", "』",
    }
    logger.info(f"使用預設停用詞: {len(default_stopwords)} 個")
    return default_stopwords


# ============================================================
# Semantic Hard Negative Mining
# ============================================================

def mine_semantic_hard_negatives(
    queries: List[str],
    positives: List[str],
    corpus: List[str],
    model: SentenceTransformer,
    hard_neg_per_query: int = 5,
    score_min: float = 0.3,
    score_max: float = 0.8,
    encode_batch_size: int = 32,
    similarity_chunk_size: int = 1000,
) -> List[List[str]]:
    """
    Semantic Hard Negative Mining: 基於 dense embedding 相似度挖掘困難負例。

    ★ O(N) 降級防禦 — 核心效能保障:
    Corpus embedding 在此函數開頭「整批」預先計算，並保存在 corpus_embeddings 中。
    整個 Query 遍歷迴圈中，「corpus_embeddings 只會被讀取，絕不重新計算」。

    若違反此原則 (在 for query in queries 迴圈內呼叫 model.encode(corpus))，
    在 N=1000 corpus、Q=200 query 的規模下，
    Forward pass 次數: 錯誤做法=200×1000=200,000 次 vs 正確做法=1,000+200=1,200 次
    效能差距約 166 倍，且 VRAM 使用量也會持續累積。

    ★ 分塊相似度計算 — 防止 OOM:
    不直接計算 (Q × N) 的完整相似度矩陣 (Q=1000, N=10000 時矩陣大小=400MB)，
    而是以 similarity_chunk_size 為單位，逐塊計算子矩陣。
    每次只有 (chunk_size × N) 的矩陣存在記憶體中。

    Args:
        queries: Query 文字列表
        positives: 對應的正例段落列表 (長度與 queries 相同)
        corpus: 全部語料段落列表 (用於挖掘負例)
        model: 已載入的 SentenceTransformer 模型 (由呼叫方管理生命週期)
        hard_neg_per_query: 每個 query 最多挖掘幾個負例
        score_min: 負例相似度下限 (低於此值 = easy negative，無訓練價值)
        score_max: 負例相似度上限 (高於此值 = false negative，可能是真正答案)
        encode_batch_size: 編碼時的 batch size
        similarity_chunk_size: 相似度矩陣的 query 方向分塊大小

    Returns:
        List of hard_negatives per query: [[neg1, neg2, ...], [neg1, ...], ...]
        長度與 queries 相同，若某個 query 找不到負例則對應空列表
    """
    num_queries = len(queries)
    num_corpus = len(corpus)
    logger.info(
        f"Semantic Mining 開始 | queries={num_queries}, corpus={num_corpus}, "
        f"encode_batch_size={encode_batch_size}"
    )

    # ================================================================
    # ★ 核心: Corpus Embedding 整批預計算 (外部迴圈，只執行一次)
    # 這是 O(N) 防禦的實作點。以下這行「必須在 for loop 外面」。
    # ================================================================
    logger.info(f"預計算 corpus embeddings (共 {num_corpus} 筆，一次性操作)...")
    corpus_embeddings = model.encode(
        corpus,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 正規化後，dot product 等同 cosine similarity
        convert_to_numpy=True,      # 轉為 numpy 以節省 GPU 記憶體
    )
    # corpus_embeddings.shape = (num_corpus, hidden_size)
    logger.info(
        f"Corpus embeddings 預計算完成: shape={corpus_embeddings.shape}, "
        f"dtype={corpus_embeddings.dtype}"
    )

    # Query embeddings 同樣整批預計算
    logger.info(f"預計算 query embeddings (共 {num_queries} 筆)...")
    query_embeddings = model.encode(
        queries,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    # query_embeddings.shape = (num_queries, hidden_size)
    logger.info(f"Query embeddings 預計算完成: shape={query_embeddings.shape}")

    # 建立 positive passage → corpus index 的映射，用於排除正例
    positive_to_corpus_idx: Dict[str, int] = {}
    corpus_text_to_idx: Dict[str, int] = {text: i for i, text in enumerate(corpus)}
    for positive in positives:
        if positive in corpus_text_to_idx:
            positive_to_corpus_idx[positive] = corpus_text_to_idx[positive]

    # ================================================================
    # 分塊相似度計算
    # ================================================================
    # 矩陣分塊策略: 一次計算 (chunk_size × num_corpus) 的子矩陣，
    # 而非一次性建立 (num_queries × num_corpus) 的完整矩陣。
    # 以 similarity_chunk_size=1000 為例:
    # - 完整矩陣: 1000 queries × 10000 corpus = 10M float32 = 40MB (可能 OOM)
    # - 分塊矩陣: 1000 × 10000 = 40MB (每塊，在 CPU 上可接受)
    #
    # 注意: corpus_embeddings 仍然是完整的 (num_corpus, hidden_size) 矩陣，
    # 只有 query 方向做分塊，因此記憶體峰值是 O(chunk_size × N)。

    results: List[List[str]] = [[] for _ in range(num_queries)]

    for chunk_start in range(0, num_queries, similarity_chunk_size):
        chunk_end = min(chunk_start + similarity_chunk_size, num_queries)
        chunk_query_embs = query_embeddings[chunk_start:chunk_end]  # (chunk, hidden)

        # 計算此批 query 與全部 corpus 的相似度
        # 因為已 L2 normalize，dot product = cosine similarity
        # 結果 shape: (chunk_size, num_corpus)
        sim_matrix = np.dot(chunk_query_embs, corpus_embeddings.T)

        # 逐個 query 挖掘 hard negatives
        for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
            sims = sim_matrix[local_idx]  # (num_corpus,) 相似度陣列

            # 取得此 query 對應的正例索引 (需從結果中排除)
            positive_text = positives[global_idx]
            positive_corpus_idx = positive_to_corpus_idx.get(positive_text, -1)

            # 尋找符合 [score_min, score_max] 範圍的候選
            candidate_indices = np.where(
                (sims >= score_min) & (sims <= score_max)
            )[0]

            # 排除正例本身
            if positive_corpus_idx >= 0:
                candidate_indices = candidate_indices[
                    candidate_indices != positive_corpus_idx
                ]

            if len(candidate_indices) == 0:
                continue

            # 按相似度降序排序，取分數最高的幾個 (最困難的負例)
            # 高相似度 = 「看起來像答案但實際上不是」= 最有訓練價值的負例
            sorted_candidate_indices = candidate_indices[
                np.argsort(sims[candidate_indices])[::-1]
            ]
            top_negatives = sorted_candidate_indices[:hard_neg_per_query]
            results[global_idx] = [corpus[i] for i in top_negatives]

        logger.info(
            f"Semantic Mining 進度: {chunk_end}/{num_queries} "
            f"({chunk_end/num_queries*100:.1f}%)"
        )

    found_count = sum(len(r) for r in results)
    logger.info(
        f"Semantic Mining 完成 | "
        f"共找到 {found_count} 個 hard negatives "
        f"(平均每 query {found_count/num_queries:.1f} 個)"
    )
    return results


# ============================================================
# BM25 Hard Negative Mining
# ============================================================

def mine_bm25_hard_negatives(
    queries: List[str],
    positives: List[str],
    corpus: List[str],
    hard_neg_per_query: int = 5,
    stopwords: Optional[set] = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> List[List[str]]:
    """
    BM25 Hard Negative Mining: 基於詞彙匹配分數挖掘困難負例。

    ★ BM25 語言防禦核心:
    BM25Okapi 需要 List[List[str]] 作為 corpus 輸入，
    即每個文件必須事先分詞為 token 列表。

    嚴禁使用的錯誤做法:
        tokenized_corpus = [doc.split() for doc in corpus]
        # ↑ 對中文文字完全無效: "資訊檢索" → ["資訊檢索"] (整句當成一個詞)

    正確做法 (本函數實作):
        tokenized_corpus = [tokenize_chinese(doc, stopwords) for doc in corpus]
        # ↑ jieba 分詞: "資訊檢索" → ["資訊", "檢索"]

    BM25 挖掘的價值:
    Semantic Mining 擅長找到「語意相近」的困難負例，
    BM25 Mining 擅長找到「關鍵詞重疊但語意不同」的困難負例。
    兩者互補，覆蓋不同類型的困難負例。

    效能備忘錄:
    - jieba 預設為單執行緒，大型語料下可能較慢
    - BM25Okapi 建立索引的時間複雜度為 O(N)，查詢為 O(Q × V) (V 為詞彙量)
    - 相比 dense retrieval，BM25 不需 GPU，對 VRAM 零佔用

    Args:
        queries: Query 文字列表
        positives: 對應的正例段落列表
        corpus: 全部語料段落列表
        hard_neg_per_query: 每個 query 最多挖掘幾個負例
        stopwords: 停用詞集合
        bm25_k1: BM25 詞頻飽和度參數 (預設 1.5)
        bm25_b: BM25 文件長度正規化參數 (預設 0.75)

    Returns:
        每個 query 的 BM25 hard negatives 列表
    """
    logger.info(
        f"BM25 Mining 開始 | queries={len(queries)}, corpus={len(corpus)}"
    )

    # ================================================================
    # ★ BM25 語言防禦: 使用 jieba 對 corpus 分詞 (嚴禁 .split())
    # ================================================================
    logger.info("對 corpus 進行 jieba 中文分詞 (建立 BM25 索引)...")
    tokenized_corpus: List[List[str]] = []
    for doc in corpus:
        tokens = tokenize_chinese(doc, stopwords=stopwords)
        # 邊界保護: 若分詞結果為空 (例如段落全是標點符號)，給一個佔位符
        # BM25Okapi 不接受空列表
        if not tokens:
            tokens = ["[EMPTY]"]
        tokenized_corpus.append(tokens)

    # 建立 BM25 索引
    # BM25Okapi 在初始化時計算 IDF，此後查詢不再重新計算
    bm25_index = BM25Okapi(tokenized_corpus, k1=bm25_k1, b=bm25_b)
    logger.info(
        f"BM25 索引建立完成 | 詞彙量: {len(bm25_index.idf)} 個唯一 token"
    )

    # 建立 corpus 文字到索引的映射 (用於排除正例)
    corpus_text_to_idx: Dict[str, int] = {text: i for i, text in enumerate(corpus)}

    # ================================================================
    # 對每個 query 進行 BM25 檢索，取 top-k 作為候選
    # ================================================================
    results: List[List[str]] = []

    for q_idx, (query, positive) in enumerate(zip(queries, positives)):
        # ★ BM25 語言防禦: Query 同樣使用 jieba 分詞
        tokenized_query = tokenize_chinese(query, stopwords=stopwords)
        if not tokenized_query:
            results.append([])
            continue

        # 取得每個 corpus 文件對此 query 的 BM25 分數
        # 時間複雜度: O(V) per query (V = query token 數量)
        scores = bm25_index.get_scores(tokenized_query)  # numpy array, shape=(num_corpus,)

        # 找出正例的索引 (需排除)
        positive_idx = corpus_text_to_idx.get(positive, -1)

        # 取分數最高的 hard_neg_per_query × 2 個候選 (多取一些以備過濾)
        top_k = hard_neg_per_query * 2
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 過濾: 排除正例本身，並確保分數 > 0 (完全無詞彙重疊的負例對 BM25 Mining 無意義)
        hard_negs = []
        for idx in top_indices:
            if idx == positive_idx:
                continue
            if scores[idx] <= 0:
                continue
            hard_negs.append(corpus[idx])
            if len(hard_negs) >= hard_neg_per_query:
                break

        results.append(hard_negs)

        if (q_idx + 1) % 100 == 0:
            logger.info(f"BM25 Mining 進度: {q_idx + 1}/{len(queries)}")

    found_count = sum(len(r) for r in results)
    logger.info(
        f"BM25 Mining 完成 | "
        f"共找到 {found_count} 個 hard negatives "
        f"(平均每 query {found_count/len(queries):.1f} 個)"
    )
    return results


# ============================================================
# 主挖掘協調器
# ============================================================

def run_hard_negative_mining(
    synthetic_data_path: Path,
    corpus_path: Path,
    output_path: Path,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    hard_neg_per_query: int = 5,
    score_min: float = 0.3,
    score_max: float = 0.8,
    encode_batch_size: int = 32,
    similarity_chunk_size: int = 1000,
    stopwords_path: Optional[str] = None,
    jieba_user_dict_path: Optional[str] = None,
    post_release_max_mb: int = 200,
) -> int:
    """
    Hard Negative Mining 的主入口函數。

    完整流程:
    1. 載入合成資料 (query-positive 配對)
    2. 載入 corpus (全部段落)
    3. ★ VRAMGuard.assert_safe_to_proceed() 確認 VRAM 已清空
    4. 載入 Embedding 模型 (GPU)
    5. Semantic Mining (整批預計算 corpus embeddings，分塊計算相似度)
    6. ★ del model + gc.collect() + empty_cache() (三步釋放)
    7. BM25 Mining (純 CPU，jieba 分詞，此時 GPU 已釋放)
    8. 合併兩軌結果，輸出 triplets JSONL

    Args:
        synthetic_data_path: 合成配對的 JSONL 路徑 (階段 A 的輸出)
        corpus_path: 原始 corpus.txt 路徑
        output_path: 輸出 triplets 的 JSONL 路徑
        model_name: 用於 semantic mining 的 embedding 模型
        device: 'cuda' 或 'cpu'
        cache_dir: HuggingFace 模型快取目錄
        hard_neg_per_query: 每個 query 最多挖掘幾個負例 (semantic + bm25 各自)
        score_min: Semantic 負例相似度下限
        score_max: Semantic 負例相似度上限
        encode_batch_size: Embedding 編碼 batch size
        similarity_chunk_size: 相似度矩陣分塊大小 (query 方向)
        stopwords_path: 停用詞檔案路徑 (None = 使用預設停用詞)
        jieba_user_dict_path: jieba 自定義辭典路徑 (None = 使用預設辭典)
        post_release_max_mb: VRAM 安全斷言閾值 (MB)

    Returns:
        成功生成的 triplet 數量
    """
    # ----------------------------------------------------------------
    # 1. 載入合成資料
    # ----------------------------------------------------------------
    logger.info(f"[階段 B] Hard Negative Mining 開始")
    logger.info(f"  synthetic_data: {synthetic_data_path}")
    logger.info(f"  corpus:         {corpus_path}")
    logger.info(f"  output:         {output_path}")

    if not synthetic_data_path.exists():
        raise FileNotFoundError(
            f"合成資料不存在: {synthetic_data_path}\n"
            f"請先執行階段 A: make data-synthetic"
        )

    pairs: List[Dict] = []
    with open(synthetic_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # 跳過生成失敗的記錄 (query 為 null)
            if record.get("query") is None:
                continue
            pairs.append(record)

    logger.info(f"載入合成配對: {len(pairs)} 筆 (已排除生成失敗的記錄)")

    # ----------------------------------------------------------------
    # 2. 載入 corpus
    # ----------------------------------------------------------------
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    logger.info(f"載入 corpus: {len(corpus)} 個段落")

    queries = [p["query"] for p in pairs]
    positives = [p["positive"] for p in pairs]

    # ----------------------------------------------------------------
    # 3. VRAM 安全斷言: 確認 Ollama 已停止，VRAM 已清空
    # ----------------------------------------------------------------
    guard = VRAMGuard(post_release_max_mb=post_release_max_mb)
    guard.assert_safe_to_proceed("mining (階段 B 開始前)")

    # ----------------------------------------------------------------
    # 4 + 5 + 6. Semantic Mining: 載入模型 → 挖掘 → 釋放
    # ----------------------------------------------------------------
    # 載入自定義 jieba 辭典 (在載入 embedding 模型前完成，避免混淆)
    stopwords = load_stopwords(stopwords_path)
    if jieba_user_dict_path and Path(jieba_user_dict_path).exists():
        jieba.load_userdict(jieba_user_dict_path)
        logger.info(f"已載入 jieba 自定義辭典: {jieba_user_dict_path}")

    semantic_results: List[List[str]] = []

    with guard.phase("semantic_mining", budget_mb=3000):
        logger.info(f"載入 Embedding 模型: {model_name} (device={device})")
        guard.log_vram_status("載入模型前")

        # 使用 sentence_transformers 封裝，自動處理 tokenization 與 pooling
        model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir,
        )

        guard.log_vram_status("模型載入後")

        semantic_results = mine_semantic_hard_negatives(
            queries=queries,
            positives=positives,
            corpus=corpus,
            model=model,
            hard_neg_per_query=hard_neg_per_query,
            score_min=score_min,
            score_max=score_max,
            encode_batch_size=encode_batch_size,
            similarity_chunk_size=similarity_chunk_size,
        )

        # ★ 三步釋放協議: 在 with 塊內部主動釋放 (不等到 finally)
        # 早釋放 = VRAM 更快可用 = 降低 OOM 風險
        logger.info("★ 執行三步 VRAM 釋放協議...")
        del model      # Step 1: 解除 Python 參照
        guard.release_vram()  # Step 2+3: gc.collect() + empty_cache()
        guard.log_vram_status("Semantic Mining 釋放後")

    # ----------------------------------------------------------------
    # 7. BM25 Mining (純 CPU，Embedding 模型已釋放)
    # ----------------------------------------------------------------
    # 此時 GPU VRAM 應已清空，BM25 完全在 CPU 上執行
    guard.log_vram_status("BM25 Mining 開始前 (應接近 0)")

    bm25_results = mine_bm25_hard_negatives(
        queries=queries,
        positives=positives,
        corpus=corpus,
        hard_neg_per_query=hard_neg_per_query,
        stopwords=stopwords,
    )

    # ----------------------------------------------------------------
    # 8. 合併兩軌結果，輸出 triplets JSONL
    # ----------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    triplet_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, pair in enumerate(pairs):
            # 合併 semantic 與 BM25 的結果，取聯集並去重
            sem_negs = semantic_results[idx] if idx < len(semantic_results) else []
            bm25_negs = bm25_results[idx] if idx < len(bm25_results) else []

            # 去重: 保持順序的聯集 (優先保留 semantic 的結果)
            seen: set = set()
            merged_negs: List[str] = []
            for neg in sem_negs + bm25_negs:
                if neg not in seen and neg != pair["positive"]:
                    seen.add(neg)
                    merged_negs.append(neg)

            # 為每個負例建立一條 triplet 記錄
            for neg_idx, hard_neg in enumerate(merged_negs):
                triplet = {
                    "id": f"{pair['id']}_neg{neg_idx:02d}",
                    "query": pair["query"],
                    "positive": pair["positive"],
                    "hard_negative": hard_neg,
                    # 標記來源，方便事後分析哪個策略效果更好
                    "neg_source": (
                        "semantic" if hard_neg in sem_negs else "bm25"
                    ),
                }
                out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                triplet_count += 1

    logger.info(
        f"[階段 B] 完成 | 共 {triplet_count} 個 triplets → {output_path}"
    )
    logger.info(
        f"[階段 B] ★ 下一步: 執行資料後處理\n"
        f"          執行: make data-process"
    )

    return triplet_count
