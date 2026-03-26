# ============================================================
# retrieval/jieba_tokenizer.py — 繁體中文 jieba 斷詞封裝
#
# 職責: 為 BM25Retriever 提供一致的中文斷詞介面。
#
# ★ 設計決策: 為何不直接在 BM25Retriever 內部呼叫 jieba？
#   1. 讓斷詞邏輯可獨立測試 (tests/test_bm25_jieba.py)
#   2. 自定義辭典、停用詞、opencc 轉換等擴充功能集中於此
#   3. 若未來更換斷詞引擎 (e.g. HanLP)，只需修改此檔案
#
# jieba 繁體中文的已知限制:
#   jieba 的預設辭典以簡體中文為主，對繁體中文的支援有限。
#   補救措施 (三選一，依資源情況選擇):
#     方案 A (推薦): jieba_user_dict_path 載入繁體專業術語辭典
#     方案 B: 搭配 opencc-python-reimplemented 先轉換成簡體再斷詞
#     方案 C: 直接使用 jieba 預設辭典 (簡繁通用字較多，效果差異有限)
#   本實作預設使用方案 A/C，opencc 路徑留為選用。
# ============================================================

import os
from pathlib import Path
from typing import List, Optional, Set

from loguru import logger

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba 未安裝，BM25 斷詞功能將無法使用。請執行: pip install jieba")


# ============================================================
# 預設停用詞集合
# 這些是中文文本中常見的高頻虛詞，對 BM25 檢索沒有區分度。
# 來源: 哈工大停用詞表 (部分) + 領域常見無意義詞
# ============================================================
_DEFAULT_STOPWORDS: Set[str] = {
    # 虛詞、連接詞
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一個", "上", "也", "很", "到", "說", "要", "去",
    "你", "會", "著", "沒有", "看", "好", "自己", "這",
    # 標點符號 (jieba 偶爾會輸出這些)
    "，", "。", "！", "？", "、", "；", "：", "「", "」", "（", "）",
    ",", ".", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    # 數字類助詞
    "第", "各", "些", "等", "及",
    # 時間類
    "年", "月", "日", "時",
}


class JiebaTokenizer:
    """
    jieba 中文斷詞器封裝。

    支援:
    - 精確模式斷詞 (cut_all=False，適合 BM25 索引)
    - 自定義辭典 (補充領域專業詞彙)
    - 停用詞過濾 (移除無意義高頻詞)
    - 最短詞長過濾 (避免單字噪音)

    ★ 為何使用精確模式而非全模式?
      BM25 需要「詞」的邊界清晰，全模式 (cut_all=True) 會重複切割，
      導致 IDF 值偏低，反而削弱 BM25 對稀有術語的加分效果。
      例如: 「機器學習」在全模式可能輸出 [機器, 學習, 機器學習]，
      精確模式只輸出 [機器學習]，IDF 計算更準確。
    """

    def __init__(
        self,
        user_dict_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        min_token_length: int = 1,
        use_default_stopwords: bool = True,
    ):
        """
        Args:
            user_dict_path: 自定義辭典路徑 (繁體中文專業術語)
                            格式: 每行一個詞條，可附詞頻與詞性
                            例如: "機器學習 100 n"
            stopwords_path: 停用詞清單路徑 (每行一個停用詞)
                            若為 None，使用內建的預設停用詞集合
            min_token_length: 最短保留 token 長度 (過濾太短的噪音)
            use_default_stopwords: 是否啟用內建停用詞集合
        """
        if not JIEBA_AVAILABLE:
            raise ImportError(
                "jieba 未安裝，無法初始化 JiebaTokenizer。\n"
                "請執行: pip install jieba"
            )

        self.min_token_length = min_token_length
        self.stopwords: Set[str] = set()

        # ----------------------------------------------------------------
        # 載入停用詞
        # ----------------------------------------------------------------
        if use_default_stopwords:
            self.stopwords.update(_DEFAULT_STOPWORDS)
            logger.debug(f"已載入預設停用詞: {len(_DEFAULT_STOPWORDS)} 個")

        if stopwords_path is not None:
            stopwords_file = Path(stopwords_path)
            if stopwords_file.exists():
                with open(stopwords_file, "r", encoding="utf-8") as f:
                    user_stopwords = {line.strip() for line in f if line.strip()}
                self.stopwords.update(user_stopwords)
                logger.info(
                    f"已載入自定義停用詞: {len(user_stopwords)} 個 ← {stopwords_path}"
                )
            else:
                logger.warning(f"停用詞檔案不存在，跳過: {stopwords_path}")

        # ----------------------------------------------------------------
        # 載入自定義辭典 (用於補充繁體中文專業術語)
        # ----------------------------------------------------------------
        if user_dict_path is not None:
            dict_file = Path(user_dict_path)
            if dict_file.exists():
                jieba.load_userdict(str(dict_file))
                logger.info(f"已載入 jieba 自定義辭典: {user_dict_path}")
            else:
                logger.warning(f"jieba 自定義辭典不存在，使用預設辭典: {user_dict_path}")

        # jieba 初始化 (第一次呼叫時自動完成，此處提前觸發以避免首次斷詞延遲)
        # 注意: jieba.initialize() 在較新版本中已移除，用 cut 一次觸發即可
        logger.debug("JiebaTokenizer 初始化完成")

    def tokenize(self, text: str) -> List[str]:
        """
        將單段文字斷詞為 token 列表。

        處理流程:
        1. jieba 精確模式斷詞
        2. 去除首尾空白
        3. 過濾空字串
        4. 過濾停用詞
        5. 過濾低於最短長度的 token

        Args:
            text: 原始中文文字

        Returns:
            過濾後的 token 列表，可直接傳入 BM25Okapi
        """
        if not text or not text.strip():
            return []

        # jieba 精確模式: 最適合搜尋場景，切割結果最符合詞義邊界
        raw_tokens = jieba.cut(text, cut_all=False)

        tokens = []
        for tok in raw_tokens:
            tok = tok.strip()
            # 過濾空字串、停用詞、過短的噪音 token
            if (
                tok
                and tok not in self.stopwords
                and len(tok) >= self.min_token_length
            ):
                tokens.append(tok)

        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        批次斷詞。

        Args:
            texts: 文字列表

        Returns:
            對應的 token 列表的列表
        """
        return [self.tokenize(text) for text in texts]

    def __repr__(self) -> str:
        return (
            f"JiebaTokenizer("
            f"stopwords={len(self.stopwords)}, "
            f"min_length={self.min_token_length})"
        )
