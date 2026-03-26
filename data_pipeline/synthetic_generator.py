# ============================================================
# data_pipeline/synthetic_generator.py — LLM 合成資料生成器
#
# 功能: 讀取 corpus.txt 中的段落，透過 Ollama LLM 為每個段落
#       生成一個中文查詢 (query)，組成 (query, positive_passage) 配對。
#
# VRAM 策略 (階段 A):
#   本模組「完全不使用 GPU」。所有 LLM 推論由 Ollama 服務承擔，
#   Python 端只透過 HTTP REST API 呼叫 Ollama，零 VRAM 佔用。
#   這是 VRAM 時間分時架構的關鍵設計: 讓 LLM 佔用 GPU 的同時，
#   Python 進程無法（也不應）再載入任何 Embedding 模型。
#
# 輸出格式 (JSONL):
#   每行一個 JSON 物件: {"id": "...", "query": "...", "positive": "..."}
#   id 格式: "syn_{passage_index}_{retry_index}" 以確保唯一性
# ============================================================

import json
import re
import time
from pathlib import Path
from typing import Optional

from loguru import logger

# ollama 是純 HTTP 客戶端，不涉及任何 GPU 操作
try:
    import ollama as ollama_client
    OLLAMA_SDK_AVAILABLE = True
except ImportError:
    OLLAMA_SDK_AVAILABLE = False
    logger.warning("ollama SDK 未安裝，將退回使用 requests 直接呼叫 HTTP API")
    import requests


# ============================================================
# Prompt 模板
#
# 設計原則:
# 1. 明確指定「繁體中文」，避免 LLM 回覆簡體
# 2. 要求輸出格式為 JSON，便於解析
# 3. 設定查詢長度限制，避免生成過長的 query 導致截斷問題
#    (bge 系列 max_seq_length=256，query 本身不應過長)
# 4. 要求 query 必須能用該段落回答，確保 query-passage 相關性
#
# 已知局限:
# - LLM 有時會忽略 JSON 格式指令，需要後處理解析
# - qwen2.5 7B 量化模型的指令遵循能力低於完整版，retry 機率較高
# ============================================================

PROMPT_TEMPLATE = """你是一位資訊檢索資料集的標注專家，擅長繁體中文資料處理。

請根據以下段落，生成一個適合用於「資訊檢索任務」的繁體中文查詢 (query)。

要求:
1. 查詢必須使用繁體中文
2. 查詢長度在 10 到 50 個字之間
3. 查詢必須能夠用上述段落內容回答
4. 查詢應該是一個真實使用者可能會搜尋的問題或陳述
5. 不要直接複製段落中的句子，要提煉成自然的查詢語言
6. 僅輸出 JSON 格式，不要有任何其他文字

段落內容:
{passage}

請輸出以下 JSON 格式 (只輸出 JSON，不要有 markdown code block):
{{"query": "你生成的查詢"}}"""

# 解析失敗時的備用: 直接用 regex 從回應中萃取 query 欄位值
QUERY_EXTRACT_PATTERN = re.compile(r'"query"\s*:\s*"([^"]+)"')


class SyntheticDataGenerator:
    """
    基於 Ollama LLM 的合成 Query-Passage 配對生成器。

    架構說明:
    - 不持有任何 GPU 資源，全程 HTTP 通訊
    - 支援斷點續傳: 若輸出檔案已存在，跳過已處理的段落
    - 每個段落最多重試 max_retries 次，超過則寫入 null 記錄並繼續
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "qwen2.5:7b-instruct-q4_K_M",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_timeout: int = 120,
    ):
        """
        Args:
            ollama_host: Ollama 服務的 HTTP 地址
            model_name: Ollama 上已拉取的模型名稱。
                        建議 7B 量化版本 (q4_K_M)，13B 在 8GB VRAM 有 OOM 風險
            max_retries: 每個段落的最大重試次數
            retry_delay: 重試間隔 (秒)，給 LLM 服務恢復時間
            request_timeout: 單次 HTTP 請求的超時 (秒)，長段落生成需要較長時間
        """
        self.ollama_host = ollama_host.rstrip("/")
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout

        # 驗證 Ollama 服務是否可達
        self._check_ollama_connection()

    def _check_ollama_connection(self) -> None:
        """檢查 Ollama 服務是否在線，並確認目標模型已拉取。"""
        try:
            if OLLAMA_SDK_AVAILABLE:
                # 使用 SDK 時，client 是隱式管理的
                models = ollama_client.list()
                model_names = [m["model"] for m in models.get("models", [])]
            else:
                resp = requests.get(
                    f"{self.ollama_host}/api/tags",
                    timeout=10
                )
                resp.raise_for_status()
                model_names = [m["name"] for m in resp.json().get("models", [])]

            logger.info(f"Ollama 連線成功，已拉取的模型: {model_names}")

            # 確認目標模型存在
            # 注意: Ollama 的模型名稱可能帶 tag 或不帶，做前綴匹配
            base_name = self.model_name.split(":")[0]
            matching = [n for n in model_names if n.startswith(base_name)]
            if not matching:
                logger.warning(
                    f"目標模型 {self.model_name} 未在已拉取列表中找到。\n"
                    f"可用模型: {model_names}\n"
                    f"請執行: ollama pull {self.model_name}"
                )
            else:
                logger.info(f"目標模型確認可用: {matching[0]}")

        except Exception as e:
            raise ConnectionError(
                f"無法連線到 Ollama 服務 ({self.ollama_host})。\n"
                f"請確認:\n"
                f"  1. Ollama 容器正在運行: docker compose ps\n"
                f"  2. 端口未被防火牆封鎖: curl {self.ollama_host}/api/tags\n"
                f"原始錯誤: {e}"
            )

    def _call_llm(self, passage: str) -> Optional[str]:
        """
        呼叫 Ollama LLM 生成 query。

        Args:
            passage: 要為其生成查詢的段落文字

        Returns:
            生成的 query 字串，或 None (若解析失敗)
        """
        prompt = PROMPT_TEMPLATE.format(passage=passage.strip())

        try:
            if OLLAMA_SDK_AVAILABLE:
                response = ollama_client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": 0.7,   # 適度隨機性，避免生成重複 query
                        "top_p": 0.9,
                        "num_predict": 100,   # 限制輸出長度，JSON query 不需要很長
                    },
                )
                raw_text = response["response"]
            else:
                # 退回使用原始 HTTP API
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 100,
                    },
                }
                resp = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                raw_text = resp.json()["response"]

        except Exception as e:
            logger.warning(f"LLM API 呼叫失敗: {e}")
            return None

        return self._parse_query_from_response(raw_text)

    def _parse_query_from_response(self, raw_text: str) -> Optional[str]:
        """
        從 LLM 回應中解析出 query 字串。

        LLM 的回應格式不穩定，常見問題:
        1. 包含 markdown code block (```json ... ```)
        2. JSON 外有額外文字
        3. JSON key 使用不同大小寫

        解析策略:
        1. 先嘗試直接 json.loads (最乾淨的情況)
        2. 若失敗，嘗試去除 markdown code block 後解析
        3. 若仍失敗，用 regex 直接萃取 "query" 的值

        Args:
            raw_text: LLM 的原始輸出文字

        Returns:
            解析出的 query 字串，或 None
        """
        text = raw_text.strip()

        # 嘗試 1: 直接解析
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "query" in data:
                return str(data["query"]).strip()
        except json.JSONDecodeError:
            pass

        # 嘗試 2: 去除 markdown code block
        # 某些模型習慣在 JSON 外包 ```json ... ```
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "query" in data:
                return str(data["query"]).strip()
        except json.JSONDecodeError:
            pass

        # 嘗試 3: regex 直接萃取 query 值 (最後手段)
        match = QUERY_EXTRACT_PATTERN.search(text)
        if match:
            return match.group(1).strip()

        logger.debug(f"無法解析 LLM 回應: {text[:200]!r}")
        return None

    def _validate_query(self, query: str, passage: str) -> bool:
        """
        基本驗證: 確保 query 符合最低品質要求。

        目前驗證規則 (刻意保守，避免過濾掉好資料):
        - 長度在 5 到 150 個字元之間
        - 不是 passage 的直接複製 (相似度粗略判斷)
        - 不為空白

        已知限制:
        - 無法自動判斷「這個 query 是否真的能用 passage 回答」
        - 無法偵測 LLM 生成了簡體中文 (繁簡混用仍會通過驗證)
        """
        if not query or not query.strip():
            return False

        query = query.strip()

        # 長度檢查: 過短表示生成失敗，過長表示 LLM 忽略了長度限制
        if len(query) < 5 or len(query) > 150:
            logger.debug(f"Query 長度不符合要求 ({len(query)}): {query[:50]!r}")
            return False

        # 直接複製偵測: query 不應超過 50% 的字元與 passage 完全重疊
        # 使用最簡單的 set 交集估算，非精確 NLP 評估
        passage_chars = set(passage.replace(" ", "").replace("\n", ""))
        query_chars = set(query.replace(" ", ""))
        if len(query_chars) > 0:
            overlap_ratio = len(query_chars & passage_chars) / len(query_chars)
            if overlap_ratio > 0.95 and len(query) > len(passage) * 0.8:
                logger.debug(f"Query 疑似直接複製 passage: {query[:50]!r}")
                return False

        return True

    def generate(
        self,
        corpus_path: Path,
        output_path: Path,
        max_passages: Optional[int] = None,
        resume: bool = True,
    ) -> int:
        """
        主生成函數: 讀取 corpus.txt，為每個段落生成 query，輸出 JSONL。

        Args:
            corpus_path: corpus.txt 的路徑。格式: 每行一個段落，空行跳過
            output_path: 輸出的 JSONL 檔案路徑
            max_passages: 最多處理幾個段落 (None = 全部)。
                          開發測試時建議設為 50-100，避免浪費時間
            resume: 若輸出檔案已存在，是否跳過已處理的段落 (斷點續傳)

        Returns:
            成功生成的配對數量
        """
        # 讀取 corpus
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"corpus.txt 不存在: {corpus_path}\n"
                f"請將語料檔案放置於 data/raw/corpus.txt，"
                f"每行一個段落 (空行會被自動跳過)"
            )

        passages = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:  # 跳過空行
                    passages.append(stripped)

        logger.info(f"讀取 corpus 完成: {len(passages)} 個段落")

        if max_passages is not None:
            passages = passages[:max_passages]
            logger.info(f"已限制處理數量: {len(passages)} 個段落")

        # 斷點續傳: 讀取已處理的 passage id
        already_done_ids: set = set()
        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            already_done_ids.add(record.get("id", ""))
                        except json.JSONDecodeError:
                            pass
            logger.info(f"斷點續傳: 找到 {len(already_done_ids)} 筆已處理記錄")

        # 開始生成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success_count = 0
        fail_count = 0

        # 使用 append 模式，支援斷點續傳
        mode = "a" if resume and output_path.exists() else "w"

        with open(output_path, mode, encoding="utf-8") as out_f:
            for idx, passage in enumerate(passages):
                record_id = f"syn_{idx:06d}"

                # 斷點續傳: 跳過已處理的段落
                if record_id in already_done_ids:
                    success_count += 1  # 算入已完成數
                    continue

                # 重試機制: 最多嘗試 max_retries 次
                query = None
                for attempt in range(self.max_retries):
                    query = self._call_llm(passage)

                    if query and self._validate_query(query, passage):
                        break  # 生成成功，跳出重試迴圈

                    if attempt < self.max_retries - 1:
                        logger.debug(
                            f"段落 {idx} 第 {attempt + 1} 次嘗試失敗，"
                            f"{self.retry_delay}s 後重試..."
                        )
                        time.sleep(self.retry_delay)

                if query and self._validate_query(query, passage):
                    record = {
                        "id": record_id,
                        "query": query,
                        "positive": passage,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()  # 確保斷點續傳時不丟失資料
                    success_count += 1

                    if success_count % 10 == 0:
                        logger.info(
                            f"進度: {idx + 1}/{len(passages)} | "
                            f"成功: {success_count} | 失敗: {fail_count}"
                        )
                else:
                    # 生成失敗: 記錄失敗資訊，方便事後分析失敗段落
                    fail_record = {
                        "id": record_id,
                        "query": None,
                        "positive": passage,
                        "_error": "generation_failed",
                    }
                    out_f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    fail_count += 1
                    logger.warning(
                        f"段落 {idx} 生成失敗 (已嘗試 {self.max_retries} 次): "
                        f"{passage[:50]!r}"
                    )

        logger.info(
            f"生成完成 | 成功: {success_count} | 失敗: {fail_count} | "
            f"輸出: {output_path}"
        )
        return success_count


def run_synthetic_generation(
    corpus_path: Path,
    output_path: Path,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M",
    max_passages: Optional[int] = None,
    resume: bool = True,
) -> int:
    """
    合成資料生成的入口函數。供 scripts/run_data_pipeline.py 呼叫。

    VRAM 備忘錄:
        此函數執行期間，Ollama 服務會佔用約 4-6 GB VRAM。
        Python 進程本身的 VRAM 佔用為 0。
        函數結束後，必須在外部停止 Ollama 服務 (make stop-ollama)，
        才能釋放 VRAM 供下一個階段 (Hard Negative Mining) 使用。

    Args:
        corpus_path: 輸入語料路徑 (每行一個段落的純文字檔)
        output_path: 輸出 JSONL 路徑
        ollama_host: Ollama 服務位址
        ollama_model: 要使用的 LLM 模型名稱
        max_passages: 最多處理的段落數 (None = 全部)
        resume: 是否啟用斷點續傳

    Returns:
        成功生成的配對數量
    """
    logger.info(f"[階段 A] 合成資料生成")
    logger.info(f"  corpus: {corpus_path}")
    logger.info(f"  output: {output_path}")
    logger.info(f"  model:  {ollama_model}")
    logger.info(f"  注意: 此階段 Python 端 VRAM 使用量應為 0")

    generator = SyntheticDataGenerator(
        ollama_host=ollama_host,
        model_name=ollama_model,
    )

    count = generator.generate(
        corpus_path=corpus_path,
        output_path=output_path,
        max_passages=max_passages,
        resume=resume,
    )

    logger.info(f"[階段 A] 完成。生成 {count} 個配對")
    logger.info(
        f"[階段 A] ★ 下一步: 停止 Ollama 服務以釋放 VRAM\n"
        f"          執行: make stop-ollama"
    )

    return count
