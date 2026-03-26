"""
pdf_to_corpus.py — PDF 轉 corpus.txt 通用清洗腳本

功能:
  1. 讀取專案根目錄下所有 .pdf（或指定路徑）
  2. 逐行智慧合併：偵測未結束的句子自動接合下一行，確保 Embedding 讀到完整上下文
  3. 多層噪音過濾（header/footer、TOC、圖表索引、版權、碎片表格…）
  4. 輸出一行一段落到 data/raw/corpus.txt

使用方式:
  python pdf_to_corpus.py                          # 自動掃描根目錄所有 .pdf
  python pdf_to_corpus.py "a.pdf" "b.pdf"          # 指定檔案
  python pdf_to_corpus.py --min-length 60           # 調整最小段落長度

依賴: pip install pdfminer.six
      （已收錄於 requirements.txt，Docker 容器內自動安裝）
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

from pdfminer.high_level import extract_text


# ============================================================
# 第一層：行級噪音判定
# ============================================================

_NOISE_PATTERNS = [
    # ----- Header / Footer -----
    re.compile(r"Phison aiDAPTIV\+?\s*Pro Suite", re.I),
    re.compile(r"aiDAPTIV\+?\s+Pro Suite.*User\s*guide", re.I),
    re.compile(r"^\d{1,3}\s*$"),                         # 純頁碼
    re.compile(r"^Page\s+\d+", re.I),                    # Page X of Y

    # ----- TOC（連續點 + 頁碼）-----
    re.compile(r"\.{4,}\s*\d*\s*$"),                     # "........ 27"
    re.compile(r"…{2,}"),

    # ----- 圖表索引行（帶頁碼的那種，內文引用不算）-----
    re.compile(r"^(Figure|Table|圖|表)\s*[\dA-Z][\d\-A-Z]*\s+.{0,80}$", re.I),

    # ----- 版權 / 法律 -----
    re.compile(r"ALL RIGHTS.{0,20}RESERVED", re.I),
    re.compile(r"SHALL NOT BE REPRODUCED", re.I),
    re.compile(r"©\s*\d{4}", re.I),
    re.compile(r"WITHOUT PERMISSION FROM", re.I),

    # ----- 聯絡資訊 -----
    re.compile(r"(Tel:|Fax:)\s*\+?\d", re.I),
    re.compile(r"E-mail:\s*\S+@\S+", re.I),

    # ----- 版本 metadata（短行）-----
    re.compile(r"^(Revision|Draft Date|Pro Suite Version|NWUN_)", re.I),

    # ----- 純數字 / 純標點 / 純空白 -----
    re.compile(r"^[\d\s\W]+$"),

    # ----- 殘留的編號碎片（如 "1. 2. 3. Dataset 4. 5. 6."）-----
    re.compile(r"^[\d\.\s]{6,}$"),

    # ----- 純 URL 或 bash 單行指令（不構成可讀段落）-----
    re.compile(r"^(https?://|bash\s+<\(curl|sudo\s+cp\s+|cd\s+\S+)"),
    re.compile(r"^download\s+meta-llama/", re.I),
]


def _is_noise_line(line: str) -> bool:
    """判斷單行是否為噪音。"""
    s = line.strip()
    if len(s) < 3:
        return True
    for pat in _NOISE_PATTERNS:
        if pat.search(s):
            return True
    return False


# ============================================================
# 第二層：逐行智慧合併（核心改進）
# ============================================================

# 判斷「行尾是否為完整句結束」的正則
_SENTENCE_END = re.compile(r"[.!?:;。！？：；)\]）」】]\s*$")

# 判斷「行首是否為新段落開頭」的正則
_NEW_PARA_START = re.compile(
    r"^("
    r"\d+\.\d*\s"              # 編號段落 "3.2. Creating..."
    r"|[A-Z][A-Z\s]{5,}"       # 全大寫標題 "ENVIRONMENT PREPARATION"
    r"|Note\s*:"               # 備註
    r"|Field description"      # 欄位說明
    r"|Step\s+\d"              # 步驟
    r")"
)


def _smart_merge_lines(raw_text: str) -> list[str]:
    """
    逐行掃描，用啟發式規則決定「接到上一段」或「開新段落」。

    規則：
      1. 空行 → 強制段落分隔
      2. 行首符合 _NEW_PARA_START → 開新段落
      3. 上一行結尾不是句末標點 且 當前行首為小寫字母 → 接續上一段
      4. 其餘情況 → 開新段落
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    paragraphs: list[str] = []
    current: list[str] = []

    def _flush():
        if current:
            joined = " ".join(current)
            # 壓縮多餘空白
            joined = re.sub(r"\s{2,}", " ", joined).strip()
            if joined:
                paragraphs.append(joined)
            current.clear()

    for line in lines:
        stripped = line.strip()

        # --- 空行 = 段落分隔 ---
        if not stripped:
            _flush()
            continue

        # --- 噪音行直接跳過（不接、不存）---
        if _is_noise_line(stripped):
            # 噪音行也會打斷段落接續
            _flush()
            continue

        # --- 連字號斷字修復 "configu-" + "ration" ---
        if current and current[-1].endswith("-"):
            current[-1] = current[-1][:-1] + stripped
            continue

        # --- 決定：接續上一段 or 開新段落 ---
        if not current:
            # 目前沒有累積段落，直接開新的
            current.append(stripped)
        elif _NEW_PARA_START.match(stripped):
            # 行首明確是新段落
            _flush()
            current.append(stripped)
        elif (
            not _SENTENCE_END.search(current[-1])
            and stripped[0].islower()
        ):
            # 上一行沒結束 + 本行小寫開頭 → 接續
            current.append(stripped)
        elif (
            not _SENTENCE_END.search(current[-1])
            and len(stripped) > 20
            and not stripped[0].isupper()
        ):
            # 上一行沒結束 + 本行非大寫開頭 + 夠長 → 可能是接續
            current.append(stripped)
        else:
            _flush()
            current.append(stripped)

    _flush()
    return paragraphs


# ============================================================
# 第三層：段落級清洗
# ============================================================

# 段落級噪音（整段匹配）
_PARA_NOISE = [
    re.compile(r"^Field description:\s*[\d\.\s,]+$"),          # "Field description: 1. 2. 3."
    re.compile(r"^(Previous|Next|Submit|Cancel|Reset)\s*:"),   # UI 按鈕描述碎片
    re.compile(r"^(Category|Detail)\s+\w.{0,30}$"),            # 表格碎片
]


def _clean_paragraphs(
    paragraphs: list[str],
    min_length: int = 40,
) -> list[str]:
    """段落級過濾 + 去重。"""
    cleaned = []
    seen: set[str] = set()

    for para in paragraphs:
        # 段落級噪音
        skip = False
        for pat in _PARA_NOISE:
            if pat.match(para):
                skip = True
                break
        if skip:
            continue

        # 最小長度
        if len(para) < min_length:
            continue

        # 去重（正規化比對）
        key = re.sub(r"\s+", " ", para.lower())
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(para)

    return cleaned


# ============================================================
# 主流程
# ============================================================

def extract_pdf(pdf_path: str) -> str:
    """抽取單一 PDF 全文。"""
    print(f"  讀取: {os.path.basename(pdf_path)}")
    try:
        text = extract_text(pdf_path)
        print(f"    原始字元數: {len(text):,}")
        return text
    except Exception as e:
        print(f"    [錯誤] 無法讀取: {e}", file=sys.stderr)
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="將 PDF 檔案轉換為乾淨的 corpus.txt（一行一段落，完整上下文）"
    )
    parser.add_argument(
        "pdfs", nargs="*",
        help="PDF 路徑。省略則自動掃描專案根目錄 *.pdf",
    )
    parser.add_argument(
        "--min-length", type=int, default=40,
        help="最小段落長度（字元），預設 40",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="輸出路徑，預設 data/raw/corpus.txt",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output_path = (
        Path(args.output) if args.output
        else project_root / "data" / "raw" / "corpus.txt"
    )

    # 收集 PDF
    if args.pdfs:
        pdf_files = [p for p in args.pdfs if os.path.isfile(p)]
    else:
        pdf_files = sorted(glob.glob(str(project_root / "*.pdf")))

    if not pdf_files:
        print("[錯誤] 找不到任何 PDF 檔案。", file=sys.stderr)
        print("  放 .pdf 檔到專案根目錄，或指定路徑: python pdf_to_corpus.py a.pdf b.pdf")
        sys.exit(1)

    print(f"找到 {len(pdf_files)} 個 PDF:")
    for f in pdf_files:
        print(f"  • {os.path.basename(f)}")
    print()

    # ---- 逐檔處理 ----
    all_paragraphs: list[str] = []

    for pdf_path in pdf_files:
        raw = extract_pdf(pdf_path)
        if not raw:
            continue

        merged = _smart_merge_lines(raw)
        print(f"    智慧合併後: {len(merged)} 段")

        cleaned = _clean_paragraphs(merged, min_length=args.min_length)
        print(f"    清洗後:     {len(cleaned)} 段")
        print()

        all_paragraphs.extend(cleaned)

    # ---- 跨檔全域去重 ----
    seen: set[str] = set()
    unique: list[str] = []
    for para in all_paragraphs:
        key = re.sub(r"\s+", " ", para.lower())
        if key not in seen:
            seen.add(key)
            unique.append(para)

    # ---- 寫入 ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique))

    size_kb = output_path.stat().st_size / 1024
    print("=" * 60)
    print(f"完成！共 {len(unique)} 個段落 → {output_path.name}")
    print(f"檔案大小: {size_kb:.1f} KB")
    print()

    # ---- 品質檢查 ----
    lengths = [len(p) for p in unique]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    short_count = sum(1 for l in lengths if l < 80)

    print(f"段落長度統計:")
    print(f"  平均: {avg_len:.0f} 字元")
    print(f"  最短: {min(lengths)} / 最長: {max(lengths)}")
    print(f"  < 80 字元的短段落: {short_count} ({short_count/len(unique)*100:.1f}%)")
    print()

    # ---- 前 5 段預覽 ----
    print("前 5 段預覽:")
    print("-" * 60)
    for i, para in enumerate(unique[:5], 1):
        preview = para[:150] + "..." if len(para) > 150 else para
        print(f"  [{i}] {preview}")
    print("-" * 60)
    print()
    print(f"✓ 輸出路徑: {output_path}")
    print(f"  下一步: 確認品質後執行 make data-synthetic 重跑管線")


if __name__ == "__main__":
    main()
