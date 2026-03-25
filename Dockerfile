# ============================================================
# Domain-Adaptive Retrieval System — Dockerfile
# 基底映像: PyTorch 2.5.0 + CUDA 12.4 (支援 RTX 4060 Ada Lovelace)
# 建置指令: docker build -t retrieval-system .
# ============================================================

# ---- 階段 1: 基底環境 ----
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# 避免安裝過程中出現互動式提示
ENV DEBIAN_FRONTEND=noninteractive

# ---- 系統層依賴 ----
# build-essential: 部分 Python 套件 (如 pynvml) 需要 C 編譯器
# git: 版本控制 + HuggingFace 模型下載
# curl: 健康檢查 + Ollama 連線測試
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- 工作目錄 ----
WORKDIR /app

# ---- Python 依賴 (獨立複製，利用 Docker layer cache) ----
# 策略: 只要 requirements.txt 不變，此層就不會重建
# 這對開發階段頻繁修改程式碼但不改依賴時，可節省大量建置時間
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 專案程式碼 ----
COPY . .

# ---- 環境變數 ----
# PYTHONPATH: 確保所有模組可從專案根目錄 import
ENV PYTHONPATH=/app
# PYTHONUNBUFFERED: 強制 stdout/stderr 即時輸出，避免容器日誌延遲
ENV PYTHONUNBUFFERED=1
# HF_HOME: HuggingFace 模型快取路徑 (對應 docker-compose 的 volume 掛載)
ENV HF_HOME=/app/models/base_models

# ---- 預設啟動指令 ----
# 可被 docker-compose 的 command 或 docker run 的引數覆蓋
CMD ["python", "scripts/run_all.py"]

# ============================================================
# 建置注意事項:
#
# 1. 映像大小: 此映像基於 devel 版本，體積約 8-10 GB。
#    若僅需推論 (不需編譯 CUDA 擴充)，可改用 runtime 版:
#    FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
#    體積可縮減至 ~5 GB，但部分依賴可能需要預編譯 wheel。
#
# 2. 模型預下載: 若網路環境不穩定，可取消下方註解，
#    在建置階段預先下載模型 (會增加映像體積 ~300MB):
#    RUN python -c "\
#      from transformers import AutoModel, AutoTokenizer; \
#      AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5'); \
#      AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5'); \
#      AutoModel.from_pretrained('BAAI/bge-base-zh-v1.5'); \
#      AutoTokenizer.from_pretrained('BAAI/bge-base-zh-v1.5')"
#
# 3. NVIDIA 驅動需求: 宿主機 NVIDIA 驅動版本需 >= 525.60 (Linux)
#    或 >= 528.33 (Windows) 才能支援 CUDA 12.4。
#    Windows 11 需透過 WSL2 + Docker Desktop 使用 GPU 直通。
# ============================================================
