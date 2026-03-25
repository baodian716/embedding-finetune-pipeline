# Domain-Adaptive Retrieval System

領域自適應檢索系統：基於 LoRA 微調與混合搜索實務

## 專案概述

本專案實作一套完整的中文領域自適應檢索管線，包含：

- **合成資料生成** — 透過 Ollama LLM 自動產生 query-passage 訓練對
- **Hard Negative Mining** — Semantic + BM25 雙軌挖掘困難負例
- **LoRA 微調** — 在 `bge-small-zh-v1.5` 與 `bge-base-zh-v1.5` 上掛載 LoRA adapter
- **混合檢索** — Dense (GPU) + BM25/jieba (CPU) 雙軌檢索，RRF 融合排序
- **量化評估** — 四象限 (Small-Baseline, Small-LoRA, Base-Baseline, Base-LoRA) 的 MRR@10 / NDCG@10 對比

### 硬體需求

| 項目 | 最低需求 |
|------|---------|
| GPU | NVIDIA RTX 4060 8GB (或同等 VRAM) |
| RAM | 16 GB |
| 磁碟 | 20 GB 可用空間 (含 Docker 映像 + 模型) |
| 作業系統 | Windows 11 (WSL2) / Ubuntu 22.04+ |
| NVIDIA 驅動 | >= 528.33 (Windows) / >= 525.60 (Linux) |

---

## 環境建置 (Docker)

### 前置條件

1. **安裝 Docker Desktop** (Windows)
   - 下載: https://www.docker.com/products/docker-desktop/
   - 安裝時確認勾選「Use WSL 2 instead of Hyper-V」
   - 安裝完成後重新開機

2. **啟用 WSL2 GPU 支援**
   ```powershell
   # 在 PowerShell (管理員) 中執行
   wsl --update
   ```

3. **安裝 NVIDIA Container Toolkit**
   ```bash
   # 在 WSL2 Ubuntu 終端中執行
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   # 重啟 Docker Desktop
   ```

4. **驗證 GPU 直通**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   # 應顯示 RTX 4060 的資訊
   ```

### 建置步驟

```bash
# 1. 複製環境變數模板
cp .env.example .env

# 2. 建置 Docker 映像 (首次約需 10-15 分鐘)
make build

# 3. 啟動 Ollama 並下載 LLM 模型 (約需 5-10 分鐘)
make setup

# 4. 驗證 Ollama 運作正常
curl http://localhost:11434/api/tags
```

---

## 執行流程

### 方式一：逐步手動執行 (建議首次使用)

```bash
# Step 1: 合成資料生成 (Ollama 佔用 GPU)
make data-synthetic

# Step 2: 停止 Ollama 釋放 VRAM (必要步驟)
make stop-ollama

# Step 3: Hard Negative Mining
make data-mining

# Step 4: 資料後處理
make data-process

# Step 5: LoRA 微調 (small + base)
make train

# Step 6: 四象限評估 + 產出圖表
make eval
```

### 方式二：一鍵全流程

```bash
make pipeline
```

### 查看所有可用指令

```bash
make help
```

---

## 專案結構

```
├── configs/           # 集中設定 (超參數、路徑、VRAM 預算)
├── data_pipeline/     # 資料生成 + 難負例挖掘 + VRAM 守衛
├── training/          # LoRA 微調 (peft + fp16 + 梯度累積)
├── retrieval/         # Dense (GPU) + BM25/jieba (CPU) + RRF
├── evaluation/        # MRR@10, NDCG@10, 長條圖, UMAP
├── scripts/           # 可執行入口腳本
├── data/              # 資料儲存 (不進 Git)
├── models/            # 模型權重 (不進 Git)
└── outputs/           # 評估結果 (不進 Git)
```

---

## 關鍵設計決策

### VRAM 分時調度

8GB VRAM 無法同時容納 LLM + Embedding 模型 + 訓練。本專案透過 `VRAMGuard` 機制，強制各階段序列化執行，在階段切換時執行三步釋放協議：

```
del model → gc.collect() → torch.cuda.empty_cache()
```

### CPU/GPU 解耦

- BM25 + jieba 斷詞：純 CPU 執行，`bm25_retriever.py` 不 import torch
- Dense Embedding：GPU 執行
- RRF 融合：CPU 執行

### LoRA 設定

- 嚴禁全參數微調
- `r=8`, `lora_alpha=16`, `target_modules=["query", "value"]`
- 僅保存 adapter 權重 (~2-10 MB)，推論時動態掛載

---

## 已知限制

1. 真實 batch size 4-8 導致 in-batch negatives 數量有限，對比學習效果低於大 VRAM 環境
2. jieba 預設辭典為簡體中文，繁體中文需搭配 opencc 轉換或載入自定義辭典
3. Ollama 建議使用 7B 量化模型，13B 在 8GB VRAM 上有 OOM 風險
4. UMAP 視覺化需對 embedding 取樣 (建議 ≤ 10,000 點) 以避免 CPU RAM 不足
