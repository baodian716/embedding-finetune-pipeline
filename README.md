# embedding-finetune-pipeline

LoRA fine-tuning for BGE embedding models with hybrid retrieval evaluation on RTX 4060 Ti 8GB

---

## 專案概述

本專案實作一套完整的領域自適應檢索管線，透過 LoRA 微調讓 BGE embedding 模型學習特定領域的語意，並以量化指標驗證微調成效。

**核心展示能力：**
- **LoRA 微調** — 在 `bge-small-zh-v1.5` 與 `bge-base-zh-v1.5` 上掛載 LoRA adapter，僅更新 ~2% 參數
- **VRAM 分時調度** — 單張 8GB 顯卡依序執行 LLM 合成資料 → Hard Negative Mining → 訓練 → 評估
- **混合檢索** — Dense (GPU) + BM25/jieba (CPU) 雙軌，RRF 融合排序
- **消融實驗** — 四組對照（Small/Base × Baseline/LoRA）+ Hybrid，量化各元件貢獻

---

## 評估結果

語料：1,170 段落 ｜ 查詢：94 條 ｜ 模型：BGE zh-v1.5

| 條件 | MRR@10 | NDCG@10 | Hit Rate@10 |
|------|-------:|--------:|------------:|
| small — baseline | 0.107 | 0.124 | 0.181 |
| small — LoRA | 0.154 | 0.173 | 0.234 |
| base — baseline | 0.228 | 0.245 | 0.298 |
| **base — LoRA** | **0.333** | **0.351** | **0.404** |
| hybrid (base LoRA + BM25) | 0.325 | 0.354 | **0.447** |

**LoRA 微調帶來的提升：**
- bge-small：MRR@10 +44%（0.107 → 0.154）
- bge-base：MRR@10 +46%（0.228 → 0.333）

> **MRR@10**：第一個正確結果的排名倒數平均值，越高代表正確答案排越前面。
> **Hit Rate@10**：正確結果出現在前 10 名的比例。

---

## 系統需求

| 項目 | 需求 |
|------|------|
| GPU | NVIDIA RTX 4060 Ti 8GB（或同等 VRAM） |
| RAM | 16 GB |
| 磁碟 | 20 GB 可用空間（含 Docker 映像 + 模型） |
| 作業系統 | Windows 11 (WSL2) / Ubuntu 22.04+ |
| NVIDIA 驅動 | >= 528.33 (Windows) / >= 525.60 (Linux) |
| Docker Desktop | >= 4.20 |

---

## 環境建置

### 前置條件

1. **安裝 Docker Desktop**
   - 安裝時確認勾選「Use WSL 2 instead of Hyper-V」
   - 安裝完成後重新開機

2. **啟用 WSL2 GPU 支援**
   ```powershell
   # 在 PowerShell (管理員) 中執行
   wsl --update
   ```

3. **安裝 NVIDIA Container Toolkit**（WSL2 Ubuntu 終端）
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

4. **驗證 GPU 直通**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

### 建置步驟

```bash
# 建置 Docker 映像（首次約需 10-15 分鐘）
make build

# 啟動 Ollama 並下載 LLM 模型（llama3.2 3B，約需 5 分鐘）
make setup
```

---

## 執行流程

### 準備語料

將 PDF 文件放置於專案根目錄，執行轉換腳本（已提供 80 段範例語料於 `data/sample/`）：

```bash
python pdf_to_corpus.py --min-length 80
# 輸出至 data/raw/corpus.txt
```

### 方式一：一鍵全流程

```bash
make pipeline
```

### 方式二：逐步手動執行

```bash
make data-synthetic   # 階段 A：LLM 合成 query-passage 訓練對
make stop-ollama      # 階段 A→B：停止 Ollama，釋放 VRAM
make data-mining      # 階段 B：Hard Negative Mining
make data-process     # 資料後處理（清洗 + train/val 切分）
make train            # 階段 C：LoRA 微調（small + base）
make eval             # 階段 D+E：評估 + 產出圖表
```

```bash
make help             # 查看所有可用指令
```

---

## 專案結構

```
embedding-finetune-pipeline/
├── configs/                    # 集中設定（超參數、路徑、VRAM 預算）
│   ├── base_config.py          # 路徑、裝置、Ollama 設定
│   ├── vram_config.py          # 各階段 VRAM 預算（針對 8GB 校準）
│   ├── model_config.py         # BGE small / base 模型參數
│   ├── training_config.py      # LoRA 超參數、batch size、學習率
│   └── retrieval_config.py     # BM25 / Dense / RRF 參數
├── data_pipeline/              # 資料生成 + 難負例挖掘 + VRAM 守衛
├── training/                   # LoRA 微調（peft + fp16 + 梯度累積）
├── retrieval/                  # Dense (GPU) + BM25/jieba (CPU) + RRF
├── evaluation/                 # MRR@10、NDCG@10、長條圖、UMAP
├── scripts/                    # 可執行入口腳本
├── pdf_to_corpus.py            # PDF → corpus.txt 雜訊清理腳本
├── data/
│   └── sample/
│       └── corpus_sample.txt   # 80 段範例語料（可直接用於測試）
├── Makefile                    # 任務執行器
├── docker-compose.yml          # 容器編排（app + ollama 服務）
└── requirements.txt
```

不進版控的目錄（`.gitignore` 排除）：

```
data/raw/       # 完整語料（來自 PDF，含版權）
data/synthetic/ # LLM 合成資料（訓練產物）
models/         # HuggingFace 基底模型 + LoRA adapter 權重
outputs/        # 評估圖表、報告
```

---

## 關鍵設計決策

### VRAM 分時調度（VRAMGuard）

8GB VRAM 無法同時容納 LLM + Embedding 模型 + 訓練。各階段序列化執行，切換時強制三步釋放：

```python
del model
gc.collect()
torch.cuda.empty_cache()
```

Windows WDDM 模式下，桌面 UI 固定佔用 ~800–1100 MB，`post_release_max_mb` 設為 1500 MB（Linux 無頭環境可改回 200 MB）。

### LoRA 設定

```python
r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1
```

只掛載 Attention 的 query / value 投影層，adapter 體積約 2–10 MB，推論時動態載入。

### 混合檢索（RRF 融合）

```
score(d) = Σ 1 / (k + rank_i(d)),  k=60
```

Dense 排名 + BM25 排名透過 RRF 合併，無需調整權重參數，Hit Rate@10 較純 Dense 提升約 10%。

### Ollama LLM 選型

使用 `llama3.2`（3B，q4_K_M 量化，約佔 2.0 GB VRAM），在 8GB 環境下為合成資料生成保留足夠餘量。

---

## 已知限制

1. 真實 batch size 4–8 導致 in-batch negatives 數量有限，對比學習效果低於大 VRAM 環境（已透過 hard negatives 部分補償）
2. jieba 預設辭典為簡體中文，繁體中文場景建議搭配 opencc 轉換或載入自定義辭典
3. UMAP 視覺化對 embedding 取樣（建議 ≤ 10,000 點）以避免 CPU RAM 不足
4. 語料規模較小（1,170 段落），絕對指標偏低；LoRA 的**相對提升幅度**（+44–46%）是主要觀察指標
