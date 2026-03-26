# ============================================================
# Domain-Adaptive Retrieval System — Makefile
# 將常用的多步驟 Docker 指令封裝為單一 make 目標
# 使用方式: make <目標名稱>，例如 make train
# ============================================================

# 預設 Shell
SHELL := /bin/bash

# Docker Compose 指令前綴
DC := docker compose

# 容器內 Python 執行指令
RUN_APP := $(DC) run --rm app python

# ============================================================
# 環境建置
# ============================================================

.PHONY: build
build: ## 建置 Docker 映像 (首次使用或修改 Dockerfile/requirements.txt 後執行)
	$(DC) build

.PHONY: setup
setup: build ## 完整初始化: 建置映像 + 啟動 Ollama + 下載 LLM 模型
	$(DC) up -d ollama
	@echo "等待 Ollama 啟動..."
	@timeout /t 5 /nobreak > NUL
	docker exec ollama-server ollama pull $(OLLAMA_MODEL)
	@echo "環境初始化完成。"

# 預設 Ollama 模型 (可透過環境變數覆蓋)
# llama3.2 (3B) 在 q4_K_M 量化下約佔 2.0 GB VRAM，
# 相較 qwen2.5:7b (~4.5 GB) 可多留 2.5 GB 給後續 Mining / Training 階段
OLLAMA_MODEL ?= llama3.2

# ============================================================
# VRAM 分時調度流程 (依序執行)
# ============================================================

.PHONY: data-synthetic
data-synthetic: ## 階段 A: 啟動 Ollama + 執行 LLM 合成資料
	$(DC) up -d ollama
	@echo "等待 Ollama 就緒..."
	@timeout /t 5 /nobreak > NUL
	$(RUN_APP) scripts/run_data_pipeline.py --phase synthetic

.PHONY: stop-ollama
stop-ollama: ## 階段 A→B 過渡: 停止 Ollama 釋放 VRAM
	$(DC) stop ollama
	@echo "Ollama 已停止，等待 GPU 驅動回收 VRAM..."
	@timeout /t 5 /nobreak > NUL

.PHONY: data-mining
data-mining: ## 階段 B: 執行 Hard Negative Mining
	$(RUN_APP) scripts/run_data_pipeline.py --phase mining

.PHONY: data-process
data-process: ## 資料後處理: 清洗 + 格式轉換 + train/val 切分
	$(RUN_APP) scripts/run_data_pipeline.py --phase process

.PHONY: train
train: train-small train-base ## 階段 C: 依序執行 LoRA 微調 (bge-small → bge-base 兩輪)

.PHONY: train-small
train-small: ## 階段 C: 僅微調 bge-small-zh
	$(RUN_APP) scripts/run_training.py --model small

.PHONY: train-base
train-base: ## 階段 C: 僅微調 bge-base-zh
	$(RUN_APP) scripts/run_training.py --model base

# 預設使用 small 模型，可透過 MODEL=base 覆蓋，例如: make retrieve MODEL=base
MODEL ?= small

.PHONY: retrieve
retrieve: ## 階段 D: 執行混合檢索示範 (MODEL=small|base，預設 small)
	$(RUN_APP) scripts/run_retrieval.py --model $(MODEL)

.PHONY: retrieve-base
retrieve-base: ## 階段 D: 使用 bge-base 模型執行混合檢索示範
	$(RUN_APP) scripts/run_retrieval.py --model base

.PHONY: eval
eval: ## 階段 E: 執行四象限評估 + 產出圖表
	$(RUN_APP) scripts/run_evaluation.py

# ============================================================
# 全流程
# ============================================================

.PHONY: pipeline
pipeline: data-synthetic stop-ollama data-mining data-process train eval ## 一鍵全流程 (約需數小時)
	@echo "全流程執行完畢。結果位於 outputs/ 目錄。"

# ============================================================
# 工具指令
# ============================================================

.PHONY: shell
shell: ## 進入容器內的互動式 Shell (除錯用)
	$(DC) run --rm app bash

.PHONY: logs-ollama
logs-ollama: ## 查看 Ollama 服務日誌
	$(DC) logs -f ollama

.PHONY: gpu-status
gpu-status: ## 查看 GPU 使用狀態 (透過容器內 nvidia-smi)
	$(DC) run --rm app nvidia-smi

.PHONY: clean
clean: ## 停止所有容器並移除 Docker 網路
	$(DC) down

.PHONY: clean-all
clean-all: clean ## 停止容器 + 清除所有輸出結果 (不刪除資料與模型)
	rm -rf outputs/charts/* outputs/embeddings/* outputs/reports/*
	@echo "輸出目錄已清空 (保留 .gitkeep)。"

.PHONY: test
test: ## 執行單元測試
	$(RUN_APP) -m pytest tests/ -v

# ============================================================
# 幫助
# ============================================================

.PHONY: help
help: ## 顯示所有可用目標
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 預設目標
.DEFAULT_GOAL := help