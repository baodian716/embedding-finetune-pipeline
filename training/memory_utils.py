# ============================================================
# training/memory_utils.py — 訓練階段記憶體監控工具
#
# 與 data_pipeline/vram_guard.py 的分工:
#   vram_guard.py — 「階段邊界」的 VRAM 釋放與安全斷言
#   memory_utils.py — 「訓練迴圈內部」的動態監控與 OOM 緊急應變
#
# RTX 4060 訓練階段的 VRAM 使用分析:
#
#   bge-small-zh-v1.5 (batch=8, seq=256, fp16):
#     模型參數 (fp16):   ~48  MB
#     LoRA 參數 (fp16):  ~2   MB
#     優化器狀態 (fp32): ~8   MB  ← LoRA 參數的 AdamW first+second moment
#     啟動值快取:        ~200 MB  ← 8 × 256 × 512 × 6層 × 2 (fp16)
#     梯度:              ~50  MB
#     總計估算:          ~308 MB  ← 遠低於 2000 MB 預算，安全
#
#   bge-base-zh-v1.5 (batch=4, seq=256, fp16):
#     模型參數 (fp16):   ~204 MB
#     LoRA 參數 (fp16):  ~8   MB
#     優化器狀態 (fp32): ~32  MB
#     啟動值快取:        ~200 MB  ← 4 × 256 × 768 × 12層 × 2
#     梯度:              ~200 MB
#     總計估算:          ~644 MB  ← 遠低於 3500 MB 預算，安全
#
# 注意: 以上估算為保守上限，實際使用量通常更低 (fp16 可節省 50%)。
#       真正的 VRAM 瓶頸是三路編碼 (query + positive + negative) 同時存在梯度圖。
# ============================================================

import time
from typing import Dict, Optional

from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# ============================================================
# VRAM 查詢工具函數
# ============================================================

def get_vram_mb(device_index: int = 0) -> Dict[str, float]:
    """
    查詢當前 GPU 的 VRAM 使用情況。

    同時查詢 pynvml (全局視角) 和 torch (PyTorch 視角)，
    兩者差異可以幫助診斷 VRAM 問題。

    Returns:
        dict with keys: 'used_total', 'free_total', 'torch_allocated', 'torch_reserved'
        值皆為 MB。若無法查詢則對應欄位為 None。
    """
    result = {}

    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result["used_total"] = mem.used / (1024 ** 2)
            result["free_total"] = mem.free / (1024 ** 2)
            result["total"]      = mem.total / (1024 ** 2)
        except Exception:
            pass

    if TORCH_AVAILABLE and torch.cuda.is_available():
        result["torch_allocated"] = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
        result["torch_reserved"]  = torch.cuda.memory_reserved(device_index)  / (1024 ** 2)

        if "total" not in result:
            _, total = torch.cuda.mem_get_info(device_index)
            free, _ = torch.cuda.mem_get_info(device_index)
            result["total"]      = total / (1024 ** 2)
            result["free_total"] = free  / (1024 ** 2)
            result["used_total"] = result["total"] - result["free_total"]

    return result


def log_vram(label: str = "", device_index: int = 0) -> None:
    """
    將當前 VRAM 使用量輸出到日誌，標準化訓練迴圈內的 VRAM 監控格式。

    使用範例:
        log_vram("epoch 1 開始")
        log_vram(f"step {step}")
    """
    stats = get_vram_mb(device_index)
    if not stats:
        return

    parts = []
    if "used_total" in stats:
        parts.append(f"已用={stats['used_total']:.0f}MB")
    if "free_total" in stats:
        parts.append(f"可用={stats['free_total']:.0f}MB")
    if "torch_allocated" in stats:
        parts.append(f"torch_alloc={stats['torch_allocated']:.0f}MB")

    label_str = f"[{label}] " if label else ""
    logger.debug(f"VRAM {label_str}| {' | '.join(parts)}")


def is_vram_near_limit(
    threshold_ratio: float = 0.85,
    device_index: int = 0,
) -> bool:
    """
    檢查 VRAM 使用率是否超過閾值 (超過則回傳 True)。

    在訓練迴圈中定期呼叫，在 OOM 發生前提前預警。
    若回傳 True，應考慮降低 batch size 或啟用梯度 checkpointing。

    Args:
        threshold_ratio: 警戒水位 (預設 0.85 = 85%)
    """
    stats = get_vram_mb(device_index)
    if "used_total" not in stats or "total" not in stats:
        return False  # 無法查詢，不預警

    usage_ratio = stats["used_total"] / stats["total"]
    return usage_ratio > threshold_ratio


# ============================================================
# 訓練過程的記憶體監控器
# ============================================================

class TrainingMemoryMonitor:
    """
    訓練迴圈中的 VRAM 使用量監控器。

    功能:
    1. 定期記錄 VRAM 使用量 (每 N 步)
    2. 追蹤訓練過程中的 VRAM 峰值
    3. 在 VRAM 超過警戒水位時發出警告

    使用範例:
        monitor = TrainingMemoryMonitor(monitor_interval=10, threshold_ratio=0.85)
        for step, batch in enumerate(train_loader):
            ...
            monitor.step(step)
    """

    def __init__(
        self,
        monitor_interval: int = 10,
        threshold_ratio: float = 0.85,
        device_index: int = 0,
    ):
        """
        Args:
            monitor_interval: 每隔幾個 gradient update step 查詢一次 VRAM
            threshold_ratio: VRAM 使用率警戒水位
            device_index: GPU 設備索引
        """
        self.monitor_interval = monitor_interval
        self.threshold_ratio = threshold_ratio
        self.device_index = device_index
        self.peak_vram_mb: float = 0.0
        self._warned_near_limit: bool = False

    def step(self, global_step: int, extra_label: str = "") -> None:
        """
        在每個 gradient update step 後呼叫。

        Args:
            global_step: 當前全局步數 (gradient update 計數，非 forward pass 計數)
            extra_label: 附加在日誌的額外標籤
        """
        if global_step % self.monitor_interval != 0:
            return

        stats = get_vram_mb(self.device_index)
        if not stats or "used_total" not in stats:
            return

        used_mb = stats["used_total"]
        total_mb = stats.get("total", 8192)

        # 更新峰值記錄
        if used_mb > self.peak_vram_mb:
            self.peak_vram_mb = used_mb

        usage_ratio = used_mb / total_mb if total_mb > 0 else 0
        label = f"step={global_step}" + (f" | {extra_label}" if extra_label else "")

        logger.debug(
            f"[VRAM Monitor] {label} | "
            f"used={used_mb:.0f}MB | "
            f"ratio={usage_ratio:.1%} | "
            f"peak={self.peak_vram_mb:.0f}MB"
        )

        # 警告: VRAM 超過警戒水位
        if usage_ratio > self.threshold_ratio and not self._warned_near_limit:
            logger.warning(
                f"\n{'!'*55}\n"
                f"[VRAM 警告] 使用率 {usage_ratio:.1%} 超過警戒水位 "
                f"{self.threshold_ratio:.1%}\n"
                f"  當前使用: {used_mb:.0f} MB / {total_mb:.0f} MB\n"
                f"  建議措施:\n"
                f"    1. 降低 batch_size (修改 configs/training_config.py)\n"
                f"    2. 增加 gradient_accumulation_steps (維持有效 batch size)\n"
                f"    3. 縮短 max_seq_length (從 256 → 128，但會損失語意)\n"
                f"{'!'*55}\n"
            )
            self._warned_near_limit = True

    def report_peak(self) -> None:
        """訓練結束後輸出 VRAM 峰值報告。"""
        logger.info(f"[VRAM Monitor] 訓練期間 VRAM 峰值: {self.peak_vram_mb:.0f} MB")


# ============================================================
# OOM 處理輔助
# ============================================================

def handle_oom_error(
    current_batch_size: int,
    min_batch_size: int = 2,
) -> Optional[int]:
    """
    發生 CUDA OOM 時的應急處理。

    嘗試清空快取並建議降低 batch size。
    回傳建議的新 batch size，若已達最小值則回傳 None (應終止訓練)。

    ★ 設計說明:
    本函數的設計初衷是「提供明確的除錯資訊」，而非「靜默地自動恢復」。
    自動降低 batch size 後繼續訓練在工程上是可行的，
    但會使最終的 ablation 結果難以解讀 (兩個模型的有效 batch size 不同)。
    因此，OOM 發生時我們選擇「清理後終止」，讓使用者調整設定後重新執行。

    Args:
        current_batch_size: 發生 OOM 時的 batch size
        min_batch_size: batch size 的最小允許值

    Returns:
        建議的新 batch size，或 None (若已無法降低)
    """
    import gc

    logger.error(
        f"\n{'='*60}\n"
        f"CUDA OOM 錯誤! 當前 batch_size={current_batch_size}\n"
        f"正在執行緊急記憶體清理...\n"
        f"{'='*60}"
    )

    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    log_vram("OOM 緊急清理後")

    suggested = current_batch_size // 2
    if suggested < min_batch_size:
        logger.error(
            f"建議的 batch_size ({suggested}) 低於最小允許值 ({min_batch_size})。\n"
            f"請考慮:\n"
            f"  1. 縮短 max_seq_length (256 → 128)\n"
            f"  2. 關閉其他佔用 GPU 的程序\n"
            f"  3. 確認 Ollama 服務已停止 (make stop-ollama)"
        )
        return None

    logger.warning(
        f"建議將 batch_size 從 {current_batch_size} 調整為 {suggested}\n"
        f"請修改 configs/training_config.py 後重新執行訓練"
    )
    return suggested


def count_trainable_parameters(model) -> Dict[str, int]:
    """
    計算模型的可訓練 (LoRA) 與凍結 (Base Model) 參數數量。

    LoRA 正確應用後，可訓練參數佔比應在 0.1% ~ 1% 之間。
    若比例過高，表示 LoRA 配置有誤 (可能不小心解凍了 base model 參數)。

    Args:
        model: peft 包裝後的模型

    Returns:
        dict with 'trainable', 'frozen', 'total', 'trainable_ratio'
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    ratio = trainable / total if total > 0 else 0.0

    logger.info(
        f"\n{'='*50}\n"
        f"模型參數統計:\n"
        f"  可訓練 (LoRA): {trainable:>12,}  ({ratio:.4%})\n"
        f"  凍結 (Base):   {frozen:>12,}  ({1-ratio:.4%})\n"
        f"  總計:          {total:>12,}\n"
        f"{'='*50}"
    )

    if ratio > 0.05:  # 可訓練比例超過 5% 可能是異常
        logger.warning(
            f"可訓練參數比例 ({ratio:.2%}) 異常偏高！\n"
            f"LoRA 正確配置下應為 0.1% ~ 1%。\n"
            f"請確認 peft LoraConfig 的 target_modules 設定正確，\n"
            f"且 Base Model 參數已正確凍結。"
        )

    return {
        "trainable":       trainable,
        "frozen":          frozen,
        "total":           total,
        "trainable_ratio": ratio,
    }
