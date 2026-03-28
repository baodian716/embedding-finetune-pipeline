# ============================================================
# data_pipeline/vram_guard.py — VRAM 時間分時守衛
#
# 核心設計理念:
#   8GB VRAM 無法同時容納 LLM + Embedding 模型 + 訓練器。
#   本模組透過「三步釋放協議」強制各階段序列化，
#   在階段切換時確保 GPU 記憶體完全歸還給驅動。
#
# 三步釋放協議 (缺一不可，順序不可顛倒):
#   Step 1: 呼叫端執行 del model  → 解除 Python 物件參照
#   Step 2: gc.collect()          → 強制觸發 Python GC，處理循環參照
#   Step 3: torch.cuda.empty_cache() → 將 PyTorch 快取池歸還給 CUDA 驅動
#
# 重要限制:
#   empty_cache() 只能釋放「沒有 Python 參照指向」的 CUDA 記憶體。
#   若呼叫端仍持有模型/張量的參照，則釋放無效。
#   因此 del 必須在 empty_cache() 之前完成，且不能有多餘的參照存留。
# ============================================================

import gc
import time
from contextlib import contextmanager
from typing import Optional

from loguru import logger

# 延遲匯入: 若在純 CPU 環境執行也不會崩潰
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


class VRAMGuard:
    """
    VRAM 時間分時守衛。

    提供三類介面:
    1. release_vram()           — 執行三步釋放協議 (呼叫前必須在外部 del 模型)
    2. assert_safe_to_proceed() — 斷言 VRAM 殘餘量低於安全閾值
    3. phase()                  — Context manager，進入時記錄狀態，離開時自動清理

    典型使用模式:
        guard = VRAMGuard(post_release_max_mb=200)

        # 階段 B: Hard Negative Mining
        with guard.phase("mining", budget_mb=3000):
            model = SentenceTransformer(...)
            embeddings = model.encode(corpus)
            del model          # ← 在 with 塊內部顯式 del
            guard.release_vram()  # ← 然後立即執行三步協議

        # with 塊結束後會再執行一次 release_vram() 作為保險
        guard.assert_safe_to_proceed("training")
    """

    def __init__(self, post_release_max_mb: int = 200, device_index: int = 0):
        """
        Args:
            post_release_max_mb: 釋放後允許的最大殘餘 VRAM (MB)。
                                  RTX 4060 Ti 建議值: 200 MB (CUDA context 本身約佔 100-150 MB)
            device_index: GPU 設備索引。單卡環境固定為 0
        """
        self.post_release_max_mb = post_release_max_mb
        self.device_index = device_index
        self._nvml_handle = None

        # 優先嘗試初始化 pynvml，它可以看到包含 Ollama 在內的所有程序的 VRAM 佔用
        # torch.cuda.memory_allocated() 只能看到 PyTorch 自己分配的部分，
        # Ollama 佔用的 VRAM 對 torch 是不可見的
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                logger.debug(f"VRAMGuard: pynvml 初始化成功，監控 GPU {device_index}")
            except Exception as e:
                logger.warning(f"VRAMGuard: pynvml 初始化失敗 ({e})，退回使用 torch 查詢")

        if not PYNVML_AVAILABLE and not TORCH_AVAILABLE:
            logger.warning("VRAMGuard: 無可用的 VRAM 監控工具，將在無監控模式下運行")

    # ------------------------------------------------------------------
    # VRAM 查詢
    # ------------------------------------------------------------------

    def get_used_vram_mb(self) -> Optional[float]:
        """
        查詢當前 GPU 已使用的 VRAM (MB)。

        查詢策略:
        - 優先: pynvml → 可見所有程序 (包含 Ollama)，最精確
        - 退而: torch.cuda.memory_reserved() → 只含 PyTorch 快取池
          (注意: 用 reserved 而非 allocated，前者才是 GPU 層面真正佔用的量)

        Returns:
            float: 已使用的 MB，None 表示無法取得
        """
        if PYNVML_AVAILABLE and self._nvml_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return mem_info.used / (1024 ** 2)
            except Exception as e:
                logger.debug(f"pynvml 查詢失敗: {e}")

        if TORCH_AVAILABLE and torch.cuda.is_available():
            # memory_reserved 包含 PyTorch 快取但未使用的 VRAM，
            # 這部分雖未存放張量，但 CUDA 已將其配置給 PyTorch，外部看來是「已用」
            return torch.cuda.memory_reserved(self.device_index) / (1024 ** 2)

        return None

    def get_free_vram_mb(self) -> Optional[float]:
        """查詢當前 GPU 可用的 VRAM (MB)。"""
        if PYNVML_AVAILABLE and self._nvml_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return mem_info.free / (1024 ** 2)
            except Exception:
                pass

        if TORCH_AVAILABLE and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(self.device_index)
            return free / (1024 ** 2)

        return None

    def log_vram_status(self, label: str = "") -> None:
        """將當前 VRAM 狀態輸出到日誌，供 debug 使用。"""
        used = self.get_used_vram_mb()
        free = self.get_free_vram_mb()
        parts = ["VRAM"]
        if label:
            parts.append(f"[{label}]")
        if used is not None:
            parts.append(f"已用={used:.0f}MB")
        if free is not None:
            parts.append(f"可用={free:.0f}MB")
        if used is None and free is None:
            parts.append("(無法查詢)")
        logger.info(" ".join(parts))

    # ------------------------------------------------------------------
    # 核心: 三步釋放協議
    # ------------------------------------------------------------------

    def release_vram(self) -> None:
        """
        執行三步 VRAM 釋放協議。

        ★ 呼叫前的必要步驟 (由呼叫端負責):
            del model          # 解除對模型的 Python 參照
            # 確認沒有其他變數指向同一模型物件

        本方法只負責 Step 2 + Step 3:
            Step 2: gc.collect()              — 清理循環參照
            Step 3: torch.cuda.empty_cache()  — 清空 CUDA 快取池

        之所以不在此方法內執行 Step 1，是因為 Python 函數無法刪除呼叫端的
        區域變數 (傳入的物件只是多了一個臨時參照，函數結束後才消除此臨時參照，
        但呼叫端的原始參照仍然存在)。
        """
        # Step 2: 強制觸發 Python 垃圾回收
        # 特別針對循環參照 (PyTorch 模型層之間常有循環參照，GC 才能處理)
        collected = gc.collect()
        logger.debug(f"gc.collect() 清理了 {collected} 個物件")

        # Step 3: 清空 PyTorch 管理的 CUDA 快取池
        # 作用: 將 PyTorch 「已標記為可回收但尚未歸還 CUDA 驅動」的記憶體區塊歸還
        # 效果: 讓 nvidia-smi 看到的已用 VRAM 下降
        # 限制: 若張量仍有 Python 參照指向，此操作對其無效
        if TORCH_AVAILABLE and torch.cuda.is_available():
            before_mb = torch.cuda.memory_reserved(self.device_index) / (1024 ** 2)
            torch.cuda.empty_cache()
            after_mb = torch.cuda.memory_reserved(self.device_index) / (1024 ** 2)
            logger.debug(
                f"torch.cuda.empty_cache(): "
                f"PyTorch 快取 {before_mb:.0f} → {after_mb:.0f} MB"
            )

        # 短暫等待，讓 CUDA 驅動完成非同步記憶體回收
        # (CUDA 某些操作是非同步的，立即查詢可能尚未反映)
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # 安全斷言
    # ------------------------------------------------------------------

    def assert_safe_to_proceed(self, phase_name: str = "") -> None:
        """
        斷言當前 VRAM 使用量低於 post_release_max_mb，否則拋出 RuntimeError。

        在每個 GPU 密集階段開始前呼叫，確認上個階段已正確清空 VRAM。
        這是防止「前一個階段的模型殘留在 GPU 上」的最後一道防線。

        Args:
            phase_name: 即將開始的階段名稱 (用於錯誤訊息)

        Raises:
            RuntimeError: 殘餘 VRAM 超過閾值，需人工介入
        """
        used_mb = self.get_used_vram_mb()

        if used_mb is None:
            logger.warning(f"[{phase_name}] 無法查詢 VRAM，跳過安全斷言 (請手動確認 nvidia-smi)")
            return

        hint = f"[{phase_name}] " if phase_name else ""
        logger.info(
            f"{hint}VRAM 安全斷言: 當前 {used_mb:.0f} MB ≤ 閾值 {self.post_release_max_mb} MB"
            if used_mb <= self.post_release_max_mb
            else f"{hint}VRAM 安全斷言: 當前 {used_mb:.0f} MB > 閾值 {self.post_release_max_mb} MB ← 失敗"
        )

        if used_mb > self.post_release_max_mb:
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"VRAM 安全斷言失敗！準備進入階段: {phase_name}\n"
                f"當前 VRAM 使用量: {used_mb:.0f} MB\n"
                f"允許上限: {self.post_release_max_mb} MB\n"
                f"\n可能原因:\n"
                f"  1. 上一個階段的模型未正確 del 後執行 gc.collect()\n"
                f"  2. Ollama 仍在 GPU 上運行 (請先執行 make stop-ollama)\n"
                f"  3. 有其他程序佔用 GPU (執行 nvidia-smi 確認)\n"
                f"\n解決方式:\n"
                f"  - 重新啟動此 Python 程序\n"
                f"  - 或執行 make stop-ollama 後再重試\n"
                f"{'='*60}"
            )

    # ------------------------------------------------------------------
    # Context Manager 介面
    # ------------------------------------------------------------------

    @contextmanager
    def phase(self, name: str, budget_mb: Optional[int] = None):
        """
        階段上下文管理器。

        進入時: 記錄 VRAM 基準狀態
        離開時: 自動執行 release_vram()，即使階段內拋出異常也會執行

        Args:
            name: 階段名稱，用於日誌識別
            budget_mb: 此階段的 VRAM 預算 (MB)，只用於日誌記錄，不強制執行

        使用範例:
            with guard.phase("mining", budget_mb=3000) as g:
                model = SentenceTransformer(...)
                embeddings = model.encode(corpus)
                del model           # ← 顯式刪除，必須在 with 內部執行
                g.release_vram()    # ← 立即觸發三步協議，儘早釋放 VRAM

            # 離開 with 時還會再執行一次 release_vram() 作為保險
        """
        separator = "=" * 55
        budget_info = f" (預算上限: {budget_mb} MB)" if budget_mb else ""
        logger.info(f"\n{separator}")
        logger.info(f"▶ 進入階段: {name}{budget_info}")
        self.log_vram_status("進入前")

        try:
            yield self
        except Exception as e:
            logger.error(f"✗ 階段 [{name}] 發生例外: {type(e).__name__}: {e}")
            raise
        finally:
            # finally 保證: 無論正常結束或發生異常，都執行清理
            logger.info(f"◀ 離開階段: {name}，執行 VRAM 清理...")
            self.release_vram()
            self.log_vram_status("清理後")
            logger.info(separator)

    # ------------------------------------------------------------------
    # 靜態工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def force_release(model=None, optimizer=None, extra_tensors: Optional[list] = None) -> None:
        """
        靜態便利方法: 在不需要 VRAMGuard 實例時快速釋放資源。

        這是一個「緊急清理」方法，設計用途是在腳本頂層的 except 區塊中使用。
        正常流程應透過 release_vram() 搭配手動 del 操作。

        Args:
            model: 要刪除的模型 (若已在外部 del，傳 None)
            optimizer: 要刪除的 optimizer
            extra_tensors: 其他要清理的張量列表
        """
        # 刪除傳入的物件 (只能刪除此函數內的臨時參照，呼叫端仍需自行 del)
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if extra_tensors:
            for t in extra_tensors:
                del t

        gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
