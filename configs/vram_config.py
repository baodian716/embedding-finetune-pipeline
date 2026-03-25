# ============================================================
# configs/vram_config.py — VRAM 預算與安全閾值設定
# ★ 本檔案是整個專案防 OOM 策略的參數化核心
# 若更換 GPU (例如升級為 12GB)，只需修改此檔案
# ============================================================

from dataclasses import dataclass


@dataclass
class VRAMConfig:
    """
    VRAM 預算管理設定。
    所有數值單位為 MB，基於 RTX 4060 8GB (實際可用 ~7.6 GB) 校準。
    """

    # ---- GPU 硬體規格 ----
    # RTX 4060 標稱 8192 MB，但系統/驅動會保留一部分
    total_vram_mb: int = 8192
    # 安全餘量: 預留給作業系統 GUI、CUDA context 等
    # 8GB 環境建議預留 1024 MB (1GB)
    safety_margin_mb: int = 1024
    # 實際可用 VRAM = total - safety_margin
    # RTX 4060: 8192 - 1024 = 7168 MB

    # ---- 各階段 VRAM 預算上限 ----
    # 階段 A: Ollama LLM 合成資料
    # Ollama 自行管理 VRAM，此處設定的是「Python 端允許使用的上限」
    # 由於 Python 在此階段僅做 HTTP 呼叫，應為 0
    phase_a_budget_mb: int = 0

    # 階段 B: Hard Negative Mining (embedding 模型 + 相似度運算)
    phase_b_budget_mb: int = 3000

    # 階段 C: LoRA 訓練
    # small 模型: 模型 48MB + optimizer 4MB + activations ~800MB + 緩衝 ~500MB
    phase_c_small_budget_mb: int = 2000
    # base 模型: 模型 204MB + optimizer 8MB + activations ~1.5GB + 緩衝 ~500MB
    phase_c_base_budget_mb: int = 3500

    # 階段 D: 檢索推論 (單模型載入 + 編碼)
    phase_d_budget_mb: int = 2000

    # 階段 E: 評估 + UMAP (embedding 生成)
    phase_e_budget_mb: int = 2000

    # ---- VRAM 釋放後的驗證閾值 ----
    # 階段切換時，執行 empty_cache() 後殘餘 VRAM 不應超過此值
    # 若超過，表示有模型/張量未正確釋放，應拋出錯誤
    post_release_max_mb: int = 200

    # ---- 監控頻率 ----
    # 訓練迴圈中每 N 步檢查一次 VRAM 使用量
    monitor_interval_steps: int = 10

    # ---- 動態 batch size 回退 ----
    # 若偵測到 VRAM 使用率超過此比例，自動將 batch size 減半
    vram_usage_threshold_ratio: float = 0.85
    # batch size 的絕對下限 (低於此值直接報錯，不再縮減)
    min_batch_size: int = 2

    @property
    def usable_vram_mb(self) -> int:
        """計算實際可用的 VRAM (MB)。"""
        return self.total_vram_mb - self.safety_margin_mb

    def get_phase_budget(self, phase: str) -> int:
        """
        依階段名稱回傳 VRAM 預算 (MB)。

        Args:
            phase: 階段識別碼，可選值:
                   "synthetic" (A), "mining" (B),
                   "training_small" (C-small), "training_base" (C-base),
                   "retrieval" (D), "evaluation" (E)
        """
        budgets = {
            "synthetic": self.phase_a_budget_mb,
            "mining": self.phase_b_budget_mb,
            "training_small": self.phase_c_small_budget_mb,
            "training_base": self.phase_c_base_budget_mb,
            "retrieval": self.phase_d_budget_mb,
            "evaluation": self.phase_e_budget_mb,
        }
        if phase not in budgets:
            raise ValueError(
                f"未知階段: {phase}，可選值: {list(budgets.keys())}"
            )
        return budgets[phase]
