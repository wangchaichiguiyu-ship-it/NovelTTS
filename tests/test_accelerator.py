"""设备加速测试。

**全部用注入的假硬件信息**，因此可以在没有 NVIDIA 卡的 Mac 上完整验证
CUDA 分支的逻辑——这是本文件存在的核心价值：等你真的上了 AutoDL，
批大小/注意力/TF32 的判定已经被测过了，不需要现场试错。
"""

from __future__ import annotations

import pytest

from noveltts.accelerator import (
    AccelProfile,
    DeviceInfo,
    build_profile,
    choose_attn_implementation,
    choose_tf32,
    flash_attention_available,
    inspect_device,
    is_oom_error,
    recommend_batch_size,
    recommend_clear_interval,
)
from noveltts.config import GenerationConfig


# --------------------------------------------------------------------------- #
# 假硬件
# --------------------------------------------------------------------------- #


def cuda_info(
    vram: float = 24.0,
    *,
    capability: tuple[int, int] | None = (8, 9),
    free: float = 0.0,
    index: int = 0,
    name: str = "NVIDIA GeForce RTX 4090",
) -> DeviceInfo:
    """造一个 CUDA 设备（不会真的用 GPU）。"""
    return DeviceInfo(
        kind="cuda",
        name=name,
        index=index,
        total_vram_gb=vram,
        free_vram_gb=free,
        compute_capability=capability,
        cuda_version="12.4",
        torch_version="2.4.0",
    )


def flash_yes() -> tuple[bool, str]:
    return True, "可用（flash-attn 2.7.0）"


def flash_no() -> tuple[bool, str]:
    return False, "未安装 flash-attn"


# --------------------------------------------------------------------------- #
# 设备探测
# --------------------------------------------------------------------------- #


def test_inspect_device_returns_valid_kind() -> None:
    info = inspect_device("auto")
    assert info.kind in {"cuda", "mps", "cpu"}
    assert info.torch_version


def test_inspect_device_falls_back_without_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("本机有 CUDA，无法测试回退")
    assert inspect_device("cuda").kind == "cpu"


def test_device_info_describe_mentions_vram_for_cuda() -> None:
    text = cuda_info(24.0).describe()
    assert "24.0G" in text
    assert "sm_89" in text
    assert "cuda" in text


def test_device_info_describe_for_non_cuda() -> None:
    text = DeviceInfo(kind="mps", name="Apple Silicon (MPS)").describe()
    assert "mps" in text
    assert "G" not in text.split("|")[0]  # 不显示显存


# --------------------------------------------------------------------------- #
# 批大小推导
# --------------------------------------------------------------------------- #


def test_non_cuda_batch_is_always_one() -> None:
    for kind in ("mps", "cpu"):
        info = DeviceInfo(kind=kind)
        batch, reason = recommend_batch_size(info)
        assert batch == 1
        assert "固定 1" in reason


@pytest.mark.parametrize(
    ("vram", "expected"),
    [
        # 依据：固定占用 4.5G，每样本 1.2G（实测 0.75G 留 60% 余量），
        # 预算 = 总显存 × 0.8，上限 16。
        (8, 2),
        (12, 5),
        (16, 7),
        (24, 13),   # RTX 3090 的档位
        (32, 16),   # 触达上限
        (48, 16),
        (80, 16),
    ],
)
def test_batch_scales_with_vram(vram: float, expected: int) -> None:
    batch, _ = recommend_batch_size(cuda_info(vram))
    assert batch == expected, f"{vram}G 期望 {expected}，得到 {batch}"


def test_batch_is_capped() -> None:
    batch, reason = recommend_batch_size(cuda_info(200.0))
    assert batch == 16
    assert "截断" in reason


def test_batch_never_below_one() -> None:
    """显存小到装不下模型时，也只能返回 1（而不是 0 或负数）。"""
    batch, reason = recommend_batch_size(cuda_info(4.0))
    assert batch == 1
    assert "不足" in reason


def test_batch_uses_free_vram_when_available() -> None:
    """可用显存远小于总显存时（卡被别的进程占了），必须按可用值算。"""
    generous, _ = recommend_batch_size(cuda_info(48.0, free=46.0))
    squeezed, reason = recommend_batch_size(cuda_info(48.0, free=12.0))
    assert squeezed < generous
    assert "可用" in reason


def test_batch_ignores_free_vram_when_unknown() -> None:
    """探测不到可用显存（free=0）时退回总显存口径，而不是算出 1。"""
    batch, reason = recommend_batch_size(cuda_info(24.0, free=0.0))
    assert batch > 1
    assert "探测不到可用显存" in reason


# --------------------------------------------------------------------------- #
# 注意力实现
# --------------------------------------------------------------------------- #


def test_attn_auto_picks_flash_on_cuda() -> None:
    attn, reason = choose_attn_implementation("auto", cuda_info(), flash_probe=flash_yes)
    assert attn == "flash_attention_2"
    assert "flash" in reason


def test_attn_auto_falls_back_when_flash_missing() -> None:
    """CUDA 上没装 flash-attn 时必须退回 sdpa，而不是让加载崩掉。"""
    attn, reason = choose_attn_implementation("auto", cuda_info(), flash_probe=flash_no)
    assert attn == "sdpa"
    assert "未安装" in reason


def test_attn_auto_on_non_cuda_is_sdpa() -> None:
    for kind in ("mps", "cpu"):
        attn, _ = choose_attn_implementation("auto", DeviceInfo(kind=kind), flash_probe=flash_yes)
        assert attn == "sdpa"


def test_attn_explicit_flash_downgrades_without_cuda() -> None:
    attn, reason = choose_attn_implementation(
        "flash_attention_2", DeviceInfo(kind="mps"), flash_probe=flash_yes
    )
    assert attn == "sdpa"
    assert "不是 CUDA" in reason


def test_attn_explicit_flash_downgrades_without_package() -> None:
    attn, reason = choose_attn_implementation(
        "flash_attention_2", cuda_info(), flash_probe=flash_no
    )
    assert attn == "sdpa"
    assert "不可用" in reason


def test_attn_explicit_sdpa_is_respected() -> None:
    attn, reason = choose_attn_implementation("sdpa", cuda_info(), flash_probe=flash_yes)
    assert attn == "sdpa"
    assert "指定" in reason


def test_attn_eager_is_respected() -> None:
    attn, _ = choose_attn_implementation("eager", cuda_info(), flash_probe=flash_yes)
    assert attn == "eager"


def test_flash_probe_does_not_raise() -> None:
    """探测函数在任何环境下都必须安全返回（本机没装 flash-attn）。"""
    available, detail = flash_attention_available()
    assert isinstance(available, bool)
    assert detail


# --------------------------------------------------------------------------- #
# TF32
# --------------------------------------------------------------------------- #


def test_tf32_on_modern_cuda() -> None:
    enabled, reason = choose_tf32("auto", cuda_info(capability=(8, 9)))
    assert enabled is True
    assert "sm_89" in reason


def test_tf32_on_ampere_30_series() -> None:
    """30 系是 sm_86，同样支持 TF32。"""
    enabled, _ = choose_tf32("auto", cuda_info(capability=(8, 6)))
    assert enabled is True


@pytest.mark.parametrize("capability", [(7, 5), (7, 0), (6, 1)])
def test_tf32_disabled_on_old_cuda(capability: tuple[int, int]) -> None:
    """20 系及更老不支持 TF32，必须显式关闭。"""
    enabled, reason = choose_tf32("auto", cuda_info(capability=capability))
    assert enabled is False
    assert "算力" in reason


def test_tf32_disabled_on_non_cuda() -> None:
    for kind in ("mps", "cpu"):
        enabled, _ = choose_tf32("auto", DeviceInfo(kind=kind))
        assert enabled is False


def test_tf32_explicit_true_ignored_on_non_cuda() -> None:
    """配置写 true 但设备不是 CUDA：忽略并说明，而不是照做。"""
    enabled, reason = choose_tf32(True, DeviceInfo(kind="mps"))
    assert enabled is False
    assert "非 CUDA" in reason


def test_tf32_explicit_false_respected_on_cuda() -> None:
    enabled, _ = choose_tf32(False, cuda_info())
    assert enabled is False


# --------------------------------------------------------------------------- #
# 缓存清理间隔
# --------------------------------------------------------------------------- #


def test_clear_interval_zero_on_cpu() -> None:
    assert recommend_clear_interval(DeviceInfo(kind="cpu")) == 0


def test_clear_interval_positive_on_gpu() -> None:
    assert recommend_clear_interval(cuda_info()) > 0
    assert recommend_clear_interval(DeviceInfo(kind="mps")) > 0


# --------------------------------------------------------------------------- #
# 组装 profile
# --------------------------------------------------------------------------- #


def test_profile_for_24g_cuda_with_flash() -> None:
    generation = GenerationConfig(device="cuda", batch_size="auto")
    profile = build_profile(generation, device_info=cuda_info(24.0), flash_probe=flash_yes)

    assert profile.device == "cuda:0"
    assert profile.attn_implementation == "flash_attention_2"
    assert profile.batch_size == 13
    assert profile.allow_tf32 is True
    assert profile.cudnn_benchmark is True
    assert profile.clear_cache_every == 50
    assert profile.empty_cache is True
    assert len(profile.reasons) >= 5


def test_profile_respects_explicit_batch_size() -> None:
    generation = GenerationConfig(device="cuda", batch_size=2)
    profile = build_profile(generation, device_info=cuda_info(48.0), flash_probe=flash_yes)
    assert profile.batch_size == 2
    assert "写死" in " ".join(profile.reasons)


def test_profile_auto_batch_accepts_minus_one() -> None:
    generation = GenerationConfig(device="cuda", batch_size=-1)
    profile = build_profile(generation, device_info=cuda_info(24.0), flash_probe=flash_yes)
    assert profile.batch_size == 13


def test_profile_for_mac_never_enables_cuda_features() -> None:
    """本机（MPS）绝不能出现任何 CUDA 专属开关——那会直接报错。"""
    generation = GenerationConfig(device="mps", batch_size="auto")
    profile = build_profile(generation, device_info=DeviceInfo(kind="mps"), flash_probe=flash_yes)

    assert profile.device == "mps"
    assert profile.attn_implementation == "sdpa"
    assert profile.batch_size == 1
    assert profile.allow_tf32 is False
    assert profile.cudnn_benchmark is False


def test_profile_for_cpu() -> None:
    generation = GenerationConfig(device="cpu", batch_size="auto")
    profile = build_profile(generation, device_info=DeviceInfo(kind="cpu"), flash_probe=flash_no)

    assert profile.device == "cpu"
    assert profile.batch_size == 1
    assert profile.clear_cache_every == 0
    assert profile.empty_cache is False


def test_profile_cuda_index_is_preserved() -> None:
    generation = GenerationConfig(device="cuda", batch_size=1)
    profile = build_profile(
        generation, device_info=cuda_info(vram=24.0, index=3), flash_probe=flash_yes
    )
    assert profile.device == "cuda:3"


def test_profile_explain_is_readable() -> None:
    profile = build_profile(
        GenerationConfig(device="cuda", batch_size="auto"),
        device_info=cuda_info(24.0),
        flash_probe=flash_yes,
    )
    text = profile.explain()
    assert "设备：" in text
    assert "配置：" in text
    assert "开关：" in text
    assert "判定依据：" in text


def test_profile_survives_unknown_compute_capability() -> None:
    """算力探测不到时应当乐观开启 TF32，而不是崩掉。"""
    generation = GenerationConfig(device="cuda", batch_size=1)
    profile = build_profile(
        generation, device_info=cuda_info(capability=None), flash_probe=flash_yes
    )
    assert profile.allow_tf32 is True


def test_preflight_does_not_crash_offline() -> None:
    """在线下（无 GPU）跑 preflight 也必须安全返回。"""
    from noveltts.accelerator import preflight

    profile = preflight(GenerationConfig(device="cpu", batch_size=1))
    assert isinstance(profile, AccelProfile)
    assert profile.device in {"cuda:0", "mps", "cpu"}


# --------------------------------------------------------------------------- #
# OOM 判定
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "torch.cuda.OutOfMemoryError: CUDA out of memory",
        "MPS backend out of memory (MPS allocated: 12.3 GB)",
        "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate",
        "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    ],
)
def test_oom_detected(message: str) -> None:
    assert is_oom_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: device-side assert triggered",
        "RuntimeError: engine closed",
        "FileNotFoundError: no such file or directory",
        "ValueError: texts 不能为空",
        "CUDA error: an illegal memory access was encountered",
    ],
)
def test_non_oom_not_detected(message: str) -> None:
    """非 OOM 的 CUDA 错误不能被误判——否则会无意义地缩小批大小。"""
    assert is_oom_error(RuntimeError(message)) is False
