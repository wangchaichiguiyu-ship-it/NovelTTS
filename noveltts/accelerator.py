"""设备加速：把「硬件能力」翻译成「一组可执行的优化开关」。

为什么单独一个模块
------------------

同一份代码要跑在三种差别很大的环境上：

* **云端 NVIDIA**：追求吞吐，显存 24~80 GB，能用 FlashAttention 2 与 TF32
* **本机 Apple Silicon**：统一内存 16 GB，只能保守，**绝不能**开 CUDA 专属开关
* **纯 CPU**：能用就行，只要别崩

如果把这些判断散落在 engine / synthesis 里，很快就会变成一堆
``if device == "cuda"`` 的意大利面，而且**换个设备就悄悄失效**。
所以这里集中处理，产出一个 :class:`AccelProfile` 描述"这台机器该怎么跑"。

全部开关都由 ``config.yaml`` 的 ``generation`` 段驱动；写 ``auto`` 就让本模块
按实测硬件决定。**写死的值永远优先**，方便你手工压测。

设计原则
--------

1. **不做任何假设**：显存、算力、flash-attn 是否可用，一律**实测**。
2. **不越权**：非 CUDA 设备上绝不设置 CUDA 相关标志（某些版本会直接抛错）。
3. **可解释**：每个决定都记进 ``profile.reasons``，日志里能看清"为什么这么配"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import GenerationConfig
from .utils import get_logger

logger = get_logger("accelerator")

#: bf16 权重 + 说话人编码器 + 运行时开销的固定占用（GB）。
#: 1.7B 参数在 bf16 下约 3.4 GB；RTX 3090 上实测 ``max_memory_allocated`` 约 4.7 GB
#: （含参考音频编码与 KV cache 的起步开销），这里取 4.5。
_BASE_VRAM_GB = 4.5

#: 每个「批内额外样本」的显存估算（GB）。
#:
#: 实测（RTX 3090，片段平均约 60 字符 ≈ 5 秒音频，max_new_tokens=1024）：
#:
#: ==========  ==========  ==================
#: 批大小       峰值显存     相对上一行增量
#: ==========  ==========  ==================
#: 1           4.7 G       —
#: 2           5.4 G       +0.70 G
#: 5           7.7 G       +0.77 G / 样本
#: ==========  ==========  ==================
#:
#: 即约 **0.75 GB/样本**。这里取 1.2 留约 60% 余量——因为显存还会随
#: «片段实际时长» 与 «max_new_tokens» 变化，而那两个都是可配置的。
_PER_SAMPLE_VRAM_GB = 1.2

#: 安全系数：给 PyTorch 的碎片化留空间。
_VRAM_SAFETY = 0.8

#: 默认最大批大小。
#:
#: 为什么不无限放大：批越大 OOM 的代价越高（整批重算），而且收益递减。
#: 本书片段平均只有 60 字符，16 段一批已足够把 GPU 喂饱。
_MAX_BATCH = 16


@dataclass
class DeviceInfo:
    """实测到的一次设备信息。

    Attributes:
        kind: ``cuda`` / ``mps`` / ``cpu``。
        name: 设备名（如 ``NVIDIA GeForce RTX 4090``）。
        index: CUDA 设备序号；非 CUDA 为 0。
        total_vram_gb: 总显存（GB）；非 CUDA 为 0.0。
        free_vram_gb: 当前可用显存（GB）；探测不到时为 0.0。
        compute_capability: CUDA 算力，如 ``(8, 9)``；非 CUDA 为 ``None``。
        cuda_version: PyTorch 编译时使用的 CUDA 版本；无则 ``None``。
        torch_version: PyTorch 版本。
    """

    kind: str = "cpu"
    name: str = "CPU"
    index: int = 0
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    compute_capability: tuple[int, int] | None = None
    cuda_version: str | None = None
    torch_version: str = ""

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def describe(self) -> str:
        parts = [f"{self.kind}:{self.name}"]
        if self.is_cuda:
            parts.append(f"显存 {self.total_vram_gb:.1f}G（可用 {self.free_vram_gb:.1f}G）")
            if self.compute_capability:
                parts.append(f"算力 sm_{self.compute_capability[0]}{self.compute_capability[1]}")
        parts.append(f"torch {self.torch_version}")
        if self.cuda_version:
            parts.append(f"cuda {self.cuda_version}")
        return " | ".join(parts)


@dataclass
class AccelProfile:
    """一次运行要用的全部加速开关。"""

    device: str = "cpu"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    batch_size: int = 1
    allow_tf32: bool = False
    cudnn_benchmark: bool = False
    clear_cache_every: int = 0
    empty_cache: bool = False

    info: DeviceInfo = field(default_factory=DeviceInfo)
    reasons: list[str] = field(default_factory=list)

    def explain(self) -> str:
        """返回多行说明，用于日志。"""
        lines = [
            f"设备：{self.info.describe()}",
            f"配置：device={self.device} dtype={self.dtype} "
            f"attn={self.attn_implementation} batch_size={self.batch_size}",
            f"开关：tf32={self.allow_tf32} cudnn_benchmark={self.cudnn_benchmark} "
            f"empty_cache={self.empty_cache} clear_every={self.clear_cache_every}",
        ]
        if self.reasons:
            lines.append("判定依据：")
            lines.extend(f"  - {reason}" for reason in self.reasons)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 设备探测
# --------------------------------------------------------------------------- #


def inspect_device(requested: str = "auto", *, index: int = 0) -> DeviceInfo:
    """实测当前设备信息。

    Args:
        requested: ``auto`` / ``cuda`` / ``cuda:N`` / ``mps`` / ``cpu``。
        index: CUDA 设备序号（``requested`` 里没写 ``:N`` 时生效）。

    Returns:
        :class:`DeviceInfo`。

    Note:
        探测失败（例如老版本 torch 没有某些 API）不会抛异常，而是退回
        ``cpu``——跑得慢总比跑不起来好。
    """
    try:
        import torch
    except ImportError:
        return DeviceInfo(kind="cpu", name="CPU（未安装 torch）")

    version = getattr(torch, "__version__", "")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

    target = (requested or "auto").lower()
    if target.startswith("cuda") and ":" in target:
        try:
            index = int(target.split(":", 1)[1])
        except ValueError:
            index = 0
        target = "cuda"

    # --- CUDA ---
    if target in {"auto", "cuda"} and _cuda_available(torch):
        try:
            properties = torch.cuda.get_device_properties(index)
            total = float(properties.total_memory) / 1024**3
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
                free = float(free_bytes) / 1024**3
            except Exception:  # noqa: BLE001 - 某些驱动/版本不支持
                free = 0.0
            capability = None
            if hasattr(properties, "major") and hasattr(properties, "minor"):
                capability = (int(properties.major), int(properties.minor))
            return DeviceInfo(
                kind="cuda",
                name=str(getattr(properties, "name", "NVIDIA GPU")),
                index=index,
                total_vram_gb=round(total, 2),
                free_vram_gb=round(free, 2),
                compute_capability=capability,
                cuda_version=str(cuda_version) if cuda_version else None,
                torch_version=str(version),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 CUDA 设备信息失败，回退 CPU：%s", exc)
            return DeviceInfo(kind="cpu", name="CPU", torch_version=str(version))

    if target == "cuda":  # 显式要 CUDA 但没有
        logger.warning("指定了 CUDA 但没有可用 GPU，回退 CPU")

    # --- MPS ---
    if target in {"auto", "mps"} and _mps_available(torch):
        return DeviceInfo(
            kind="mps",
            name="Apple Silicon (MPS)",
            torch_version=str(version),
        )

    return DeviceInfo(kind="cpu", name="CPU", torch_version=str(version))


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _mps_available(torch: Any) -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def flash_attention_available() -> tuple[bool, str]:
    """探测 FlashAttention 2 是否真的可用。

    只看"能不能 import"，因为配置里写 ``flash_attention_2`` 而实际没装时，
    transformers 会在**加载模型之后**才报错——那时候已经浪费了下载与加载时间。

    Returns:
        ``(是否可用, 说明)``。
    """
    try:
        import flash_attn  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False, "未安装 flash-attn（pip install flash-attn --no-build-isolation）"
    except Exception as exc:  # noqa: BLE001 - 装坏了（例如缺 CUDA 库）
        return False, f"flash-attn 已安装但导入失败：{exc}"

    version = ""
    try:
        import importlib.metadata as md

        version = md.version("flash-attn")
    except Exception:  # noqa: BLE001
        pass
    return True, f"可用（flash-attn {version}）" if version else "可用"


# --------------------------------------------------------------------------- #
# 批大小推导
# --------------------------------------------------------------------------- #


def recommend_batch_size(info: DeviceInfo, *, max_batch: int = _MAX_BATCH) -> tuple[int, str]:
    """按实测显存推荐批大小。

    估算模型（保守）：

    * 固定占用 :data:`_BASE_VRAM_GB`（bf16 权重 + 说话人编码器 + 运行时）
    * 每个额外样本 :data:`_PER_SAMPLE_VRAM_GB`
    * 再乘 :data:`_VRAM_SAFETY` 安全系数，给显存碎片留余地

    Args:
        info: 设备信息。
        max_batch: 上限（收益递减，且更难排查 OOM）。

    Returns:
        ``(批大小, 判定说明)``。

    Note:
        基准是"可用显存"与"总显存 × 安全系数"里的**较小值**。用可用显存是
        因为实例上可能已有别的进程占着卡；只看总显存会乐观到 OOM。
    """
    if not info.is_cuda or info.total_vram_gb <= 0:
        return 1, "非 CUDA 设备：批大小固定 1（统一内存 / CPU 上批处理只会更危险）"

    budget = info.total_vram_gb * _VRAM_SAFETY
    if info.free_vram_gb > 0:
        budget = min(budget, info.free_vram_gb)
        basis = f"总显存 {info.total_vram_gb:.1f}G × {_VRAM_SAFETY} 与可用 {info.free_vram_gb:.1f}G 取小"
    else:
        basis = f"总显存 {info.total_vram_gb:.1f}G × {_VRAM_SAFETY}（探测不到可用显存）"

    usable = budget - _BASE_VRAM_GB
    if usable <= 0:
        return 1, f"{basis} = {budget:.1f}G，不足以容纳固定占用 {_BASE_VRAM_GB}G，批大小取 1"

    raw = 1 + int(math.floor(usable / _PER_SAMPLE_VRAM_GB))
    batch = max(1, min(max_batch, raw))
    reason = (
        f"{basis} = {budget:.1f}G；扣除固定占用 {_BASE_VRAM_GB}G 后约 {usable:.1f}G，"
        f"按每样本 {_PER_SAMPLE_VRAM_GB}G 估算得 {batch}"
    )
    if raw > max_batch:
        reason += f"（已按上限 {max_batch} 截断）"
    return batch, reason


def recommend_clear_interval(info: DeviceInfo) -> int:
    """推荐"每多少段清理一次设备缓存"。

    * CUDA / MPS：每 50 段一次。太频繁会打断流水线，太久缓存会缓慢膨胀。
    * CPU：不清理（没有设备缓存可清）。
    """
    return 0 if info.kind == "cpu" else 50


# --------------------------------------------------------------------------- #
# 应用开关
# --------------------------------------------------------------------------- #


def apply_torch_settings(profile: AccelProfile) -> None:
    """把 profile 里的全局开关真正应用到 torch。

    目前只影响 CUDA：

    * ``allow_tf32``：允许 TF32 参与矩阵乘。在 Ampere 及以后的卡上，
      能换来可观的吞吐提升，且对语音合成的听感影响可忽略。
    * ``cudnn_benchmark``：让 cuDNN 自动为固定形状挑最快算法。
      我们的输入形状在同一章内是稳定的，正好适合。

    Note:
        非 CUDA 设备上**什么都不做**——某些 torch 版本在 CPU/MPS 上碰
        ``torch.backends.cuda`` 会直接抛错。
    """
    if not profile.info.is_cuda:
        return

    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    applied: list[str] = []

    if profile.allow_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            applied.append("TF32")
        except Exception as exc:  # noqa: BLE001
            logger.debug("开启 TF32 失败（忽略）：%s", exc)

    if profile.cudnn_benchmark:
        try:
            torch.backends.cudnn.benchmark = True
            applied.append("cudnn.benchmark")
        except Exception as exc:  # noqa: BLE001
            logger.debug("开启 cudnn.benchmark 失败（忽略）：%s", exc)

    if applied:
        logger.info("已开启 CUDA 加速：%s", "、".join(applied))


def empty_device_cache(device: str) -> None:
    """清理指定设备的缓存（CUDA / MPS）。

    长时间跑批时不清理的话，分配器缓存会缓慢膨胀；对 16 GB 机器尤其明显。
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception as exc:  # noqa: BLE001 - 清缓存失败不该中断跑批
        logger.debug("清理设备缓存失败（忽略）：%s", exc)


def is_oom_error(exc: BaseException) -> bool:
    """判断异常是不是"显存不足"。

    不同后端抛的异常类型与文案都不一致，所以多重判定：

    * CUDA：``torch.cuda.OutOfMemoryError``（``RuntimeError`` 的子类）
    * MPS：通常只是普通 ``RuntimeError``，文本里带 "out of memory"
    * 兜底：文本匹配 ``out of memory`` / ``outofmemory`` /
      ``cuda error: out of memory`` / ``cublas`` + ``alloc``

    Note:
        刻意**不**用 ``"cuda error" and "memory"`` 这种松散的组合判断——
        那样会把 "CUDA error: device-side assert" 之类也误判成 OOM，
        导致无意义地缩小批大小。
    """
    try:
        import torch

        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except ImportError:  # pragma: no cover
        pass

    text = str(exc).lower()
    markers = (
        "out of memory",
        "outofmemory",
        "out-of-memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "mps backend out of memory",
    )
    return any(marker in text for marker in markers)


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #


def choose_attn_implementation(
    requested: str,
    info: DeviceInfo,
    *,
    flash_probe: "Callable[[], tuple[bool, str]] | None" = None,
) -> tuple[str, str]:
    """决定注意力实现，并给出理由。

    抽成独立函数是为了**可以在没有 GPU 的机器上测试**：调用方注入一个假的
    :class:`DeviceInfo` 与假的 flash-attn 探针即可，不需要真的插一张卡。

    Args:
        requested: 配置里的值（``auto`` / ``flash_attention_2`` / ``sdpa`` / ``eager``）。
        info: 实测设备信息。
        flash_probe: 探测 flash-attn 的函数；``None`` 时用
            :func:`flash_attention_available`。

    Returns:
        ``(实际使用的实现, 理由)``。
    """
    probe = flash_probe or flash_attention_available
    requested = (requested or "auto").lower()

    if requested == "auto":
        if not info.is_cuda:
            return "sdpa", "注意力 auto -> sdpa（非 CUDA 设备）"
        available, detail = probe()
        if available:
            return "flash_attention_2", f"注意力 auto -> flash_attention_2（{detail}）"
        return "sdpa", f"注意力 auto -> sdpa（{detail}）"

    if requested.startswith("flash"):
        if not info.is_cuda:
            return "sdpa", "配置要求 flash_attention_2，但设备不是 CUDA -> 改用 sdpa"
        available, detail = probe()
        if not available:
            return "sdpa", f"配置要求 flash_attention_2 但不可用（{detail}）-> 改用 sdpa"
        return "flash_attention_2", f"注意力由配置指定为 flash_attention_2（{detail}）"

    return requested, f"注意力由配置指定为 {requested}"


def choose_tf32(configured: object, info: DeviceInfo) -> tuple[bool, str]:
    """决定是否开启 TF32。

    TF32 只在**算力 >= 8.0** 的 NVIDIA 卡上存在（Ampere 及以后）。
    在更老的卡上设置这个标志是无效甚至有害的，所以这里显式拦掉。
    """
    if str(configured).lower() == "auto":
        if not info.is_cuda:
            return False, "TF32 auto -> 关闭（仅 CUDA 有意义）"
        capability = info.compute_capability
        if capability and capability[0] < 8:
            return False, (
                f"TF32 需要算力 >= 8.0，当前 sm_{capability[0]}{capability[1]} -> 关闭"
            )
        suffix = f"sm_{capability[0]}{capability[1]}" if capability else "算力未知（乐观开启）"
        return True, f"TF32 auto -> 开启（CUDA，{suffix}）"

    enabled = bool(configured) and info.is_cuda
    if bool(configured) and not info.is_cuda:
        return False, "配置要求开启 TF32，但非 CUDA 设备 -> 忽略"
    return enabled, f"TF32 由配置指定为 {enabled}"


def build_profile(
    generation: GenerationConfig,
    *,
    device: str | None = None,
    device_info: DeviceInfo | None = None,
    flash_probe: "Callable[[], tuple[bool, str]] | None" = None,
) -> AccelProfile:
    """按配置 + 实测硬件，产出完整的加速方案。

    Args:
        generation: 生成配置（``device`` / ``dtype`` / ``batch_size`` /
            ``attn_implementation`` / ``tf32`` / ``cudnn_benchmark``）。
        device: 命令行覆盖的设备；``None`` 时用配置值。
        device_info: 注入设备信息（测试用）；``None`` 时实测。
        flash_probe: 注入 flash-attn 探针（测试用）。

    Returns:
        :class:`AccelProfile`。

    规则总览：

    ====================  ==================================================
    配置项                行为
    ====================  ==================================================
    ``device: auto``      实测：CUDA > MPS > CPU
    ``batch_size: auto``  按显存推导；写数字则原样使用
    ``batch_size: -1``    同上（负数视为 auto，兼容习惯写法）
    ``attn_implementation:
    auto``                CUDA 且 flash-attn 可用 -> flash_attention_2，否则 sdpa
    ``tf32: auto``        CUDA 且算力 >= 8.0 时开启；非 CUDA 忽略
    ``cudnn_benchmark:
    auto``                CUDA 上开启；非 CUDA 忽略
    ``clear_cache_every:
    0``                   CUDA/MPS 每 50 段；CPU 不清理
    ====================  ==================================================
    """
    info = device_info if device_info is not None else inspect_device(device or generation.device)
    reasons: list[str] = []

    resolved_device = "cuda" if info.kind == "cuda" else info.kind
    if info.is_cuda:
        resolved_device = f"cuda:{info.index}"

    # --- 批大小 ---
    if generation.batch_size in {None, -1} or str(generation.batch_size).lower() == "auto":
        batch_size, reason = recommend_batch_size(info)
        reasons.append(f"批大小 auto -> {batch_size}：{reason}")
    else:
        batch_size = max(1, int(generation.batch_size))
        reasons.append(f"批大小由配置写死为 {batch_size}")

    # --- 注意力实现 ---
    attn, attn_reason = choose_attn_implementation(
        generation.attn_implementation, info, flash_probe=flash_probe
    )
    reasons.append(attn_reason)

    # --- TF32 / cudnn.benchmark ---
    allow_tf32, tf32_reason = choose_tf32(getattr(generation, "tf32", "auto"), info)
    reasons.append(tf32_reason)

    benchmark_cfg = getattr(generation, "cudnn_benchmark", "auto")
    if str(benchmark_cfg).lower() == "auto":
        cudnn_benchmark = info.is_cuda
        reasons.append(f"cudnn.benchmark auto -> {'开启' if cudnn_benchmark else '关闭'}（仅 CUDA）")
    else:
        cudnn_benchmark = bool(benchmark_cfg) and info.is_cuda
        reasons.append(f"cudnn.benchmark 由配置指定为 {cudnn_benchmark}")

    # --- 缓存清理 ---
    clear_every = generation.clear_cache_every
    if clear_every <= 0:
        clear_every = recommend_clear_interval(info)
        if clear_every:
            reasons.append(f"缓存清理间隔 auto -> 每 {clear_every} 段清理")
        else:
            reasons.append("缓存清理间隔 auto -> CPU 无需清理")
    else:
        reasons.append(f"缓存清理间隔由配置指定为每 {clear_every} 段")

    return AccelProfile(
        device=resolved_device,
        dtype=generation.dtype,
        attn_implementation=attn,
        batch_size=batch_size,
        allow_tf32=allow_tf32,
        cudnn_benchmark=cudnn_benchmark,
        clear_cache_every=clear_every,
        empty_cache=info.kind != "cpu",
        info=info,
        reasons=reasons,
    )


def preflight(generation: GenerationConfig, *, device: str | None = None) -> AccelProfile:
    """开跑前的自检：探测硬件、算出方案、应用开关、打印结论。

    在加载模型**之前**调用。这样配置写错（例如要求 flash-attn 但没装）会在
    几秒内以清晰的日志暴露，而不是等权重加载完再崩。

    Args:
        generation: 生成配置。
        device: 命令行覆盖的设备。

    Returns:
        已应用的 :class:`AccelProfile`。
    """
    profile = build_profile(generation, device=device)
    apply_torch_settings(profile)

    logger.info("硬件与加速方案：\n%s", profile.explain())

    if profile.info.is_cuda and profile.batch_size == 1 and profile.info.total_vram_gb >= 20:
        logger.warning(
            "检测到 %.0fG 显存但批大小为 1，吞吐会被浪费。"
            "可把 generation.batch_size 设为 auto，或显式设为 4~8。",
            profile.info.total_vram_gb,
        )

    return profile
