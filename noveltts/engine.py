"""Qwen3-TTS 推理封装。

**这是唯一会 import torch / qwen_tts 的模块**，而且这些 import 全部写在函数
内部——所以 ``import noveltts.engine`` 本身是零成本的，单元测试可以在没装
torch 的机器上跑（用 :class:`FakeSynthesisEngine` 替换）。

内存设计（16 GB Mac）
--------------------

* **权重只加载一次**。:class:`SynthesisEngine` 持有一个模型实例，整个跑批
  过程复用，不做「每段重新 load」。1.7B 参数 bf16 约 3.4 GB，加载两次就完蛋。
* **音色 prompt 只算一次**。参考音频要被 speech tokenizer 编码 + 说话人向量
  提取；这两个结果通过 ``create_voice_clone_prompt`` 缓存复用，
  否则每段都要重算，慢且浪费显存。
* **逐段返回、立即落盘**。``synthesize`` 返回的是 ``numpy`` 数组，调用方
  （:mod:`noveltts.synthesis`）会马上写盘并释放；引擎内部不保留任何音频。
* **``batch_size`` 默认 1**。批处理能提速，但会把多段音频同时留在显存/内存里，
  在统一内存架构的 Mac 上尤其危险：显存和内存抢同一块物理内存。
  云端显存充足时再调大。

采样率说明
----------

Qwen3-TTS-12Hz 的语音 tokenizer 解码输出固定 24 kHz，与
:data:`noveltts.SAMPLE_RATE` 一致。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from . import accelerator
from .config import GenerationConfig, VoiceConfig
from .utils import get_logger

logger = get_logger("engine")


# --------------------------------------------------------------------------- #
# 返回结构
# --------------------------------------------------------------------------- #


@dataclass
class SynthesisResult:
    """一次合成请求的原始结果。

    Attributes:
        waveforms: 每段的 float32 numpy 数组（24 kHz 单声道）。
        sample_rate: 采样率。
        prompt_seconds: 构建音色 prompt 耗时（秒）；复用缓存时为 0。
        generate_seconds: 实际生成耗时（秒）。
        texts: 与 ``waveforms`` 对应的输入文本（便于日志与排查）。
    """

    waveforms: list[Any] = field(default_factory=list)
    sample_rate: int = 24_000
    prompt_seconds: float = 0.0
    generate_seconds: float = 0.0
    texts: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.waveforms)


@runtime_checkable
class SynthesisEngine(Protocol):
    """合成引擎协议。

    :class:`QwenTTSEngine` 是真实实现；测试里用
    :class:`FakeSynthesisEngine`。只要满足这个协议，上层
    :mod:`noveltts.synthesis` 不关心底下是什么。
    """

    sample_rate: int

    def synthesize(self, texts: list[str], *, language: str | None = None) -> SynthesisResult:
        """把一批文本合成为音频。"""
        ...

    def close(self) -> None:
        """释放模型与缓存。"""
        ...


# --------------------------------------------------------------------------- #
# 设备选择（薄封装，实际逻辑统一在 accelerator 里）
# --------------------------------------------------------------------------- #


def resolve_device(requested: str = "auto") -> str:
    """决定用哪个设备。

    顺序：``cuda`` > ``mps`` > ``cpu``。``requested`` 不是 ``auto`` 时
    会校验其可用性，不可用就退回 ``cpu`` 并给出警告（而不是直接崩）。

    Args:
        requested: ``"auto"`` / ``"cuda"`` / ``"cuda:1"`` / ``"mps"`` / ``"cpu"``。

    Returns:
        实际要用的设备字符串。

    Note:
        具体探测逻辑在 :func:`noveltts.accelerator.inspect_device`，
        这里只是一层保持向后兼容的封装。
    """
    info = accelerator.inspect_device(requested)
    if info.kind == "cuda":
        return f"cuda:{info.index}"
    return info.kind


def resolve_attn_implementation(requested: str, device: str) -> str:
    """决定注意力实现。

    ``flash_attention_2`` 只在 CUDA 上**且已安装时**可用；MPS / CPU 上用
    ``sdpa`` 或 ``eager``。给错实现会直接报错，所以这里替用户兜住。

    Note:
        与 :func:`resolve_device` 一样，是对
        :func:`noveltts.accelerator.build_profile` 的向后兼容封装。
    """
    generation = GenerationConfig(
        device=device,  # type: ignore[arg-type]
        attn_implementation=requested,
        batch_size=1,
    )
    return accelerator.build_profile(generation, device=device).attn_implementation


def _empty_device_cache(device: str) -> None:
    """清理设备缓存（委托给 accelerator）。

    每段生成后调用一次，把 MPS / CUDA 的临时缓冲还给系统。对 16 GB 机器
    很重要：不清理的话几百段之后缓存会持续膨胀。
    """
    accelerator.empty_device_cache(device)


# --------------------------------------------------------------------------- #
# 真实引擎
# --------------------------------------------------------------------------- #


class QwenTTSEngine:
    """Qwen3-TTS Base（声音克隆）推理引擎。

    Args:
        generation: 生成参数配置。
        voice: 音色克隆配置。

    Example:
        >>> engine = QwenTTSEngine(generation, voice)   # doctest: +SKIP
        >>> result = engine.synthesize(["Hello there."])  # doctest: +SKIP
        >>> engine.close()                                # doctest: +SKIP

    Important:
        构造这个对象会**加载模型权重**（数 GB）。请确保这是你有意为之：
        跑批量任务时全局只建一个实例；跑测试时请用
        :class:`FakeSynthesisEngine`。
    """

    def __init__(
        self,
        generation: GenerationConfig,
        voice: VoiceConfig,
        *,
        device: str | None = None,
    ) -> None:
        self.generation = generation
        self.voice = voice

        # 交给 accelerator 统一决定设备/精度/注意力/加速开关。
        # 它在**加载模型之前**就会把配置写错的情况（例如要求 flash-attn 但没装）
        # 以清晰的日志暴露出来，而不是等权重加载完再崩。
        self.profile = accelerator.preflight(generation, device=device)
        self.device = self.profile.device
        self.attn_implementation = self.profile.attn_implementation
        self.sample_rate: int = 24_000

        self._model: Any = None
        self._prompt: Any = None
        self._prompt_seconds: float = 0.0
        self._closed = False

        self._load_model()
        self._build_prompt()

    # ------------------------------------------------------------------ 加载
    def _load_model(self) -> None:
        """加载模型权重（只做一次）。"""
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.generation.dtype]

        model_path = self.generation.model_path
        logger.info(
            "加载模型 %s（device=%s, dtype=%s, attn=%s）…",
            model_path,
            self.device,
            self.generation.dtype,
            self.attn_implementation,
        )
        started = time.time()

        self._model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=dtype,
            attn_implementation=self.attn_implementation,
        )
        logger.info("模型加载完成，用时 %.1fs", time.time() - started)

        # 记录真实采样率，避免硬编码与模型不一致
        self.sample_rate = int(
            getattr(self._model.model, "sample_rate", None)
            or getattr(self._model.model, "speech_tokenizer_sample_rate", None)
            or 24_000
        )

    def _build_prompt(self) -> None:
        """构建（并缓存）音色克隆 prompt。"""
        ref_audio = self.voice.ref_audio
        if not ref_audio:
            raise ValueError(
                "voice.ref_audio 为空。Base 模型必须提供参考音频，"
                "请在 config.yaml 的 voice.ref_audio 里填写路径。"
            )
        if not Path(ref_audio).is_file():
            raise FileNotFoundError(f"参考音频不存在：{ref_audio}")

        xvec_only = self.voice.x_vector_only_mode or self.voice.mode == "xvec"
        if not xvec_only and not self.voice.ref_text.strip():
            logger.warning(
                "voice.ref_text 为空。ICL 模式需要参考音频的文本才能取得最佳相似度；"
                "将自动退回 x_vector_only 模式（只用说话人向量，相似度略低）。"
                "建议在 config.yaml 里补上 ref_text。"
            )
            xvec_only = True

        logger.info(
            "构建音色 prompt（参考音频=%s, 模式=%s）…",
            ref_audio,
            "xvec" if xvec_only else "icl",
        )
        started = time.time()
        self._prompt = self._model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=None if xvec_only else self.voice.ref_text,
            x_vector_only_mode=xvec_only,
        )
        self._prompt_seconds = time.time() - started
        logger.info("音色 prompt 构建完成，用时 %.1fs", self._prompt_seconds)

    # ------------------------------------------------------------------ 推理
    def synthesize(
        self, texts: list[str], *, language: str | None = None
    ) -> SynthesisResult:
        """合成一批文本。

        Args:
            texts: 待合成文本列表。**保持短小**（建议 < 200 字符），
                这也正是切分模块存在的意义。
            language: 语言覆盖；``None`` 时用配置里的值。

        Returns:
            :class:`SynthesisResult`。

        Raises:
            RuntimeError: 引擎已关闭。
            ValueError: ``texts`` 为空。
        """
        if self._closed:
            raise RuntimeError("引擎已关闭，不能再合成")
        if not texts:
            raise ValueError("texts 不能为空")

        cleaned = [text.strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("texts 中存在空字符串")

        lang = language or self.generation.language or "Auto"
        languages = [lang] * len(cleaned)

        kwargs: dict[str, Any] = {"max_new_tokens": self.generation.max_new_tokens}
        for name in ("temperature", "top_k", "top_p", "repetition_penalty"):
            value = getattr(self.generation, name, None)
            if value is not None:
                kwargs[name] = value

        started = time.time()
        waveforms, sample_rate = self._model.generate_voice_clone(
            text=cleaned,
            language=languages,
            voice_clone_prompt=self._prompt,
            non_streaming_mode=self.generation.non_streaming_mode,
            **kwargs,
        )
        elapsed = time.time() - started

        if sample_rate:
            self.sample_rate = int(sample_rate)

        _empty_device_cache(self.device)

        return SynthesisResult(
            waveforms=list(waveforms),
            sample_rate=self.sample_rate,
            prompt_seconds=0.0,
            generate_seconds=elapsed,
            texts=cleaned,
        )

    # ------------------------------------------------------------------ 收尾
    def close(self) -> None:
        """释放模型与 prompt，归还内存。"""
        if self._closed:
            return
        self._prompt = None
        self._model = None
        self._closed = True
        _empty_device_cache(self.device)

        # 主动 gc：大对象引用断掉后立刻回收，别等下一轮跑批
        import gc

        gc.collect()
        logger.debug("引擎已关闭，内存已释放")

    def __enter__(self) -> "QwenTTSEngine":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ 信息
    def describe(self) -> str:
        """返回一行引擎摘要，便于日志记录。"""
        mode = "xvec" if (self.voice.x_vector_only_mode or self.voice.mode == "xvec") else "icl"
        parts = [
            f"model={self.generation.model_path}",
            f"device={self.device}",
            f"dtype={self.generation.dtype}",
            f"attn={self.attn_implementation}",
            f"batch={self.profile.batch_size}",
            f"voice_mode={mode}",
            f"sample_rate={self.sample_rate}",
        ]
        if self.profile.info.is_cuda:
            parts.append(f"vram={self.profile.info.total_vram_gb:.0f}G")
            if self.profile.allow_tf32:
                parts.append("tf32=on")
        return f"QwenTTSEngine({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# 测试用假引擎
# --------------------------------------------------------------------------- #


class FakeSynthesisEngine:
    """不加载模型、直接产出可听正弦波的假引擎。

    存在的意义：让整条流水线（切分 -> 生成 -> 拼接 -> 压缩 -> 断点续跑）
    可以在**完全不碰模型**的情况下被端到端验证。它满足
    :class:`SynthesisEngine` 协议，因此可以直接替换 :class:`QwenTTSEngine`。

    Args:
        sample_rate: 输出采样率。
        ms_per_char: 每个字符对应多少毫秒（用来模拟语速）。
        tone_hz: 正弦波频率。
        fail_on: 文本包含这些子串时故意抛异常，用于测试重试与容错。

    Example:
        >>> engine = FakeSynthesisEngine()
        >>> result = engine.synthesize(["Hello there."])
        >>> len(result.waveforms)
        1
    """

    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        ms_per_char: float = 45.0,
        tone_hz: float = 220.0,
        fail_on: tuple[str, ...] = (),
        latency: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.tone_hz = tone_hz
        self.fail_on = fail_on
        self.latency = latency
        self.calls: list[list[str]] = []
        self.closed = False

    def synthesize(
        self, texts: list[str], *, language: str | None = None
    ) -> SynthesisResult:
        """生成一段与文本长度成正比的音频（内容不重要，长度才重要）。"""
        import numpy as np

        if self.closed:
            raise RuntimeError("引擎已关闭")
        self.calls.append(list(texts))

        if self.latency:
            time.sleep(self.latency)

        waveforms = []
        for text in texts:
            if any(token in text for token in self.fail_on):
                raise RuntimeError(f"FakeSynthesisEngine 故意失败：{text[:30]!r}")
            seconds = max(0.25, len(text) * self.ms_per_char / 1000.0)
            count = int(self.sample_rate * seconds)
            t = np.arange(count, dtype=np.float32) / self.sample_rate
            wave = 0.25 * np.sin(2 * np.pi * self.tone_hz * t)
            # 首尾各 10ms 淡入淡出，模拟真实音频的包络
            edge = int(self.sample_rate * 0.01)
            if count > 2 * edge:
                wave[:edge] *= np.linspace(0, 1, edge, dtype=np.float32)
                wave[-edge:] *= np.linspace(1, 0, edge, dtype=np.float32)
            waveforms.append(wave)

        return SynthesisResult(
            waveforms=waveforms,
            sample_rate=self.sample_rate,
            generate_seconds=0.0,
            texts=list(texts),
        )

    def close(self) -> None:
        self.closed = True

    def describe(self) -> str:
        return f"FakeSynthesisEngine(sample_rate={self.sample_rate})"

    def __enter__(self) -> "FakeSynthesisEngine":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_engine(
    generation: GenerationConfig,
    voice: VoiceConfig,
    *,
    device: str | None = None,
    dry_run: bool = False,
    **fake_kwargs: Any,
) -> SynthesisEngine:
    """按配置构造引擎。

    Args:
        generation: 生成参数。
        voice: 音色配置。
        device: 覆盖设备。
        dry_run: ``True`` 时返回 :class:`FakeSynthesisEngine`（**不加载模型**），
            用于在正式跑批前演练整条流水线。
        **fake_kwargs: 传给 :class:`FakeSynthesisEngine` 的额外参数。

    Returns:
        满足 :class:`SynthesisEngine` 协议的引擎实例。
    """
    if dry_run:
        logger.warning("dry_run 已启用：使用假引擎，不会加载任何模型")
        return FakeSynthesisEngine(**fake_kwargs)
    return QwenTTSEngine(generation, voice, device=device)
