"""音频读写工具：全部走 streaming，任何一步都不会把整本书读进内存。

内存预算（16 GB Mac 的关键约束）
--------------------------------

一段 120 字符的旁白合成出来大约 8~12 秒，24 kHz 单声道 float32 是
约 1.2 MB；一本书 3200 段如果全留在内存里就是 ~4 GB，再加上模型
（bf16 约 3.4 GB）和 Python 开销，会直接把机器拖进 swap。

因此本模块的每个函数都遵守一条规则：**只处理当前需要的那一小段**。

* :func:`write_audio` 直接写盘，不返回大数组；
* :func:`iter_audio_blocks` 按固定块大小读文件（默认 1 秒一块）；
* :func:`read_audio` 只在片段级别使用（单段最多几十秒，安全）；
* 拼接与压缩通过块迭代完成，峰值内存是 ``块大小 × 声道数`` 量级。

响度归一化
----------

模型逐段生成的音量会有波动（同一句话可能差 3~6 dB），直接拼起来会
忽大忽小。这里提供两种模式：

* ``peak``：把峰值抬到目标 dBFS。简单、稳定、绝不削波，是默认值；
* ``rms``：把整体 RMS 抬到目标值，段间响度更一致，但对短促的
  爆破音（"Hey!"）可能放大过头，所以用 ``max_gain_db`` 兜住。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import AUDIO_DTYPE
from .config import AudioConfig
from .utils import get_logger

logger = get_logger("audio")

#: 分块读取的默认块时长（秒）。1 秒 @24 kHz 单声道 float32 只有 96 KB。
DEFAULT_BLOCK_SECONDS = 1.0

#: float 采样值的理论上限。
_FULL_SCALE = 1.0


# --------------------------------------------------------------------------- #
# 底层依赖（延迟导入，保证 import noveltts 不需要装 numpy/soundfile）
# --------------------------------------------------------------------------- #


def _np() -> "object":
    """延迟导入 numpy。"""
    import numpy as np

    return np


def _sf() -> "object":
    """延迟导入 soundfile。"""
    import soundfile as sf

    return sf


def audio_available() -> tuple[bool, str]:
    """检查音频读写依赖是否可用。

    Returns:
        ``(是否可用, 说明)``。不可用时说明里会写清缺哪个包。
    """
    missing: list[str] = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import soundfile  # noqa: F401
    except ImportError:
        missing.append("soundfile")
    if missing:
        return False, "缺少依赖：" + "、".join(missing) + "（pip install numpy soundfile）"
    return True, "ok"


# --------------------------------------------------------------------------- #
# 响度工具
# --------------------------------------------------------------------------- #


def db_to_linear(db: float) -> float:
    """分贝转线性增益。"""
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    """线性增益转分贝（``value <= 0`` 时返回 ``-inf``）。"""
    return 20.0 * math.log10(value) if value > 0 else float("-inf")


def peak_dbfs(samples: "object") -> float:
    """返回数组的峰值（dBFS）。空数组返回 ``-inf``。"""
    np = _np()
    if len(samples) == 0:  # type: ignore[arg-type]
        return float("-inf")
    peak = float(np.max(np.abs(samples)))
    return linear_to_db(peak)


def rms_dbfs(samples: "object") -> float:
    """返回数组的 RMS（dBFS）。空数组返回 ``-inf``。"""
    np = _np()
    if len(samples) == 0:  # type: ignore[arg-type]
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return linear_to_db(rms)


def normalize_audio(samples: "object", config: AudioConfig) -> "object":
    """按配置对一小段音频做响度归一化（原地返回新数组）。

    Args:
        samples: float32 单声道/多声道数组（shape ``(n,)`` 或 ``(n, ch)``）。
        config: 音频配置。

    Returns:
        归一化后的数组。``normalize == "none"`` 时原样返回。
    """
    np = _np()
    if config.normalize == "none" or len(samples) == 0:  # type: ignore[arg-type]
        return samples

    if config.normalize == "rms":
        current = rms_dbfs(samples)
        target = config.target_rms_dbfs
    else:
        current = peak_dbfs(samples)
        target = config.target_dbfs

    if not math.isfinite(current):
        return samples

    # 整段是数字静音（或极低电平）时直接放弃：放大纯静音只会把量化噪声抬起来，
    # 没有任何意义，而且增益会冲向无穷大。
    if float(np.max(np.abs(samples))) <= 1e-7:
        return samples

    gain_db = target - current
    # 只允许放大到 max_gain_db，且不允许把峰值推过 1.0
    gain_db = min(gain_db, config.max_gain_db)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        headroom_db = linear_to_db(_FULL_SCALE / peak)
        gain_db = min(gain_db, headroom_db)

    if abs(gain_db) < 0.05:  # 变化太小就别动，省一次乘法
        return samples
    return (samples * db_to_linear(gain_db)).astype(np.float32, copy=False)


def trim_silence(
    samples: "object",
    *,
    threshold_dbfs: float = -50.0,
    sample_rate: int = 24_000,
    keep_ms: float = 20.0,
) -> "object":
    """裁掉首尾的静音，但保留 ``keep_ms`` 毫秒作为缓冲。

    片段开头/结尾的静音在拼接时会造成忽长忽短的停顿，裁掉后由
    :func:`join_gap_samples` 统一插入间隔，节奏更均匀。

    Args:
        samples: 单声道或多声道数组。
        threshold_dbfs: 静音门限。
        sample_rate: 采样率。
        keep_ms: 裁切后保留的缓冲时长。

    Returns:
        裁切后的数组；若整段都在门限以下则原样返回（避免产出空数组）。
    """
    np = _np()
    if len(samples) == 0:  # type: ignore[arg-type]
        return samples

    mono = samples if samples.ndim == 1 else np.max(np.abs(samples), axis=1)  # type: ignore[attr-defined]
    threshold = db_to_linear(threshold_dbfs)
    above = np.flatnonzero(np.abs(mono) > threshold)
    if len(above) == 0:
        return samples

    keep = int(sample_rate * keep_ms / 1000.0)
    start = max(0, int(above[0]) - keep)
    end = min(len(samples), int(above[-1]) + 1 + keep)  # type: ignore[arg-type]
    if start == 0 and end == len(samples):  # type: ignore[arg-type]
        return samples
    return samples[start:end]


# --------------------------------------------------------------------------- #
# 语速调节（变速不变调）
# --------------------------------------------------------------------------- #


def time_stretch(
    samples: "object",
    *,
    rate: float,
    sample_rate: int = 24_000,
) -> "object":
    """改变语速但**保持音高不变**。

    为什么不用简单重采样
    --------------------

    直接改采样率（``samples[::2]`` 之类）会连带把音高一起抬高，
    听感像是"快进的唐老鸭"。这里用的是**相位声码器（phase vocoder）**，
    它把时间轴拉伸与音高解耦：

    * ``rate > 1`` -> 说得更快，时长变短（``1.25`` 表示快 25%）
    * ``rate < 1`` -> 说得更慢，时长变长
    * ``rate == 1`` -> 原样返回（不做任何计算）

    实现来源：``librosa.effects.time_stretch``。librosa 已经是
    ``qwen_tts`` 的依赖，所以**不引入任何新依赖**；万一环境里没有
    librosa，会自动退回"不处理"并给出警告，而不是让整个跑批失败。

    Args:
        samples: 单声道或多声道 float 数组。
        rate: 速度倍率，必须 > 0。建议范围 ``0.5 ~ 2.0``，
            超出后相位声码器的瑕疵会明显。
        sample_rate: 采样率（仅用于日志与校验）。

    Returns:
        处理后的数组；``rate == 1`` 或 librosa 不可用时原样返回。

    Note:
        效果最好的是微调（0.9~1.15）。差别越大，金属感/回声感越明显——
        这是相位声码器的固有代价，不是 bug。
    """
    if rate <= 0:
        raise ValueError(f"speech_rate 必须 > 0，收到 {rate}")
    if abs(rate - 1.0) < 1e-6:
        return samples
    if len(samples) == 0:  # type: ignore[arg-type]
        return samples

    try:
        import warnings

        from librosa.effects import time_stretch as _librosa_stretch
    except ImportError as exc:
        logger.warning(
            "无法调节语速：librosa 不可用（%s）。将按原速输出。", exc
        )
        return samples

    try:
        # librosa 1.0 会为已废弃参数发 FutureWarning，这里静音掉免得刷屏
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stretched = _librosa_stretch(samples, rate=float(rate))
    except Exception as exc:  # noqa: BLE001 - 变速失败不应让整段音频报废
        logger.warning("语速调节失败（%s），按原速输出：%s", rate, exc)
        return samples

    np = _np()
    result = np.asarray(stretched, dtype=np.float32)
    logger.debug(
        "语速 %.3f：%.2fs -> %.2fs", rate, len(samples) / sample_rate, len(result) / sample_rate
    )
    return result


def speed_factor_to_rate(value: float) -> float:
    """把可读性更好的写法换算成 ``time_stretch`` 的 rate。

    约定（与 ffmpeg 的 ``atempo`` / 播放器习惯一致）：

    * ``value > 1`` -> **说得更快**（``1.25`` = 快 25%），rate 直接等于它
    * ``value <= 1`` -> **说得更慢**（``0.8`` = 慢到 80%），rate 同样等于它

    也就是说这个函数目前是"透传 + 校验"，存在的意义是**把语义写清楚**，
    避免以后有人误以为 ``0.8`` 表示"快 0.8 倍"。

    Args:
        value: 速度倍率。

    Returns:
        合法的 rate（> 0）。

    Raises:
        ValueError: 非正数。
    """
    if value <= 0:
        raise ValueError(f"速度倍率必须 > 0，收到 {value}")
    return float(value)


def tail_is_quiet(
    samples: "object",
    *,
    tail_ms: float = 120.0,
    threshold_dbfs: float = -38.0,
    sample_rate: int = 24_000,
) -> bool:
    """判断一段音频的结尾是否已经自带静音。

    用于拼接时决定要塞多长的间隔：如果模型自己已经在句尾留了停顿，
    再插一个完整间隔就会听起来「断得太开」。

    Args:
        samples: 音频数组。
        tail_ms: 考察结尾多少毫秒。
        threshold_dbfs: 静音判定门限。
        sample_rate: 采样率。

    Returns:
        ``True`` 表示结尾已经足够安静。
    """
    np = _np()
    length = len(samples)  # type: ignore[arg-type]
    if length == 0:
        return True
    window = min(length, max(1, int(sample_rate * tail_ms / 1000.0)))
    tail = samples[length - window:]

    # 数字静音（全 0）的 RMS 是 -inf，而 -inf 不是有限数。
    # 必须单独判掉，否则「整段都是静音」会被判成「不安静」——这是实际踩过的坑。
    if float(np.max(np.abs(tail))) <= 1e-7:
        return True

    level = rms_dbfs(tail)
    return math.isfinite(level) and level <= threshold_dbfs


def join_gap_samples(
    config: AudioConfig,
    *,
    scale: float = 1.0,
) -> "object":
    """生成片段之间的静音间隔。

    Args:
        config: 音频配置（提供 ``min_gap_seconds`` 与 ``sample_rate``）。
        scale: 缩放系数；调用方用 ``config.tail_quiet_gain`` 缩短停顿。

    Returns:
        形状 ``(n,)`` 的 float32 全零数组。
    """
    np = _np()
    seconds = max(0.0, config.min_gap_seconds * scale)
    count = int(config.sample_rate * seconds)
    return np.zeros(count, dtype=np.float32)


def apply_fade(
    samples: "object",
    *,
    fade_in_samples: int = 0,
    fade_out_samples: int = 0,
) -> "object":
    """对音频首尾施加线性淡入淡出，消除拼接处的咔嗒声。

    Args:
        samples: 音频数组（原地不改，返回副本）。
        fade_in_samples: 开头淡入的采样点数。
        fade_out_samples: 结尾淡出的采样点数。

    Returns:
        处理后的新数组。
    """
    np = _np()
    if len(samples) == 0:  # type: ignore[arg-type]
        return samples
    out = np.array(samples, dtype=np.float32, copy=True)

    if fade_in_samples > 0:
        count = min(fade_in_samples, len(out))
        out[:count] *= np.linspace(0.0, 1.0, count, dtype=np.float32)
    if fade_out_samples > 0:
        count = min(fade_out_samples, len(out))
        out[len(out) - count:] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
    return out


# --------------------------------------------------------------------------- #
# 文件读写
# --------------------------------------------------------------------------- #


@dataclass
class AudioInfo:
    """一个音频文件的基本信息。"""

    path: Path
    sample_rate: int
    channels: int
    frames: int
    duration: float
    subtype: str = ""

    @property
    def is_mono(self) -> bool:
        return self.channels == 1


def probe_audio(path: str | Path) -> AudioInfo:
    """读取音频文件头信息（不解码全部数据，非常快）。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    sf = _sf()
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"找不到音频文件：{target}")
    info = sf.info(str(target))
    return AudioInfo(
        path=target,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration=float(info.duration),
        subtype=str(getattr(info, "subtype", "") or ""),
    )


def read_audio(
    path: str | Path,
    *,
    start: int = 0,
    frames: int = -1,
) -> tuple["object", int]:
    """读取（一段）音频。

    Args:
        path: 音频文件路径。
        start: 起始采样点。
        frames: 读取长度；``-1`` 表示读到结尾。

    Returns:
        ``(samples, sample_rate)``，``samples`` 是 float32 数组，
        单声道为 ``(n,)``，多声道为 ``(n, ch)``。
    """
    sf = _sf()
    samples, sample_rate = sf.read(
        str(path), start=start, frames=frames, dtype="float32", always_2d=False
    )
    return samples, int(sample_rate)


def write_audio(
    path: str | Path,
    samples: "object",
    sample_rate: int,
    *,
    subtype: str = AUDIO_DTYPE,
) -> Path:
    """把音频写到文件（自动建目录）。

    Args:
        path: 目标路径。
        samples: float 数组，``(n,)`` 或 ``(n, ch)``。
        sample_rate: 采样率。
        subtype: 位深，默认 ``PCM_16``（比 float32 省一半磁盘，听感无损）。

    Returns:
        写入的路径。
    """
    sf = _sf()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), samples, sample_rate, subtype=subtype)
    return target


def iter_audio_blocks(
    path: str | Path,
    *,
    block_frames: int = 0,
    block_seconds: float = DEFAULT_BLOCK_SECONDS,
) -> Iterator["object"]:
    """按块迭代读取音频，用于低内存拼接。

    Args:
        path: 音频文件。
        block_frames: 每块采样点数；``0`` 表示用 ``block_seconds`` 计算。
        block_seconds: 每块时长（秒）。

    Yields:
        float32 数组，形状与源文件声道数一致。
    """
    sf = _sf()
    with sf.SoundFile(str(path)) as handle:
        if block_frames <= 0:
            block_frames = max(1024, int(handle.samplerate * block_seconds))
        while True:
            block = handle.read(block_frames, dtype="float32", always_2d=False)
            if block is None or len(block) == 0:
                break
            yield block


def open_writer(
    path: str | Path,
    *,
    sample_rate: int,
    channels: int = 1,
    subtype: str = AUDIO_DTYPE,
) -> "object":
    """打开一个流式写入器（``soundfile.SoundFile``）。

    调用方负责 ``close()``；推荐用 ``with`` 包裹。配合
    :func:`iter_audio_blocks` 就能做到「边读边写、内存恒定」。

    Example:
        >>> with open_writer("out.wav", sample_rate=24000) as writer:  # doctest: +SKIP
        ...     for block in iter_audio_blocks("in.wav"):
        ...         writer.write(block)
    """
    sf = _sf()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return sf.SoundFile(
        str(target),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype=subtype,
    )


def concat_audio_files(
    inputs: list[str | Path],
    output: str | Path,
    *,
    sample_rate: int,
    channels: int = 1,
    gaps: list["object"] | None = None,
    lead_silence: "object | None" = None,
    tail_silence: "object | None" = None,
    block_seconds: float = DEFAULT_BLOCK_SECONDS,
) -> Path:
    """把多个音频文件顺序拼成一个文件（流式，内存恒定）。

    Args:
        inputs: 输入文件列表（必须都是 ``sample_rate`` / ``channels``）。
        output: 输出文件。
        sample_rate: 采样率。
        channels: 声道数。
        gaps: 与 ``inputs`` 等长的静音数组列表，``gaps[i]`` 插在第 ``i`` 段之后；
            ``None`` 表示不插。
        lead_silence: 开头插入的静音。
        tail_silence: 结尾插入的静音。
        block_seconds: 读取块时长。

    Returns:
        输出文件路径。

    Note:
        不做重采样。声道/采样率不一致会在写入时直接报错，比悄悄产出
        半损坏的文件更容易排查。
    """
    sf = _sf()
    np = _np()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(
        str(out_path), mode="w", samplerate=sample_rate, channels=channels, subtype=AUDIO_DTYPE
    ) as writer:
        if lead_silence is not None and len(lead_silence):
            writer.write(lead_silence)

        for index, source in enumerate(inputs):
            for block in iter_audio_blocks(source, block_seconds=block_seconds):
                if block.ndim == 1 and channels > 1:
                    block = np.repeat(block[:, None], channels, axis=1)
                writer.write(block)

            if gaps is not None and index < len(gaps) and len(gaps[index]):
                writer.write(gaps[index])

        if tail_silence is not None and len(tail_silence):
            writer.write(tail_silence)

    return out_path
