"""音频压缩：把无损章节 wav 转成手机/播放器友好的 m4a / mp3。

为什么需要压缩
--------------

PCM_16 单声道 24 kHz 大约是 **48 KB/s ≈ 173 MB/小时**，整本有声书 4 小时
就是 700 MB。AAC/MP3 在 48~64 kbps 下听感几乎无损（语音内容尤其如此），
体积能降到 **1/8~1/10**。

策略
----

* 优先用 **ffmpeg**（跨平台、参数可控）；
* macOS 上没有 ffmpeg 时退回系统自带的 **afconvert**（能出 m4a/aac）；
* 两者都没有时给出明确的安装指引，而不是抛一个看不懂的异常。

命令构造采用 ``subprocess`` 列表形式（不做 shell 拼接），避免文件名里的
空格/引号出问题。
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, EncodeConfig
from .utils import format_duration, get_logger, human_size

logger = get_logger("encode")

#: 各封装格式对应的 ffmpeg 编码参数。
_FORMAT_ARGS: dict[str, list[str]] = {
    "m4a": ["-c:a", "aac", "-profile:a", "aac_low", "-movflags", "+faststart"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "4"],
    "opus": ["-c:a", "libopus", "-application", "voip", "-vbr", "on"],
    "flac": ["-c:a", "flac", "-compression_level", "8"],
    "copy": ["-c:a", "copy"],
}

#: 各格式的默认扩展名。
_FORMAT_SUFFIX: dict[str, str] = {
    "m4a": ".m4a",
    "mp3": ".mp3",
    "opus": ".opus",
    "flac": ".flac",
    "copy": ".m4a",
}

#: ffmpeg 里表示「无损/忽略码率」的取值。
_LOSSLESS_BITRATES = {"", "none", "lossless", "0"}


@dataclass
class EncodeResult:
    """一次压缩的结果。"""

    source: Path
    output: Path
    source_bytes: int
    output_bytes: int
    duration: float = 0.0
    elapsed_seconds: float = 0.0
    encoder: str = "ffmpeg"

    @property
    def ratio(self) -> float:
        """压缩比（越小越好）。"""
        return self.output_bytes / self.source_bytes if self.source_bytes else 0.0

    @property
    def saved_bytes(self) -> int:
        return max(0, self.source_bytes - self.output_bytes)

    def summary(self) -> str:
        return (
            f"{self.output.name} | {human_size(self.source_bytes)} -> "
            f"{human_size(self.output_bytes)}（{self.ratio * 100:.0f}%，"
            f"省 {human_size(self.saved_bytes)}）"
            + (f" | 耗时 {format_duration(self.elapsed_seconds)}" if self.elapsed_seconds else "")
        )


# --------------------------------------------------------------------------- #
# 工具探测
# --------------------------------------------------------------------------- #


def find_ffmpeg(binary: str = "ffmpeg") -> str | None:
    """查找 ffmpeg 可执行文件。

    Args:
        binary: 可执行文件名或绝对路径。

    Returns:
        可用路径，或 ``None``。
    """
    found = shutil.which(binary)
    if found:
        return found
    # 常见但不一定在 PATH 里的位置
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if Path(candidate).is_file():
            return candidate
    return None


def find_afconvert() -> str | None:
    """查找 macOS 自带的 afconvert。"""
    return shutil.which("afconvert")


def probe_duration(path: str | Path) -> float:
    """用 ffprobe 读取音频时长；失败返回 0。

    仅用于报告，不参与流程判断。
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def encoder_available(config: EncodeConfig | None = None) -> tuple[bool, str]:
    """检查是否有可用的编码器。

    Returns:
        ``(是否可用, 说明)``。
    """
    config = config or EncodeConfig()
    if find_ffmpeg(config.ffmpeg_bin):
        return True, "ffmpeg"
    if config.format in {"m4a", "flac"} and find_afconvert():
        return True, "afconvert"
    return False, (
        "没有找到 ffmpeg，也没有可用的 afconvert。"
        "macOS 可执行 `brew install ffmpeg`；"
        "或把 encode.format 设为 m4a 以使用系统自带的 afconvert。"
    )


# --------------------------------------------------------------------------- #
# 编码
# --------------------------------------------------------------------------- #


def build_ffmpeg_command(
    source: str | Path,
    output: str | Path,
    config: EncodeConfig,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """构造 ffmpeg 命令（列表形式，便于测试与安全调用）。

    Args:
        source: 输入音频。
        output: 输出音频。
        config: 编码配置。
        ffmpeg_bin: ffmpeg 可执行文件。

    Returns:
        可直接交给 :func:`subprocess.run` 的参数列表。
    """
    fmt = config.format if config.format in _FORMAT_ARGS else "m4a"
    command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    command += _FORMAT_ARGS[fmt]

    if config.bitrate and config.bitrate.lower() not in _LOSSLESS_BITRATES and fmt != "flac":
        command += ["-b:a", config.bitrate]

    if config.sample_rate and fmt != "copy":
        command += ["-ar", str(config.sample_rate)]
    if config.channels and fmt != "copy":
        command += ["-ac", str(config.channels)]

    command += list(config.extra_args)
    command.append(str(output))
    return command


def build_afconvert_command(
    source: str | Path,
    output: str | Path,
    config: EncodeConfig,
    *,
    afconvert_bin: str = "afconvert",
) -> list[str]:
    """构造 macOS ``afconvert`` 命令（仅支持 m4a/aac 与 alac）。"""
    command = [
        afconvert_bin,
        str(source),
        str(output),
        "-f", "m4af",
        "-d", "aac",
        "-b", str(_bitrate_to_bps(config.bitrate)),
    ]
    if config.channels:
        command += ["-c", str(config.channels)]
    return command


def _bitrate_to_bps(bitrate: str) -> int:
    """把 ``"64k"`` 转成 ``64000``。"""
    value = (bitrate or "64k").strip().lower()
    try:
        if value.endswith("k"):
            return int(float(value[:-1]) * 1000)
        if value.endswith("m"):
            return int(float(value[:-1]) * 1_000_000)
        return int(float(value))
    except ValueError:
        return 64_000


def encode_audio(
    source: str | Path,
    output: str | Path | None = None,
    *,
    config: AppConfig | None = None,
    encode_config: EncodeConfig | None = None,
) -> EncodeResult:
    """压缩单个音频文件。

    Args:
        source: 输入 wav。
        output: 输出路径；``None`` 时按 ``encode.format`` 推断扩展名。
        config: 全局配置（取 ``encode`` 段）。
        encode_config: 直接指定编码配置（优先于 ``config``）。

    Returns:
        :class:`EncodeResult`。

    Raises:
        FileNotFoundError: 输入文件不存在。
        RuntimeError: 没有可用编码器，或编码失败。
    """
    encode_config = encode_config or (config.encode if config else EncodeConfig())
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到输入音频：{source_path}")

    fmt = encode_config.format if encode_config.format in _FORMAT_SUFFIX else "m4a"
    suffix = _FORMAT_SUFFIX[fmt]
    target = Path(output) if output is not None else source_path.with_suffix(suffix)
    target.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg(encode_config.ffmpeg_bin)
    afconvert = find_afconvert() if fmt in {"m4a", "flac"} else None

    started = time.time()
    if ffmpeg:
        command = build_ffmpeg_command(source_path, target, encode_config, ffmpeg_bin=ffmpeg)
        encoder = "ffmpeg"
    elif afconvert:
        if fmt == "flac":
            raise RuntimeError("afconvert 不支持 flac，请安装 ffmpeg")
        logger.info("未找到 ffmpeg，改用 macOS 自带的 afconvert")
        command = build_afconvert_command(source_path, target, encode_config, afconvert_bin=afconvert)
        encoder = "afconvert"
    else:
        raise RuntimeError(
            "没有可用的编码器。请安装 ffmpeg（macOS: brew install ffmpeg），"
            "或把 encode.format 设为 m4a 以使用系统自带 afconvert。"
        )

    logger.info("压缩 %s -> %s（%s）", source_path.name, target.name, encoder)
    logger.debug("命令：%s", " ".join(command))

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"编码超时（>1 小时）：{source_path}") from exc
    except OSError as exc:
        raise RuntimeError(f"无法执行编码器 {command[0]}：{exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"编码失败（退出码 {result.returncode}）：{stderr[:500] or '无错误输出'}"
        )
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"编码器没有产出有效文件：{target}")

    return EncodeResult(
        source=source_path,
        output=target,
        source_bytes=source_path.stat().st_size,
        output_bytes=target.stat().st_size,
        duration=probe_duration(target),
        elapsed_seconds=time.time() - started,
        encoder=encoder,
    )


def encode_chapters(
    config: AppConfig,
    chapter_ids: list[str] | None = None,
    *,
    keep_wav: bool | None = None,
) -> list[EncodeResult]:
    """批量压缩已拼接好的章节 wav。

    Args:
        config: 全局配置。
        chapter_ids: 章节列表；``None`` 表示自动发现 ``output/audio/*.wav``。
        keep_wav: 是否保留源 wav；``None`` 时用 ``encode.keep_wav``。

    Returns:
        成功的 :class:`EncodeResult` 列表（失败会记录日志并跳过）。
    """
    if not config.encode.enabled:
        logger.info("encode.enabled 为 false，跳过压缩")
        return []

    available, reason = encoder_available(config.encode)
    if not available:
        logger.error("压缩不可用：%s", reason)
        return []

    keep = config.encode.keep_wav if keep_wav is None else keep_wav

    if chapter_ids is None:
        sources = sorted(config.project.output_path.glob("*.wav"))
    else:
        sources = [config.project.output_path / f"{cid}.wav" for cid in chapter_ids]

    results: list[EncodeResult] = []
    for source in sources:
        if not source.is_file():
            logger.warning("找不到 %s，跳过", source)
            continue
        try:
            result = encode_audio(source, config=config)
        except RuntimeError as exc:
            logger.error("压缩 %s 失败：%s", source.name, exc)
            continue

        results.append(result)
        logger.info("  %s", result.summary())

        if not keep:
            source.unlink(missing_ok=True)
            logger.debug("已删除源文件 %s", source)

    if results:
        total_in = sum(r.source_bytes for r in results)
        total_out = sum(r.output_bytes for r in results)
        logger.info(
            "压缩完成：%d 个文件，%s -> %s（省 %s）",
            len(results),
            human_size(total_in),
            human_size(total_out),
            human_size(max(0, total_in - total_out)),
        )
    return results
