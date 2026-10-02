"""音频拼接：把逐段生成的片段合成一整章/整本有声书。

内存策略（16 GB Mac）
--------------------

拼接是最容易「一次性吃光内存」的环节，因为直觉写法是
``np.concatenate(所有片段)``。一章 500 段、每段 10 秒的话，那是
500 × 10 × 24000 × 4 B ≈ 480 MB 的 float32，整本 3200 段就是 3 GB 以上。

因此这里**完全不拼接数组**，而是：

1. 用 :func:`probe_audio` 读文件头拿到每段时长（只读元数据，不解码）；
2. 打开输出文件，按块迭代读取每个片段并直接写入；
3. 片段之间写入计算好的静音。

峰值内存 = 单个读取块（默认 1 秒 ≈ 96 KB）+ 一个静音数组。与章节长度无关。

停顿处理
--------

机械地每段之间插固定静音会很难听。这里做两件事：

* **尾部静音感知**：模型经常已经在句尾留了 0.3~0.5 秒的静音。如果检测到
  片段结尾本来就安静，就把间隔缩短为 ``tail_quiet_gain`` 倍，
  避免出现「断得太开」的怪异停顿；
* **淡入淡出**：片段接缝处做 ``join_fade_ms`` 的线性淡变，消除波形不连续
  造成的「咔嗒」声。

若某个片段**不以句末标点结束**（例如被切断的台词），间隔会进一步缩短，
让语流更连贯。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .audio_io import (
    AudioInfo,
    audio_available,
    iter_audio_blocks,
    open_writer,
    probe_audio,
    read_audio,
    tail_is_quiet,
)
from .config import AppConfig
from .models import ChapterAudioManifest, Segment, SegmentRecord
from .utils import ensure_dir, format_duration, get_logger, human_size

logger = get_logger("concat")

#: 判断片段是否以句末标点结束（决定要不要缩短停顿）。
_ENDS_SENTENCE_RE = re.compile(r"[.!?\u2026\u3002\uff01\uff1f][\"'\u201d\u2019\u300d\u300f\)\]\}]*$")

#: 拼接时认为「后面还会继续」的停顿系数。
_CONTINUATION_GAIN = 0.5


@dataclass
class ConcatResult:
    """拼接结果。"""

    output_path: Path
    chapter_id: str
    segment_count: int
    duration: float
    size_bytes: int
    elapsed_seconds: float = 0.0
    missing: list[str] = field(default_factory=list)
    sample_rate: int = 24_000

    def summary(self) -> str:
        parts = [
            f"{self.chapter_id}",
            f"{self.segment_count} 段",
            f"时长 {format_duration(self.duration)}",
            f"体积 {human_size(self.size_bytes)}",
        ]
        if self.elapsed_seconds:
            parts.append(f"耗时 {format_duration(self.elapsed_seconds)}")
        if self.missing:
            parts.append(f"缺失 {len(self.missing)} 段")
        return " | ".join(parts)


def resolve_segment_paths(
    session_paths: list[tuple[Segment, Path]],
) -> tuple[list[tuple[Segment, Path]], list[str]]:
    """过滤掉不存在的片段文件。

    Args:
        session_paths: ``(片段, wav 路径)`` 列表。

    Returns:
        ``(存在的列表, 缺失的 segment_id 列表)``。缺失的片段会在结果里
        报告出来——**绝不静默跳过**，否则你会拿到一章缺了几句话的音频
        却毫不知情。
    """
    present: list[tuple[Segment, Path]] = []
    missing: list[str] = []
    for segment, path in session_paths:
        if Path(path).is_file() and Path(path).stat().st_size > 0:
            present.append((segment, path))
        else:
            missing.append(segment.segment_id)
    return present, missing


def concat_chapter(
    config: AppConfig,
    segments: list[Segment],
    *,
    segment_paths: list[Path] | None = None,
    output_path: str | Path | None = None,
    gap_seconds: float | None = None,
    lead_silence_seconds: float = 0.0,
    tail_silence_seconds: float = 0.6,
    chapter_id: str = "",
    title: str = "",
) -> ConcatResult:
    """把一章的所有片段拼成一个 wav（流式，内存恒定）。

    Args:
        config: 配置（用 ``audio`` 段的采样率、间隔、淡变参数）。
        segments: 片段列表（按播放顺序）。
        segment_paths: 与 ``segments`` 一一对应的 wav 路径；``None`` 时
            按 ``output/audio/segments/<chapter_id>/<segment_id>.wav`` 推断。
        output_path: 输出文件；``None`` 时输出到
            ``output/audio/<chapter_id>.wav``。
        gap_seconds: 覆盖默认段间停顿。
        lead_silence_seconds: 开头静音（留一点起播缓冲）。
        tail_silence_seconds: 结尾静音。
        chapter_id: 章节标识；``None`` 时取第一个片段的。
        title: 章节标题（仅用于日志）。

    Returns:
        :class:`ConcatResult`。

    Raises:
        ValueError: ``segments`` 为空或与 ``segment_paths`` 长度不一致。
        RuntimeError: 所有片段文件都不存在。
    """
    available, reason = audio_available()
    if not available:
        raise RuntimeError(f"音频依赖不可用：{reason}")

    if not segments:
        raise ValueError("segments 为空，没有可拼接的内容")

    chapter_id = chapter_id or segments[0].chapter_id

    if segment_paths is None:
        base = config.project.output_path / "segments" / chapter_id
        segment_paths = [base / f"{segment.segment_id}.wav" for segment in segments]
    if len(segment_paths) != len(segments):
        raise ValueError(
            f"segments({len(segments)}) 与 segment_paths({len(segment_paths)}) 长度不一致"
        )

    present, missing = resolve_segment_paths(list(zip(segments, segment_paths)))
    if not present:
        raise RuntimeError(f"{chapter_id} 没有任何可用的片段文件，请先运行 synth")
    if missing:
        logger.warning(
            "%s 有 %d 个片段缺失，将拼接剩余 %d 段：%s%s",
            chapter_id,
            len(missing),
            len(present),
            ", ".join(missing[:5]),
            " …" if len(missing) > 5 else "",
        )

    audio_config = config.audio
    target = (
        Path(output_path)
        if output_path is not None
        else config.project.output_path / f"{chapter_id}.wav"
    )
    ensure_dir(target.parent)

    started = time.time()

    # --- 1) 只读文件头，拿到采样率/声道与时长（不解码音频） ---
    infos: list[AudioInfo] = [probe_audio(path) for _segment, path in present]
    sample_rate = audio_config.sample_rate or infos[0].sample_rate
    channels = audio_config.channels or infos[0].channels

    mismatched = [
        info for info in infos if info.sample_rate != sample_rate or info.channels != channels
    ]
    if mismatched:
        logger.warning(
            "有 %d 个片段的采样率/声道与目标(%d Hz, %dch)不一致，"
            "将由 soundfile 原样写入（建议检查生成参数）",
            len(mismatched),
            sample_rate,
            channels,
        )

    base_gap = audio_config.min_gap_seconds if gap_seconds is None else gap_seconds
    fade_samples = int(sample_rate * audio_config.join_fade_ms / 1000.0)

    # --- 2) 预先算好每段之后的间隔（只需要读每段的尾巴，不读整段） ---
    gaps = _build_gaps(
        config,
        present,
        infos,
        base_gap=base_gap,
        sample_rate=sample_rate,
    )

    lead = _silence(sample_rate, lead_silence_seconds)
    tail = _silence(sample_rate, tail_silence_seconds)

    # --- 3) 流式写入 ---
    written_frames = 0
    block_seconds = 1.0
    with open_writer(target, sample_rate=sample_rate, channels=channels) as writer:
        if lead is not None:
            writer.write(lead)
            written_frames += len(lead)

        for index, (_segment, path) in enumerate(present):
            first_block = True
            for block in iter_audio_blocks(path, block_seconds=block_seconds):
                if first_block and fade_samples > 0:
                    import numpy as np

                    count = min(fade_samples, len(block))
                    block = np.array(block, dtype=np.float32, copy=True)
                    block[:count] *= np.linspace(0.0, 1.0, count, dtype=np.float32)
                    first_block = False
                writer.write(block)
                written_frames += len(block)

            gap = gaps[index]
            if gap is not None and len(gap):
                writer.write(gap)
                written_frames += len(gap)

        if tail is not None:
            writer.write(tail)
            written_frames += len(tail)

    duration = written_frames / float(sample_rate)
    elapsed = time.time() - started
    size = target.stat().st_size

    logger.info(
        "拼接完成 %s：%d 段 / %s / %s",
        chapter_id,
        len(present),
        format_duration(duration),
        human_size(size),
    )

    return ConcatResult(
        output_path=target,
        chapter_id=chapter_id,
        segment_count=len(present),
        duration=duration,
        size_bytes=size,
        elapsed_seconds=elapsed,
        missing=missing,
        sample_rate=sample_rate,
    )


def _build_gaps(
    config: AppConfig,
    present: list[tuple[Segment, Path]],
    infos: list[AudioInfo],
    *,
    base_gap: float,
    sample_rate: int,
) -> list["object | None"]:
    """为每段之后的间隔生成静音数组。

    规则（从严到松）：

    1. 最后一段之后不插（收尾静音由 ``tail_silence_seconds`` 统一处理）；
    2. 片段结尾已经自带静音 -> 间隔乘以 ``tail_quiet_gain``（默认 0.35）；
    3. 片段不以句末标点结束（被切断的句子）-> 间隔再乘 0.5，让语流连续。

    判定第 2 条只需要读该片段最后 0.2 秒，代价极低。
    """
    import numpy as np

    audio_config = config.audio
    gaps: list[object | None] = []

    for index, (segment, path) in enumerate(present):
        if index == len(present) - 1:
            gaps.append(None)
            continue

        scale = 1.0

        # --- 尾部是否已有静音 ---
        try:
            tail_frames = int(sample_rate * 0.2)
            frames = max(1, min(tail_frames, infos[index].frames))
            samples, _ = read_audio(path, start=max(0, infos[index].frames - frames))
            if tail_is_quiet(
                samples,
                tail_ms=120.0,
                threshold_dbfs=audio_config.tail_quiet_threshold_dbfs,
                sample_rate=sample_rate,
            ):
                scale *= audio_config.tail_quiet_gain
        except Exception as exc:  # noqa: BLE001 - 读不出来就用默认间隔
            logger.debug("读取 %s 尾部失败，使用默认间隔：%s", path, exc)

        # --- 句子是否被切断 ---
        if not _ENDS_SENTENCE_RE.search(segment.text.strip()):
            scale *= _CONTINUATION_GAIN

        count = int(sample_rate * base_gap * scale)
        gaps.append(np.zeros(count, dtype=np.float32) if count > 0 else None)

    return gaps


def _silence(sample_rate: int, seconds: float) -> "object | None":
    """生成一段静音；``seconds <= 0`` 时返回 ``None``。"""
    if seconds <= 0:
        return None
    import numpy as np

    return np.zeros(int(sample_rate * seconds), dtype=np.float32)


# --------------------------------------------------------------------------- #
# 从 manifest 拼接（不必重新切分）
# --------------------------------------------------------------------------- #


def concat_from_manifest(
    config: AppConfig,
    chapter_id: str,
    *,
    output_path: str | Path | None = None,
    gap_seconds: float | None = None,
) -> ConcatResult:
    """直接按 manifest.json 拼接（不需要重新解析 EPUB）。

    这是「一键把已经生成好的章节拼起来」的入口，也是断点续跑之后的
    收尾步骤。

    Args:
        config: 配置。
        chapter_id: 章节标识。
        output_path: 输出路径；``None`` 时用默认布局。
        gap_seconds: 覆盖默认段间停顿。

    Returns:
        :class:`ConcatResult`。

    Raises:
        FileNotFoundError: 找不到 manifest。
        RuntimeError: 没有可用的片段。
    """
    manifest_path = config.project.output_path / "segments" / chapter_id / "manifest.json"
    manifest = ChapterAudioManifest.load(manifest_path)
    if manifest is None:
        raise FileNotFoundError(
            f"找不到 {chapter_id} 的 manifest（{manifest_path}），请先运行 synth"
        )

    # 按 segment_id 排序，保证播放顺序与文本顺序一致
    records = _ordered_records(manifest)
    segments: list[Segment] = []
    paths: list[Path] = []

    for record in records:
        if record.error:
            continue
        path = config.project.root_path / record.audio_path
        if not path.is_file():
            path = config.project.output_path / "segments" / chapter_id / f"{record.segment_id}.wav"
        segments.append(
            Segment(
                segment_id=record.segment_id,
                chapter_id=chapter_id,
                index=_index_of(record.segment_id),
                text=_load_text_for(config, chapter_id, record.segment_id),
            )
        )
        paths.append(path)

    if not segments:
        raise RuntimeError(f"{chapter_id} 的 manifest 里没有成功生成的片段")

    return concat_chapter(
        config,
        segments,
        segment_paths=paths,
        output_path=output_path,
        gap_seconds=gap_seconds,
        chapter_id=chapter_id,
        title=manifest.title,
    )


def _ordered_records(manifest: ChapterAudioManifest) -> list[SegmentRecord]:
    """按片段序号排序（manifest 里可能是乱序写入的）。"""
    return sorted(
        manifest.records,
        key=lambda record: (
            _index_of(record.segment_id),
            record.segment_id,
        ),
    )


def _index_of(segment_id: str) -> int:
    """从 ``c001-0042`` 里取出 ``42``。"""
    try:
        return int(segment_id.rsplit("-", 1)[-1])
    except ValueError:
        return 0


def _load_text_for(config: AppConfig, chapter_id: str, segment_id: str) -> str:
    """取片段文本用于判断停顿；取不到就返回空串（按默认停顿处理）。"""
    from .synthesis import segments_index_path
    from .utils import read_json

    index = read_json(segments_index_path(config.project.work_path, chapter_id))
    if isinstance(index, dict):
        text = index.get(segment_id)
        if isinstance(text, str):
            return text
    return ""


# --------------------------------------------------------------------------- #
# 批量拼接整本
# --------------------------------------------------------------------------- #


def concat_all_chapters(
    config: AppConfig,
    chapter_ids: list[str] | None = None,
    *,
    merge_into_single: bool = False,
    output_name: str = "full_book",
) -> list[ConcatResult]:
    """批量拼接多章（按 manifest 驱动）。

    Args:
        config: 配置。
        chapter_ids: 要拼接的章节；``None`` 表示扫描 ``segments/`` 下所有章。
        merge_into_single: ``True`` 时把所有章再拼成一个整本文件
            （仍然是流式，不会爆内存）。
        output_name: 整本文件名（不含扩展名）。

    Returns:
        每章的 :class:`ConcatResult`；整本合并时会在末尾追加一个结果。
    """
    if chapter_ids is None:
        chapter_ids = discover_chapters(config)

    results: list[ConcatResult] = []
    for chapter_id in chapter_ids:
        try:
            results.append(concat_from_manifest(config, chapter_id))
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("跳过 %s：%s", chapter_id, exc)

    if merge_into_single and results:
        merged = _merge_chapter_wavs(config, results, output_name)
        results.append(merged)

    return results


def discover_chapters(config: AppConfig) -> list[str]:
    """扫描 ``output/audio/segments/`` 下所有含 manifest 的章节，按名字排序。"""
    base = config.project.output_path / "segments"
    if not base.is_dir():
        return []
    chapters = [
        path.name
        for path in sorted(base.iterdir())
        if path.is_dir() and (path / "manifest.json").is_file()
    ]
    return chapters


def _merge_chapter_wavs(
    config: AppConfig,
    results: list[ConcatResult],
    output_name: str,
) -> ConcatResult:
    """把若干章的 wav 顺序合并成整本（流式）。"""
    import numpy as np

    target = config.project.output_path / f"{output_name}.wav"
    sample_rate = results[0].sample_rate
    channels = config.audio.channels

    # 章与章之间留更长的间隔
    chapter_gap = np.zeros(int(sample_rate * max(0.8, config.audio.min_gap_seconds * 2.5)), dtype=np.float32)

    started = time.time()
    frames = 0
    with open_writer(target, sample_rate=sample_rate, channels=channels) as writer:
        for index, result in enumerate(results):
            for block in iter_audio_blocks(result.output_path, block_seconds=1.0):
                writer.write(block)
                frames += len(block)
            if index < len(results) - 1:
                writer.write(chapter_gap)
                frames += len(chapter_gap)

    duration = frames / float(sample_rate)
    logger.info(
        "整本合并完成 %s：%d 章 / %s / %s",
        output_name,
        len(results),
        format_duration(duration),
        human_size(target.stat().st_size),
    )
    return ConcatResult(
        output_path=target,
        chapter_id=output_name,
        segment_count=sum(r.segment_count for r in results),
        duration=duration,
        size_bytes=target.stat().st_size,
        elapsed_seconds=time.time() - started,
        sample_rate=sample_rate,
    )
