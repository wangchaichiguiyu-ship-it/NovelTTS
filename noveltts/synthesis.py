"""合成编排：把片段逐条送进引擎，并把结果**立刻落盘**。

这一层负责所有“工程性”的事情，让上层 CLI 保持简单：

* **断点续跑**：已存在的片段 wav 会被跳过（``generation.skip_existing``），
  所以中断后重跑不会重复烧算力；``--force`` 可以强制重生成。
* **逐段落盘**：每段合成完立即以 PCM_16 写盘并释放数组引用。整个跑批的
  峰值内存 ≈ 模型 + 一个批次的音频，与章节长度无关。
* **失败隔离**：单段失败会按 ``num_retries`` 退避重试；仍然失败则记录到
  manifest 的 ``error`` 字段并**继续跑后面的段**，最后统一汇总。
  不会因为一句话炸掉整章。
* **清单（manifest）**：每章一份 ``manifest.json``，记录每段的路径、时长、
  尝试次数、错误，以及生成时用的关键参数。音频目录因此是自解释的。
* **设备缓存清理**：每段之后清理 MPS/CUDA 缓存并定期 gc，避免长时间跑批
  内存缓慢膨胀。

典型用法::

    from noveltts.config import AppConfig
    from noveltts.synthesis import SynthesisSession

    config = AppConfig.load("config.yaml")
    with SynthesisSession(config) as session:          # 这里才会加载模型
        report = session.synthesize_chapter(chapter)   # 一章
        session.synthesize_segments(segments[:3])      # 或只跑几段试听
"""

from __future__ import annotations

import gc
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import accelerator
from .audio_io import (
    audio_available,
    normalize_audio,
    probe_audio,
    time_stretch,
    trim_silence,
    write_audio,
)
from .config import AppConfig
from .engine import SynthesisEngine, build_engine
from .models import Chapter, ChapterAudioManifest, Segment, SegmentRecord
from .utils import (
    ensure_dir,
    format_duration,
    get_logger,
    sha1_of,
    write_json,
)

logger = get_logger("synthesis")


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


@dataclass
class SynthesisReport:
    """一次跑批的结果汇总。"""

    chapter_id: str
    title: str = ""
    total: int = 0
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "total": self.total,
            "generated": self.generated,
            "skipped": self.skipped,
            "failed": self.failed,
            "audio_seconds": round(self.audio_seconds, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "rtf": round(self.elapsed_seconds / self.audio_seconds, 3)
            if self.audio_seconds > 0
            else None,
            "errors": [{"segment_id": sid, "error": err} for sid, err in self.errors],
        }

    def summary(self) -> str:
        """人类可读的一行摘要。"""
        parts = [
            f"{self.chapter_id} {self.title}".strip(),
            f"新生成 {self.generated}",
            f"跳过 {self.skipped}",
        ]
        if self.failed:
            parts.append(f"失败 {self.failed}")
        parts.append(f"音频 {format_duration(self.audio_seconds)}")
        parts.append(f"耗时 {format_duration(self.elapsed_seconds)}")
        if self.audio_seconds > 0 and self.elapsed_seconds > 0:
            parts.append(f"RTF {self.elapsed_seconds / self.audio_seconds:.2f}")
        return " | ".join(parts)


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #


class SynthesisSession:
    """持有引擎的合成会话；负责目录布局、落盘与清单维护。

    Args:
        config: 全局配置（用到 ``project`` / ``generation`` / ``voice`` / ``audio``）。
        engine: 注入自定义引擎（测试用）；``None`` 时按配置构造真实引擎。
        dry_run: ``True`` 时用假引擎，**不加载任何模型**。

    Note:
        构造会话就会触发模型加载（除非 ``dry_run`` 或注入了 ``engine``）。
        这是刻意的：把昂贵的初始化放在一个明确的点上，而不是藏在第一次
        调用里。
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        engine: SynthesisEngine | None = None,
        dry_run: bool = False,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.config.project.ensure_dirs()

        available, reason = audio_available()
        if not available:
            raise RuntimeError(f"音频读写依赖不可用：{reason}")

        if engine is not None:
            self.engine = engine
            #: 注入进来的引擎由调用方负责关闭。会话关闭时不去动它——
            #: 否则「同一个引擎跑多个会话」的用法（例如先全量生成、
            #: 再 force 重跑一部分）会在第二个会话里撞上 "引擎已关闭"。
            self._owns_engine = False
        else:
            self.engine = build_engine(
                config.generation, config.voice, device=device, dry_run=dry_run
            )
            self._owns_engine = True

        #: 真实采样率以引擎为准（模型可能不是 24 kHz）
        self.sample_rate = int(getattr(self.engine, "sample_rate", 24_000))
        self._segment_dir = ensure_dir(config.project.output_path / "segments")

    # ------------------------------------------------------------------ 布局
    def chapter_dir(self, chapter_id: str) -> Path:
        """返回某章的音频目录（会自动创建）。"""
        return ensure_dir(self._segment_dir / chapter_id)

    def segment_path(self, segment: Segment) -> Path:
        """返回某个片段的 wav 路径。"""
        return self.chapter_dir(segment.chapter_id) / f"{segment.segment_id}.wav"

    def manifest_path(self, chapter_id: str) -> Path:
        """返回某章 manifest 的路径。"""
        return self.chapter_dir(chapter_id) / "manifest.json"

    # ------------------------------------------------------------- 单段合成
    def synthesize_segment(
        self,
        segment: Segment,
        *,
        force: bool = False,
        language: str | None = None,
    ) -> SegmentRecord:
        """合成单个片段并落盘。

        Args:
            segment: 目标片段。
            force: ``True`` 时即使文件已存在也重新生成。
            language: 语言覆盖。

        Returns:
            :class:`~noveltts.models.SegmentRecord`（成功或失败都会返回）。
        """
        target = self.segment_path(segment)
        relative = self._relative(target)

        # --- 断点续跑：文件已在且非空就跳过 ---
        if not force and self.config.generation.skip_existing and target.is_file():
            try:
                info = probe_audio(target)
                if info.frames > 0:
                    return SegmentRecord(
                        segment_id=segment.segment_id,
                        audio_path=relative,
                        duration=info.duration,
                        sample_rate=info.sample_rate,
                        attempts=0,
                        skipped=True,
                    )
            except Exception as exc:  # noqa: BLE001 - 坏文件当作不存在，重生成
                logger.warning("片段 %s 已存在但读取失败，将重新生成：%s", segment.segment_id, exc)

        return self._generate_with_retries(segment, target, relative, language=language)

    def _postprocess(self, samples: "object", sample_rate: int) -> "object":
        """对单段音频做后处理：语速 -> 裁静音 -> 响度归一。

        顺序是有讲究的：

        1. **先调语速** —— 变速会改变波形长度与包络，放在归一化之前才能让
           归一化基于最终音频计算；
        2. **再裁静音** —— 变速可能引入轻微的边界噪声；
        3. **最后归一化** —— 保证最终落盘的文件响度一致。
        """
        audio_config = self.config.audio

        if abs(audio_config.speech_rate - 1.0) > 1e-6:
            samples = time_stretch(
                samples, rate=audio_config.speech_rate, sample_rate=sample_rate
            )

        if audio_config.trim_silence:
            samples = trim_silence(
                samples,
                threshold_dbfs=audio_config.silence_threshold_dbfs,
                sample_rate=sample_rate,
            )
        return normalize_audio(samples, audio_config)

    def _synthesize_batch(
        self,
        segments: list[Segment],
        *,
        language: str | None,
        attempts: int,
    ) -> tuple[list[SegmentRecord | None], str | None]:
        """一次前向生成多段（**真正的批处理**）。

        批量调用 ``engine.synthesize(texts)`` 能显著提升 GPU 利用率：
        实测在 RTX 3090 上，5 段一批的 RTF 从 1.74 降到 0.53（约 3.3 倍），
        峰值显存只从 4.7G 涨到 7.7G。

        Args:
            segments: 本批片段（长度即批大小）。
            language: 语言覆盖。
            attempts: 本批是第几次尝试（写进记录，便于排查）。

        Returns:
            ``(记录列表, 错误信息)``。整批成功时错误为 ``None``；
            整批失败时记录列表为等长的 ``None``，由调用方决定是整批重试
            还是降级为逐段生成。
        """
        texts = [segment.text for segment in segments]

        try:
            result = self.engine.synthesize(texts, language=language)
        except KeyboardInterrupt:
            logger.warning("收到中断信号，停止当前批次")
            raise
        except Exception as exc:  # noqa: BLE001 - 批失败要把原因交回调用方
            return [None] * len(segments), f"{type(exc).__name__}: {exc}"

        waveforms = list(result.waveforms)
        sample_rate = int(result.sample_rate or self.sample_rate)

        # 引擎返回的段数应当与请求一致；不一致说明有段被静默丢弃，
        # 这种情况必须暴露出来，否则会把 A 的音频写到 B 的文件名上。
        if len(waveforms) != len(segments):
            return (
                [None] * len(segments),
                f"引擎返回段数不匹配：请求 {len(segments)} 段，返回 {len(waveforms)} 段",
            )

        records: list[SegmentRecord | None] = []
        for segment, samples in zip(segments, waveforms):
            target = self.segment_path(segment)
            relative = self._relative(target)
            try:
                processed = self._postprocess(samples, sample_rate)
                write_audio(target, processed, sample_rate)
                duration = len(processed) / float(sample_rate)
                del processed
                records.append(
                    SegmentRecord(
                        segment_id=segment.segment_id,
                        audio_path=relative,
                        duration=duration,
                        sample_rate=sample_rate,
                        attempts=max(1, attempts),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 单段落盘失败不该毁掉整批
                records.append(
                    SegmentRecord(
                        segment_id=segment.segment_id,
                        audio_path=relative,
                        attempts=max(1, attempts),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        del waveforms
        del result
        return records, None

    def _generate_with_retries(
        self,
        segment: Segment,
        target: Path,
        relative: str,
        *,
        language: str | None,
    ) -> SegmentRecord:
        """逐段生成（带退避重试）。

        Note:
            批量路径见 :meth:`_synthesize_batch`。这里是**兜底路径**：
            批处理整体失败后用它逐段隔离，这样一段的问题不会连累同批其它段。
        """
        generation = self.config.generation
        attempts = 0
        last_error: str | None = None

        while attempts <= generation.num_retries:
            attempts += 1
            try:
                result = self.engine.synthesize([segment.text], language=language)
                if not result.waveforms:
                    raise RuntimeError("引擎返回了空结果")

                sample_rate = int(result.sample_rate or self.sample_rate)
                samples = self._postprocess(result.waveforms[0], sample_rate)
                write_audio(target, samples, sample_rate)

                duration = len(samples) / float(sample_rate)
                # 显式断引用：别让上一段的数组活到下一段
                del samples
                del result

                return SegmentRecord(
                    segment_id=segment.segment_id,
                    audio_path=relative,
                    duration=duration,
                    sample_rate=sample_rate,
                    attempts=attempts,
                )
            except KeyboardInterrupt:
                logger.warning("收到中断信号，停止当前片段")
                raise
            except Exception as exc:  # noqa: BLE001 - 任何异常都要记录并重试
                last_error = f"{type(exc).__name__}: {exc}"
                if attempts > generation.num_retries:
                    break
                backoff = generation.retry_backoff * (2 ** (attempts - 1))
                logger.warning(
                    "片段 %s 第 %d 次失败（%s），%.1fs 后重试",
                    segment.segment_id,
                    attempts,
                    last_error,
                    backoff,
                )
                time.sleep(backoff)

        logger.error("片段 %s 最终失败：%s", segment.segment_id, last_error)
        return SegmentRecord(
            segment_id=segment.segment_id,
            audio_path=relative,
            attempts=attempts,
            error=last_error or "未知错误",
        )

    # ------------------------------------------------------------- 多段合成
    def synthesize_segments(
        self,
        segments: list[Segment],
        *,
        chapter_id: str = "",
        title: str = "",
        force: bool = False,
        language: str | None = None,
        batch_size: int | None = None,
        write_manifest: bool = True,
    ) -> SynthesisReport:
        """合成一批片段（一章或任意子集）。

        Args:
            segments: 片段列表（按顺序）。
            chapter_id: 章节标识；为空时取第一个片段的 ``chapter_id``。
            title: 章节标题（写进 manifest）。
            force: 强制重新生成已有片段。
            language: 语言覆盖。
            batch_size: 批大小覆盖；``None`` 用配置值。**16 GB 内存请保持 1**。
            write_manifest: 是否写 ``manifest.json``。

        Returns:
            :class:`SynthesisReport`。
        """
        if not segments:
            return SynthesisReport(chapter_id=chapter_id or "empty")

        chapter_id = chapter_id or segments[0].chapter_id
        batch_size = self._resolve_batch_size(batch_size)
        report = SynthesisReport(chapter_id=chapter_id, title=title, total=len(segments))
        started = time.time()

        manifest = ChapterAudioManifest(
            chapter_id=chapter_id,
            title=title,
            text_sha1=sha1_of("".join(s.text for s in segments)),
            created_at=_now(),
            updated_at=_now(),
            sample_rate=self.sample_rate,
            segment_count=len(segments),
            settings=self._settings_snapshot(),
        )

        # 合并旧的 manifest：本次没涉及的片段记录要保留下来。
        # 否则只重跑 1 段就会把整章的清单覆盖成 1 条记录，
        # 后续 concat 会以为整章只有那一段。
        existing = ChapterAudioManifest.load(self.manifest_path(chapter_id))
        if existing is not None:
            manifest.created_at = existing.created_at
            if not title:
                manifest.title = existing.title
            # 章节的片段总数取「旧记录数」与「本批段数」的较大值，
            # 这样部分重跑不会把总数改小
            manifest.segment_count = max(manifest.segment_count, existing.segment_count)
            batch_ids = {segment.segment_id for segment in segments}
            for record in existing.records:
                if record.segment_id not in batch_ids:
                    manifest.upsert(record)

        logger.info(
            "开始合成 %s：%d 段，批大小 %d，设备 %s",
            chapter_id,
            len(segments),
            batch_size,
            getattr(self.engine, "describe", lambda: "?")(),
        )

        # 记录最近若干段的耗时，用于给出准确的剩余时间估计
        recent: deque[float] = deque(maxlen=20)
        processed = 0
        generated_since_clean = 0
        clean_every = self._cache_clear_interval()

        index = 0
        total_segments = len(segments)
        while index < total_segments:
            window = segments[index:index + batch_size]
            batch_started = time.time()

            # ---- 1) 先把已存在的片段挑出去，只对真正要生成的做批处理 ----
            todo: list[Segment] = []
            for segment in window:
                if not force and self.config.generation.skip_existing:
                    record = self._skip_if_present(segment)
                    if record is not None:
                        manifest.upsert(record)
                        report.skipped += 1
                        report.audio_seconds += record.duration
                        processed += 1
                        continue
                todo.append(segment)

            # ---- 2) 真正批处理：一次前向生成多段 ----
            records: list[SegmentRecord] = []
            if todo:
                batch_records, batch_error = self._synthesize_batch(
                    todo, language=language, attempts=1
                )

                if batch_error is None:
                    records = [r for r in batch_records if r is not None]
                else:
                    # 整批失败 -> 逐段重试，把某一段的问题与同批其它段隔离开。
                    # 这是批处理与"失败隔离"之间必须做的取舍：批处理快，
                    # 但一荣俱荣一损俱损，所以失败时要能退回去。
                    logger.warning(
                        "批处理失败（%d 段），降级为逐段重试：%s", len(todo), batch_error
                    )
                    new_size = self._maybe_shrink_batch(batch_size, batch_error)
                    if new_size != batch_size:
                        batch_size = new_size
                        logger.info("后续批大小调整为 %d", batch_size)
                    records = [
                        self._generate_with_retries(
                            segment,
                            self.segment_path(segment),
                            self._relative(self.segment_path(segment)),
                            language=language,
                        )
                        for segment in todo
                    ]

            index += len(window)

            # ---- 3) 记账 ----
            for segment, record in zip(todo, records):
                manifest.upsert(record)
                processed += 1

                if record.error:
                    report.failed += 1
                    report.errors.append((segment.segment_id, record.error))
                    # 显存不足时自动把批大小减半：不这样做的话，一个过大的
                    # 批次会让后面每一段都重复 OOM，整章白跑。
                    new_size = self._maybe_shrink_batch(batch_size, record.error)
                    if new_size != batch_size:
                        batch_size = new_size
                else:
                    report.generated += 1
                    report.audio_seconds += record.duration
                    generated_since_clean += 1

            # 批次耗时按「本批音频时长」摊到每一段，用于估算剩余时间
            if todo:
                per_segment = (time.time() - batch_started) / len(todo)
                recent.extend([per_segment] * len(todo))

            if self.config.generation.progress_every and (
                processed % self.config.generation.progress_every == 0
                or processed == total_segments
            ):
                self._log_progress(processed, total_segments, report, recent, segments)

            # 按「段数间隔」清理设备缓存，而不是每批都清。
            # 每批都清会打断流水线；从不清则分配器缓存缓慢膨胀。
            if clean_every and generated_since_clean >= clean_every:
                gc.collect()
                _release_device_cache(self.engine)
                generated_since_clean = 0

        # 收尾再做一次整理，把长跑积累的缓存还给系统
        gc.collect()
        _release_device_cache(self.engine)

        manifest.updated_at = _now()
        report.elapsed_seconds = time.time() - started

        if write_manifest:
            self._write_manifest(manifest)

        logger.info("完成 %s", report.summary())
        if report.errors:
            logger.warning("有 %d 段失败，可用 --only-failed 单独重跑", len(report.errors))
        return report

    # ------------------------------------------------------------- 章节封装
    def synthesize_chapter(
        self,
        chapter: Chapter,
        *,
        force: bool = False,
        language: str | None = None,
        include_title: bool = True,
        limit: int = 0,
        only_segments: list[int] | None = None,
    ) -> SynthesisReport:
        """合成一整章。

        Args:
            chapter: 章节对象（必须已经切分，即 ``chapter.segments`` 非空）。
            force: 强制重新生成。
            language: 语言覆盖。
            include_title: 是否把标题也读出来（作为第一个片段）。
            limit: 只生成前 N 段（试听用）；``0`` 表示全部。
            only_segments: 只生成指定序号的片段（1-based）；``None`` 表示全部。

        Returns:
            :class:`SynthesisReport`。

        Raises:
            ValueError: 章节还没有切分。
        """
        segments = chapter.segments
        if not segments:
            raise ValueError(
                f"章节 {chapter.chapter_id} 还没有切分，请先调用 segment_chapter()"
            )

        selected = list(segments)
        if not include_title:
            selected = [s for s in selected if s.kind != "heading"]
        if only_segments:
            wanted = set(only_segments)
            selected = [s for s in selected if s.index in wanted]
        if limit > 0:
            selected = selected[:limit]

        return self.synthesize_segments(
            selected,
            chapter_id=chapter.chapter_id,
            title=chapter.title,
            force=force,
            language=language,
        )

    # ------------------------------------------------------------ 失败重跑
    def retry_failed(self, chapter_id: str, *, language: str | None = None) -> SynthesisReport:
        """只重跑 manifest 里记录为失败的片段。

        完成一章后如果只有零星几段失败（长文本偶发），用这个比整章重跑
        省得多。

        Args:
            chapter_id: 章节标识。
            language: 语言覆盖。

        Returns:
            :class:`SynthesisReport`。

        Raises:
            FileNotFoundError: 没有找到 manifest。
        """
        manifest = ChapterAudioManifest.load(self.manifest_path(chapter_id))
        if manifest is None:
            raise FileNotFoundError(f"找不到 {chapter_id} 的 manifest，无法重跑失败片段")

        failed_ids = {record.segment_id for record in manifest.failed}
        if not failed_ids:
            logger.info("%s 没有失败的片段", chapter_id)
            return SynthesisReport(chapter_id=chapter_id, title=manifest.title)

        # 从 manifest 里恢复片段文本（不依赖 chapter 对象）
        segments = []
        for record in manifest.records:
            if record.segment_id not in failed_ids:
                continue
            text = _load_segment_text(
                self.config.project.work_path, chapter_id, record.segment_id
            )
            if text is None:
                logger.warning("找不到片段 %s 的文本，跳过", record.segment_id)
                continue
            segments.append(_rebuild_segment(chapter_id, record.segment_id, text))

        if not segments:
            logger.warning("没有可重跑的片段（文本索引缺失）")
            return SynthesisReport(chapter_id=chapter_id, title=manifest.title)

        return self.synthesize_segments(
            segments,
            chapter_id=chapter_id,
            title=manifest.title,
            force=True,
            language=language,
        )

    # ---------------------------------------------------------------- 内部
    def _resolve_batch_size(self, override: int | None) -> int:
        """决定本次跑批的批大小。

        优先级：显式参数 > 配置文件 > 引擎探明的硬件推荐值。

        Note:
            ``generation.batch_size`` 允许写 ``auto``（或 ``-1``），此时用
            :attr:`QwenTTSEngine.profile` 里按显存推导出的值。
            非 Qwen 引擎（例如测试用的假引擎）没有 profile，退回 1。
        """
        if override is not None and override > 0:
            return int(override)

        configured = self.config.generation.batch_size
        if isinstance(configured, str) and configured.lower() == "auto":
            configured = -1
        if isinstance(configured, int) and configured > 0:
            return configured

        profile = getattr(self.engine, "profile", None)
        if profile is not None:
            return max(1, int(profile.batch_size))
        return 1

    def _cache_clear_interval(self) -> int:
        """返回「每生成多少段清理一次设备缓存」；0 表示不清理。"""
        configured = self.config.generation.clear_cache_every
        if configured > 0:
            return configured
        profile = getattr(self.engine, "profile", None)
        if profile is not None:
            return int(profile.clear_cache_every)
        return 50

    def _maybe_shrink_batch(self, current: int, error: str | None) -> int:
        """遇到显存不足时把批大小减半（最低 1）。

        Args:
            current: 当前批大小。
            error: 失败原因文本。

        Returns:
            新的批大小；不需要调整时原样返回 ``current``。
        """
        if not error or current <= 1:
            return current
        if not self.config.generation.oom_batch_downgrade:
            return current
        if not accelerator.is_oom_error(RuntimeError(error)):
            return current

        new_size = max(1, current // 2)
        logger.warning(
            "检测到显存不足，批大小 %d -> %d（后续片段按新批大小继续）",
            current,
            new_size,
        )
        _release_device_cache(self.engine)
        return new_size

    def _skip_if_present(self, segment: Segment) -> SegmentRecord | None:
        """文件已存在则返回 skip 记录，否则 ``None``。"""
        target = self.segment_path(segment)
        if not target.is_file():
            return None
        try:
            info = probe_audio(target)
            if info.frames <= 0:
                return None
        except Exception:  # noqa: BLE001
            return None
        return SegmentRecord(
            segment_id=segment.segment_id,
            audio_path=self._relative(target),
            duration=info.duration,
            sample_rate=info.sample_rate,
            attempts=0,
            skipped=True,
        )

    def _log_progress(
        self,
        processed: int,
        total: int,
        report: SynthesisReport,
        recent: "deque[float]",
        segments: list[Segment],
    ) -> None:
        """打印带 ETA 的进度。"""
        percent = processed / total * 100
        message = f"[{processed}/{total}] {percent:5.1f}%  音频 {format_duration(report.audio_seconds)}"
        if recent:
            average = sum(recent) / len(recent)
            remaining = (total - processed) * average
            message += f"  预计剩余 {format_duration(remaining)}"
        if report.failed:
            message += f"  失败 {report.failed}"
        logger.info(message)

    def _write_manifest(self, manifest: ChapterAudioManifest) -> Path:
        """写 manifest.json。"""
        path = self.manifest_path(manifest.chapter_id)
        write_json(path, manifest.to_dict())
        logger.debug("manifest 已写入 %s", path)
        return path

    def _settings_snapshot(self) -> dict[str, object]:
        """记录影响音频结果的关键参数，方便日后回溯。"""
        generation = self.config.generation
        audio = self.config.audio
        voice = self.config.voice
        return {
            "model_path": generation.model_path,
            "dtype": generation.dtype,
            "language": generation.language,
            "max_new_tokens": generation.max_new_tokens,
            "non_streaming_mode": generation.non_streaming_mode,
            "temperature": generation.temperature,
            "top_k": generation.top_k,
            "top_p": generation.top_p,
            "repetition_penalty": generation.repetition_penalty,
            "ref_audio": voice.ref_audio,
            "voice_mode": "xvec" if (voice.x_vector_only_mode or voice.mode == "xvec") else "icl",
            "sample_rate": self.sample_rate,
            "normalize": audio.normalize,
            "target_dbfs": audio.target_dbfs,
            "min_gap_seconds": audio.min_gap_seconds,
            "speech_rate": audio.speech_rate,
        }

    def _relative(self, path: Path) -> str:
        """把绝对路径转成相对项目根目录的路径（便于整个目录搬家）。"""
        try:
            return str(path.relative_to(self.config.project.root_path))
        except ValueError:
            return str(path)

    # ---------------------------------------------------------------- 收尾
    def close(self) -> None:
        """释放引擎。

        Note:
            只关闭**本会话自己创建的**引擎。外部注入的引擎归调用方所有，
            这里不能替它做决定——否则同一个引擎跑完第一个会话就被关掉，
            第二个会话会直接报 "引擎已关闭"。
        """
        if not self._owns_engine:
            logger.debug("引擎由调用方注入，跳过关闭")
            return
        try:
            self.engine.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("关闭引擎时出错（忽略）：%s", exc)

    def __enter__(self) -> "SynthesisSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------ 便捷入口
    def synthesize_text(
        self,
        text: str,
        *,
        chapter_id: str = "adhoc",
        title: str = "",
        language: str | None = None,
    ) -> list[SegmentRecord]:
        """把一段任意文本直接合成为音频（**一键生成一小段**）。

        Args:
            text: 文本，可含空行分段。
            chapter_id: 输出目录名。
            title: 可选标题（会作为第一个片段朗读）。
            language: 语言覆盖。

        Returns:
            该文本切分后每个片段的记录。

        Example:
            >>> records = session.synthesize_text("Hello there. How are you?")  # doctest: +SKIP
            >>> records[0].audio_path                                          # doctest: +SKIP
            'output/audio/segments/adhoc/adhoc-0001.wav'
        """
        from .segmenter import TextSegmenter

        segmenter = TextSegmenter(
            self.config.segment, language=self.config.generation.language.lower()[:2] or "en"
        )
        segments = segmenter.segment_text(
            text, chapter_id=chapter_id, title=title
        )
        if not segments:
            logger.warning("输入文本切分后为空，没有可生成的内容")
            return []

        report = self.synthesize_segments(
            segments, chapter_id=chapter_id, title=title or "Ad hoc", language=language
        )
        if report.failed:
            logger.warning("有 %d 段失败", report.failed)

        records: list[SegmentRecord] = []
        manifest = ChapterAudioManifest.load(self.manifest_path(chapter_id))
        if manifest is not None:
            for segment in segments:
                record = manifest.record_for(segment.segment_id)
                if record is not None:
                    records.append(record)
        return records


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def _now() -> str:
    """当前时间（ISO 8601，秒级）。"""
    return datetime.now().isoformat(timespec="seconds")


def _release_device_cache(engine: SynthesisEngine) -> None:
    """定期清理设备缓存（不同引擎实现可能没有这个方法）。"""
    device = getattr(engine, "device", None)
    if device:
        from .engine import _empty_device_cache

        _empty_device_cache(str(device))


def _load_segment_text(work_path: Path, chapter_id: str, segment_id: str) -> str | None:
    """从片段文本索引里取出某段的文本。

    索引由 :func:`dump_segments_index` 写出；用于 :meth:`SynthesisSession.retry_failed`
    在只拿到 manifest 的情况下恢复片段文本。
    """
    from .utils import read_json

    index = read_json(segments_index_path(work_path, chapter_id))
    if not isinstance(index, dict):
        return None
    text = index.get(segment_id)
    return text if isinstance(text, str) else None


def segments_index_path(work_path: Path, chapter_id: str) -> Path:
    """片段文本索引的路径：``<work>/segments/<chapter_id>.json``。"""
    return Path(work_path) / "segments" / f"{chapter_id}.json"


def _rebuild_segment(chapter_id: str, segment_id: str, text: str) -> Segment:
    """根据 id 与文本重建一个最小可用的 :class:`Segment`。"""
    try:
        index = int(segment_id.rsplit("-", 1)[-1])
    except ValueError:
        index = 0
    return Segment(
        segment_id=segment_id,
        chapter_id=chapter_id,
        index=index,
        text=text,
    )


def dump_segments_index(
    segments: list[Segment],
    *,
    config: AppConfig | None = None,
    work_path: str | Path | None = None,
) -> Path:
    """把「片段 id -> 文本」写成索引文件，供失败重跑时恢复文本。

    Args:
        segments: 片段列表。
        config: 用来定位 ``work`` 目录。
        work_path: 直接指定 work 目录（优先于 ``config``）。

    Returns:
        索引文件路径。
    """
    if not segments:
        raise ValueError("segments 为空")
    chapter_id = segments[0].chapter_id
    if work_path is None:
        work_path = config.project.work_path if config else Path("output/work")
    path = segments_index_path(Path(work_path), chapter_id)
    ensure_dir(path.parent)
    write_json(path, {segment.segment_id: segment.text for segment in segments})
    return path
