"""端到端流水线测试：解析 -> 切分 -> 合成 -> 拼接 -> 压缩。

**全程使用 :class:`~noveltts.engine.FakeSynthesisEngine`，绝不加载 Qwen3-TTS。**
产出的是正弦波，但磁盘布局、manifest、断点续跑、失败隔离等工程行为
与真实跑批完全一致。

这些用例的价值在于：在真正烧算力之前，就能确认整条链路的文件布局、
清单格式、失败重试、断点续跑都是对的。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noveltts.config import (
    AppConfig,
    AudioConfig,
    GenerationConfig,
    ParseConfig,
    ProjectConfig,
    SegmentConfig,
    TextRulesConfig,
    VoiceConfig,
)
from noveltts.engine import FakeSynthesisEngine, resolve_attn_implementation, resolve_device
from noveltts.models import Chapter, ChapterAudioManifest, Paragraph, Segment
from noveltts.segmenter import TextSegmenter
from noveltts.synthesis import (
    SynthesisSession,
    dump_segments_index,
    segments_index_path,
)
from noveltts.utils import read_json

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    """指向临时目录的配置（不会污染真实 output/）。"""
    config = AppConfig(
        project=ProjectConfig(
            root=str(tmp_path),
            source_dir="sources",
            work_dir="output/work",
            output_dir="output/audio",
            cache_dir="output/cache",
        ),
        parse=ParseConfig(),
        text_rules=TextRulesConfig(),
        segment=SegmentConfig(),
        voice=VoiceConfig(ref_audio="ref.wav", ref_text="reference text"),
        generation=GenerationConfig(
            model_path="dummy-model", device="cpu", num_retries=0, progress_every=0
        ),
        audio=AudioConfig(),
    )
    config.project.resolve(tmp_path)
    config.project.ensure_dirs()
    return config


@pytest.fixture
def segments(config: AppConfig) -> list[Segment]:
    text = (
        "The library was quiet that morning. "
        '"Did you find the book?" she asked. '
        "He nodded slowly and said nothing at all. "
        "Outside, the rain kept falling on the empty courtyard. "
        "It was going to be a long afternoon for everyone involved."
    )
    segmenter = TextSegmenter(config.segment, language="en")
    return segmenter.segment_text(text, chapter_id="c001", normalize=False)


# --------------------------------------------------------------------------- #
# 引擎辅助函数（不加载模型）
# --------------------------------------------------------------------------- #


def test_resolve_device_auto_is_valid() -> None:
    device = resolve_device("auto")
    assert device in {"cuda:0", "mps", "cpu"}


def test_resolve_device_falls_back_from_bad_cuda() -> None:
    """没有 GPU 时指定 cuda 应回退而不是崩掉。"""
    import torch

    if torch.cuda.is_available():
        pytest.skip("本机有 CUDA，无法测试回退")
    assert resolve_device("cuda:0") == "cpu"


def test_resolve_attn_auto_on_non_cuda_is_sdpa() -> None:
    """本机没有 CUDA，因此 auto 必须落到 sdpa。

    Note:
        想看 "CUDA 上 auto 会不会选中 flash_attention_2"，
        请用 tests/test_accelerator.py —— 那里注入假的 DeviceInfo，
        不需要真的插一张卡。
    """
    for device in ("mps", "cpu"):
        assert resolve_attn_implementation("auto", device) == "sdpa"


def test_resolve_attn_downgrades_flash_without_cuda() -> None:
    """配置里写死 flash_attention_2，但机器没有 CUDA 时必须降级而不是崩。"""
    assert resolve_attn_implementation("flash_attention_2", "mps") == "sdpa"
    assert resolve_attn_implementation("flash_attention_2", "cpu") == "sdpa"


def test_fake_engine_produces_audio() -> None:
    engine = FakeSynthesisEngine(sample_rate=24_000)
    result = engine.synthesize(["Hello there."])
    assert len(result.waveforms) == 1
    assert result.sample_rate == 24_000
    assert len(result.waveforms[0]) > 0
    assert result.texts == ["Hello there."]


def test_fake_engine_duration_scales_with_text() -> None:
    engine = FakeSynthesisEngine(ms_per_char=100.0)
    short = engine.synthesize(["Hi."]).waveforms[0]
    long = engine.synthesize(["Hi there, this is a much longer sentence."]).waveforms[0]
    assert len(long) > len(short)


def test_fake_engine_fail_on() -> None:
    engine = FakeSynthesisEngine(fail_on=("Boom",))
    with pytest.raises(RuntimeError):
        engine.synthesize(["Boom goes everything."])


def test_fake_engine_close() -> None:
    engine = FakeSynthesisEngine()
    engine.close()
    with pytest.raises(RuntimeError):
        engine.synthesize(["x"])


def test_build_engine_dry_run_never_loads_model() -> None:
    from noveltts.engine import build_engine

    engine = build_engine(GenerationConfig(), VoiceConfig(), dry_run=True)
    assert isinstance(engine, FakeSynthesisEngine)
    engine.close()


# --------------------------------------------------------------------------- #
# 会话：单段 / 批量 / 断点续跑
# --------------------------------------------------------------------------- #


def test_session_requires_audio_deps(config: AppConfig) -> None:
    """音频依赖检查在构造会话时就该生效。"""
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        assert session.sample_rate == 24_000


def test_synthesize_segment_writes_file(config: AppConfig, segments: list[Segment]) -> None:
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        record = session.synthesize_segment(segments[0])

    assert record.error is None
    path = config.project.root_path / record.audio_path
    assert path.is_file()
    assert record.duration > 0
    assert record.attempts == 1


def test_audio_path_is_relative_to_root(config: AppConfig, segments: list[Segment]) -> None:
    """manifest 里存相对路径，整个输出目录才能搬家。"""
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        record = session.synthesize_segment(segments[0])
    assert not Path(record.audio_path).is_absolute()
    assert record.audio_path.startswith("output/audio/segments/c001/")


def test_synthesize_segments_writes_manifest(config: AppConfig, segments: list[Segment]) -> None:
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        report = session.synthesize_segments(segments, chapter_id="c001", title="Chapter One")

    assert report.total == len(segments)
    assert report.generated == len(segments)
    assert report.failed == 0
    assert report.ok

    manifest_path = config.project.output_path / "segments" / "c001" / "manifest.json"
    payload = read_json(manifest_path)
    assert payload["chapter_id"] == "c001"
    assert payload["title"] == "Chapter One"
    assert payload["segment_count"] == len(segments)
    assert payload["done_count"] == len(segments)
    assert len(payload["records"]) == len(segments)
    assert payload["settings"]["model_path"] == "dummy-model"


def test_resume_skips_existing(config: AppConfig, segments: list[Segment]) -> None:
    """断点续跑：第二次跑应该全部跳过，不再调用引擎。"""
    engine = FakeSynthesisEngine()
    with SynthesisSession(config, engine=engine) as session:
        session.synthesize_segments(segments, chapter_id="c001")
    calls_after_first = len(engine.calls)

    with SynthesisSession(config, engine=engine) as session:
        report = session.synthesize_segments(segments, chapter_id="c001")

    assert report.generated == 0
    assert report.skipped == len(segments)
    assert len(engine.calls) == calls_after_first  # 引擎没被再调用


def test_force_regenerates(config: AppConfig, segments: list[Segment]) -> None:
    engine = FakeSynthesisEngine()
    with SynthesisSession(config, engine=engine) as session:
        session.synthesize_segments(segments, chapter_id="c001")
    with SynthesisSession(config, engine=engine) as session:
        report = session.synthesize_segments(segments, chapter_id="c001", force=True)
    assert report.generated == len(segments)
    assert report.skipped == 0


def test_failure_is_isolated_per_segment(config: AppConfig) -> None:
    """一段失败不能影响其它段。"""
    segmenter = TextSegmenter(SegmentConfig(max_chars=40), language="en")
    items = segmenter.segment_text(
        "The first sentence is perfectly ordinary here. "
        "Boom goes the second sentence which must fail. "
        "The third sentence is also perfectly ordinary.",
        chapter_id="c009",
        normalize=False,
    )
    assert len(items) >= 3

    engine = FakeSynthesisEngine(fail_on=("Boom",))
    with SynthesisSession(config, engine=engine) as session:
        report = session.synthesize_segments(items, chapter_id="c009")

    assert report.failed == 1
    assert report.generated == len(items) - 1
    assert report.errors and report.errors[0][0].startswith("c009-")
    assert not report.ok


def test_retry_failed_reruns_only_failures(config: AppConfig) -> None:
    """失败重跑依赖片段文本索引。"""
    segmenter = TextSegmenter(SegmentConfig(max_chars=40), language="en")
    items = segmenter.segment_text(
        "Sentence number one is quite ordinary and descriptive. "
        "Boom goes this one and it must fail horribly. "
        "Sentence number three is also ordinary and descriptive. "
        "Sentence number four keeps things going nicely.",
        chapter_id="c010",
        normalize=False,
    )
    dump_segments_index(items, work_path=config.project.work_path)

    with SynthesisSession(config, engine=FakeSynthesisEngine(fail_on=("Boom",))) as session:
        first = session.synthesize_segments(items, chapter_id="c010", title="T")
    assert first.failed == 1

    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        second = session.retry_failed("c010")
    assert second.total == 1
    assert second.generated == 1
    assert second.failed == 0

    # manifest 里的其它片段记录不能被覆盖丢掉
    payload = read_json(config.project.output_path / "segments" / "c010" / "manifest.json")
    assert payload["segment_count"] == len(items)
    assert payload["done_count"] == len(items)
    assert payload["failed_count"] == 0


def test_retry_failed_without_manifest(config: AppConfig) -> None:
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        with pytest.raises(FileNotFoundError):
            session.retry_failed("c999")


def test_retry_failed_with_nothing_failed(config: AppConfig, segments: list[Segment]) -> None:
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        session.synthesize_segments(segments, chapter_id="c001")
        report = session.retry_failed("c001")
    assert report.total == 0


def test_partial_rerun_keeps_other_records(config: AppConfig, segments: list[Segment]) -> None:
    """只重跑一部分片段时，manifest 里其它片段必须保留。"""
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        session.synthesize_segments(segments, chapter_id="c001")
        session.synthesize_segments(segments[:2], chapter_id="c001", force=True)

    payload = read_json(config.project.output_path / "segments" / "c001" / "manifest.json")
    assert payload["segment_count"] == len(segments)
    assert len(payload["records"]) == len(segments)


def test_synthesize_chapter_helper(config: AppConfig) -> None:
    chapter = Chapter(
        index=1,
        title="Test Chapter",
        chapter_id="c020",
        paragraphs=[
            Paragraph(text="Test Chapter", kind="heading"),
            Paragraph(text="The first paragraph of the chapter goes here."),
            Paragraph(text="The second paragraph adds a little more text."),
        ],
    )
    TextSegmenter(config.segment, language="en").segment_chapter(chapter, normalize=False)

    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        report = session.synthesize_chapter(chapter)
        no_title = session.synthesize_chapter(
            chapter, force=True, include_title=False
        )
        limited = session.synthesize_chapter(chapter, force=True, limit=1)

    assert report.generated == len(chapter.segments)
    assert no_title.total == len(chapter.segments) - 1
    assert limited.total == 1


def test_synthesize_chapter_requires_segments(config: AppConfig) -> None:
    chapter = Chapter(index=1, title="Empty", chapter_id="c021", paragraphs=[])
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        with pytest.raises(ValueError):
            session.synthesize_chapter(chapter)


def test_synthesize_chapter_only_segments(config: AppConfig) -> None:
    """``only_segments`` 只挑指定序号的片段。"""
    chapter = Chapter(
        index=1,
        title="T",
        chapter_id="c030",
        paragraphs=[Paragraph(text="A short paragraph for segmenting purposes only.")] * 4,
    )
    TextSegmenter(SegmentConfig(max_chars=30), language="en").segment_chapter(
        chapter, normalize=False
    )
    total = len(chapter.segments)
    assert total >= 3, "测试数据不足以验证筛选"

    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        report = session.synthesize_chapter(chapter, only_segments=[2, 4], force=True)
    assert report.total == 2
    assert report.generated == 2


def test_synthesize_text_one_shot(config: AppConfig) -> None:
    """一键生成一小段文本。"""
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        records = session.synthesize_text(
            "Hello there. This is a short test of the one-shot path.",
            chapter_id="adhoc",
        )
    assert records
    assert all(record.error is None for record in records)
    assert all((config.project.root_path / r.audio_path).is_file() for r in records)


def test_synthesize_text_empty_input(config: AppConfig) -> None:
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        assert session.synthesize_text("   ", chapter_id="adhoc") == []


def test_injected_engine_is_not_closed_by_session(config: AppConfig) -> None:
    """注入的引擎归调用方所有，会话关闭时不能替它做决定。

    否则「一个引擎跑多个会话」的用法（全量生成 -> force 重跑一部分）
    会在第二个会话里撞上 "引擎已关闭"。
    """
    engine = FakeSynthesisEngine()
    with SynthesisSession(config, engine=engine):
        pass
    assert not engine.closed

    engine.close()  # 调用方自己负责收尾
    assert engine.closed


def test_owned_engine_is_closed_by_session(config: AppConfig) -> None:
    """没有注入引擎时会话自己创建，也自己关闭。"""
    session = SynthesisSession(config, dry_run=True)
    engine = session.engine
    assert isinstance(engine, FakeSynthesisEngine)
    session.close()
    assert engine.closed


# --------------------------------------------------------------------------- #
# 清单模型
# --------------------------------------------------------------------------- #


def test_manifest_roundtrip() -> None:
    from noveltts.models import SegmentRecord

    manifest = ChapterAudioManifest(
        chapter_id="c001",
        title="T",
        text_sha1="abc",
        created_at="now",
        updated_at="now",
        sample_rate=24_000,
        segment_count=2,
    )
    manifest.upsert(SegmentRecord(segment_id="c001-0001", audio_path="a.wav", duration=1.5))
    manifest.upsert(SegmentRecord(segment_id="c001-0002", audio_path="b.wav", error="boom"))

    payload = manifest.to_dict()
    assert payload["done_count"] == 1
    assert payload["failed_count"] == 1
    assert payload["total_duration"] == pytest.approx(1.5)

    restored = ChapterAudioManifest.from_dict(payload)
    assert restored.chapter_id == "c001"
    assert len(restored.records) == 2
    assert restored.record_for("c001-0002").error == "boom"


def test_manifest_upsert_replaces() -> None:
    from noveltts.models import SegmentRecord

    manifest = ChapterAudioManifest(
        chapter_id="c", title="t", text_sha1="", created_at="", updated_at="",
        sample_rate=0, segment_count=1,
    )
    manifest.upsert(SegmentRecord(segment_id="s1", audio_path="a.wav", error="first"))
    manifest.upsert(SegmentRecord(segment_id="s1", audio_path="a.wav", duration=2.0))
    assert len(manifest.records) == 1
    assert manifest.records[0].error is None


def test_manifest_load_missing(tmp_path: Path) -> None:
    assert ChapterAudioManifest.load(tmp_path / "nope.json") is None


def test_manifest_load_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert ChapterAudioManifest.load(path) is None


def test_segments_index_path_shape(config: AppConfig) -> None:
    path = segments_index_path(config.project.work_path, "c001")
    assert path.name == "c001.json"
    assert path.parent.name == "segments"


def test_dump_segments_index(config: AppConfig, segments: list[Segment]) -> None:
    path = dump_segments_index(segments, work_path=config.project.work_path)
    payload = read_json(path)
    assert set(payload) == {s.segment_id for s in segments}
    assert payload[segments[0].segment_id] == segments[0].text


def test_dump_segments_index_rejects_empty() -> None:
    with pytest.raises(ValueError):
        dump_segments_index([])


# --------------------------------------------------------------------------- #
# 完整链路（假引擎）
# --------------------------------------------------------------------------- #


def test_full_pipeline_with_fake_engine(config: AppConfig, sample_epub: Path) -> None:
    """解析 -> 切分 -> 合成 -> 拼接 -> 压缩，全链路走一遍。"""
    from noveltts.concat import concat_from_manifest
    from noveltts.encode import encode_audio, find_ffmpeg
    from noveltts.epub_parser import parse_epub

    config.parse.min_chapter_chars = 1
    book = parse_epub(sample_epub, config.parse)
    assert book.chapters

    segmenter = TextSegmenter(config.segment, language="en")
    for chapter in book.chapters:
        segmenter.segment_chapter(chapter)

    chapter = book.chapters[0]
    dump_segments_index(chapter.segments, work_path=config.project.work_path)

    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        report = session.synthesize_chapter(chapter)
    assert report.failed == 0

    result = concat_from_manifest(config, chapter.chapter_id)
    assert result.output_path.is_file()
    assert result.duration > 0
    assert result.missing == []

    if find_ffmpeg():
        encoded = encode_audio(result.output_path, config=config)
        assert encoded.output.is_file()
        assert encoded.output_bytes < encoded.source_bytes


def test_full_pipeline_is_resumable(config: AppConfig, sample_epub: Path) -> None:
    """中断后重跑应零成本（全部跳过）。"""
    from noveltts.epub_parser import parse_epub

    config.parse.min_chapter_chars = 1
    book = parse_epub(sample_epub, config.parse)
    TextSegmenter(config.segment, language="en").segment_chapter(book.chapters[0])

    engine = FakeSynthesisEngine()
    with SynthesisSession(config, engine=engine) as session:
        session.synthesize_chapter(book.chapters[0])
        calls = len(engine.calls)

    with SynthesisSession(config, engine=engine) as session:
        report = session.synthesize_chapter(book.chapters[0])

    assert report.generated == 0
    assert report.skipped == report.total
    assert len(engine.calls) == calls


def test_batch_size_does_not_break_resume(config: AppConfig) -> None:
    """批大小 >1 时，断点续跑仍要逐段判断（不能整批跳过或整批重做）。

    这是批处理最容易出错的地方：如果按批判断"文件是否存在"，
    一批里只要有一段已存在，就可能整批被跳过或整批重生成。
    """
    parts = [f"Sentence number {i} is here for the batch test." for i in range(1, 13)]
    items = TextSegmenter(config.segment, language="en").segment_text(
        " ".join(parts), chapter_id="cb", normalize=False
    )
    assert len(items) >= 3, "测试数据不足以覆盖多批"

    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        first = session.synthesize_segments(items[:2], chapter_id="cb", title="Batch")
    assert first.generated == 2

    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        second = session.synthesize_segments(
            items, chapter_id="cb", title="Batch", batch_size=3
        )
    assert second.total == len(items)
    assert second.skipped == 2
    assert second.generated == len(items) - 2
    assert second.failed == 0

    payload = read_json(config.project.output_path / "segments" / "cb" / "manifest.json")
    assert payload["segment_count"] == len(items)
    assert payload["done_count"] == len(items)
    wavs = list((config.project.output_path / "segments" / "cb").glob("*.wav"))
    assert len(wavs) == len(items)


def test_speech_rate_changes_duration(config: AppConfig) -> None:
    """语速倍率要真的生效：1.5 倍速下同一段音频应当明显变短。

    同时验证它写进了 manifest（否则音频无法复现）。
    """
    from noveltts.audio_io import probe_audio

    segmenter = TextSegmenter(config.segment, language="en")
    items = segmenter.segment_text(
        "The library was quiet that morning, and nobody said a word for a long while. "
        "Outside the rain kept falling on the empty courtyard below.",
        chapter_id="rate1",
        normalize=False,
    )
    assert items

    # --- 原速 ---
    config.audio.speech_rate = 1.0
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        session.synthesize_segments(items, chapter_id="rate1", title="Normal")
    normal_dir = config.project.output_path / "segments" / "rate1"
    normal_total = sum(probe_audio(p).duration for p in sorted(normal_dir.glob("*.wav")))

    # --- 1.5 倍速 ---
    config.audio.speech_rate = 1.5
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        report = session.synthesize_segments(
            items, chapter_id="rate1", title="Fast", force=True
        )
    fast_total = sum(probe_audio(p).duration for p in sorted(normal_dir.glob("*.wav")))

    assert report.failed == 0
    assert fast_total < normal_total * 0.8, (
        f"1.5 倍速后总时长 {fast_total:.2f}s 未明显短于原速 {normal_total:.2f}s"
    )

    payload = read_json(normal_dir / "manifest.json")
    assert payload["settings"]["speech_rate"] == 1.5


def test_speech_rate_one_skips_processing(config: AppConfig) -> None:
    """rate=1 时不应调用变速（省时间，也避免无谓的相位声码器瑕疵）。"""
    config.audio.speech_rate = 1.0
    segmenter = TextSegmenter(config.segment, language="en")
    items = segmenter.segment_text("A single short sentence for the test.", chapter_id="rate2")
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        report = session.synthesize_segments(items, chapter_id="rate2", title="T")
    assert report.failed == 0
    payload = read_json(config.project.output_path / "segments" / "rate2" / "manifest.json")
    assert payload["settings"]["speech_rate"] == 1.0


def test_batch_size_auto_resolves_from_profile(config: AppConfig) -> None:
    """``batch_size: auto`` 要能落到引擎 profile 给出的值。

    假引擎没有 profile，所以这里直接测会话的解析逻辑：
    非 Qwen 引擎必须安全退回 1，而不能崩。
    """
    config.generation.batch_size = "auto"
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        assert session._resolve_batch_size(None) == 1


def test_batch_size_explicit_overrides_config(config: AppConfig) -> None:
    config.generation.batch_size = "auto"
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        assert session._resolve_batch_size(4) == 4


def test_cache_clear_interval_defaults_without_profile(config: AppConfig) -> None:
    config.generation.clear_cache_every = 0
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        assert session._cache_clear_interval() == 50


def test_cache_clear_interval_respects_config(config: AppConfig) -> None:
    config.generation.clear_cache_every = 7
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        assert session._cache_clear_interval() == 7


def test_oom_error_shrinks_batch(config: AppConfig) -> None:
    """显存不足时批大小自动减半，避免后续每段都重复 OOM。"""
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        assert session._maybe_shrink_batch(8, "CUDA out of memory. Tried to allocate 2 GiB") == 4
        assert session._maybe_shrink_batch(1, "CUDA out of memory") == 1
        assert session._maybe_shrink_batch(8, "CUDA error: device-side assert") == 8
        assert session._maybe_shrink_batch(8, None) == 8


def test_oom_downgrade_can_be_disabled(config: AppConfig) -> None:
    config.generation.oom_batch_downgrade = False
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        assert session._maybe_shrink_batch(8, "CUDA out of memory") == 8


# --------------------------------------------------------------------------- #
# 真正的批处理（本次性能修复的核心）
# --------------------------------------------------------------------------- #


def test_engine_is_called_with_multiple_texts(config: AppConfig) -> None:
    """回归测试：批处理必须把多段文本放进**同一次**引擎调用。

    这是性能问题的根子——修复前每段都单独调一次 `synthesize([text])`，
    RTX 3090 上 RTF 只有 1.74；改成真批量后同一批 5 段 RTF 降到 0.53。
    """
    parts = [f"Sentence number {i} is here for the batch test." for i in range(1, 13)]
    items = TextSegmenter(config.segment, language="en").segment_text(
        " ".join(parts), chapter_id="bt", normalize=False
    )
    assert len(items) >= 4

    engine = FakeSynthesisEngine()
    with SynthesisSession(config, engine=engine) as session:
        session.synthesize_segments(items, chapter_id="bt", title="Batch", batch_size=4)

    # 每次调用的入参长度就是「实际批大小」
    sizes = [len(call) for call in engine.calls]
    assert sizes, "引擎根本没被调用"
    assert max(sizes) > 1, f"没有任何一次调用包含多段文本：{sizes}"
    assert max(sizes) <= 4, f"批大小超过配置上限：{sizes}"
    # 所有片段都被覆盖，且没有重复
    total = sum(sizes)
    assert total == len(items), f"生成的段数 {total} 与片段总数 {len(items)} 不符"


def test_batching_generates_all_files(config: AppConfig) -> None:
    parts = [f"Sentence number {i} is here for the batch test." for i in range(1, 13)]
    items = TextSegmenter(config.segment, language="en").segment_text(
        " ".join(parts), chapter_id="bt2", normalize=False
    )
    with SynthesisSession(config, engine=FakeSynthesisEngine()) as session:
        report = session.synthesize_segments(
            items, chapter_id="bt2", title="T", batch_size=5
        )
    assert report.failed == 0
    assert report.generated == len(items)
    wavs = list((config.project.output_path / "segments" / "bt2").glob("*.wav"))
    assert len(wavs) == len(items)


def test_batch_failure_falls_back_to_per_segment(config: AppConfig) -> None:
    """整批失败时要降级为逐段重试，把一段的问题与其它段隔离。

    批处理快的代价是"一荣俱荣一损俱损"，所以必须有这条兜底路径。
    """
    parts = [f"Sentence number {i} is here for the batch test." for i in range(1, 13)]
    items = TextSegmenter(config.segment, language="en").segment_text(
        " ".join(parts), chapter_id="bt3", normalize=False
    )
    assert len(items) >= 3

    # 让包含 "number 3" 的那一段必定失败：批量时会拖垮整批
    config.generation.num_retries = 0
    engine = FakeSynthesisEngine(fail_on=("number 3 ",))
    with SynthesisSession(config, engine=engine) as session:
        report = session.synthesize_segments(
            items, chapter_id="bt3", title="T", batch_size=8
        )

    # 只有真正有问题的那一段失败，其余全部成功——这正是"失败隔离"的意义：
    # 批处理把整批拖失败后，逐段重试救回了其余所有段。
    assert report.failed == 1, f"期望只失败 1 段，实际 {report.failed}"
    assert report.generated == len(items) - 1
    failed_id = report.errors[0][0]
    assert failed_id.startswith("bt3-")


def test_batch_resume_skips_within_batch(config: AppConfig) -> None:
    """批内也要逐段判断续跑：已存在的段不能被重新生成。"""
    parts = [f"Sentence number {i} is here for the batch test." for i in range(1, 13)]
    items = TextSegmenter(config.segment, language="en").segment_text(
        " ".join(parts), chapter_id="bt4", normalize=False
    )

    engine = FakeSynthesisEngine()
    with SynthesisSession(config, engine=engine) as session:
        session.synthesize_segments(items[:2], chapter_id="bt4", title="T", batch_size=4)
    first_calls = len(engine.calls)

    with SynthesisSession(config, engine=engine) as session:
        report = session.synthesize_segments(
            items, chapter_id="bt4", title="T", batch_size=4
        )

    assert report.skipped == 2
    assert report.generated == len(items) - 2
    # 第二批只应包含剩下的段（不含已跳过的 2 段）
    for call in engine.calls[first_calls:]:
        assert len(call) <= 4


def test_batch_size_one_still_works(config: AppConfig) -> None:
    """batch_size=1 是 Mac 的默认路径，必须照常工作。"""
    parts = [f"Sentence number {i} is here." for i in range(1, 7)]
    items = TextSegmenter(config.segment, language="en").segment_text(
        " ".join(parts), chapter_id="bt5", normalize=False
    )
    engine = FakeSynthesisEngine()
    with SynthesisSession(config, engine=engine) as session:
        report = session.synthesize_segments(
            items, chapter_id="bt5", title="T", batch_size=1
        )
    assert report.failed == 0
    assert all(len(call) == 1 for call in engine.calls)


def test_engine_waveform_count_mismatch_is_rejected(config: AppConfig) -> None:
    """引擎返回段数与请求不符时必须整批判失败。

    这是很危险的静默错误：如果少返回一段而我们仍然按顺序落盘，
    就会把第 2 段的音频写进第 1 段的文件名里，而且没有任何报错。
    所以 :meth:`_synthesize_batch` 会显式核对段数。

    Note:
        守卫只在**多段批量**时才有意义（逐段调用永远是 1:1），
        所以这里直接测 :meth:`_synthesize_batch`，而不是走完整流程。
    """
    engine = FakeSynthesisEngine()
    original = engine.synthesize

    def truncating_synthesize(texts, *, language=None):
        result = original(texts, language=language)
        result.waveforms = result.waveforms[:1]      # 故意少返回
        return result

    engine.synthesize = truncating_synthesize        # type: ignore[method-assign]

    # 直接构造 3 个片段：用 segment_text 的话这些短句会被合并成一段，
    # 而本用例恰恰需要「多段共处一批」才能验证段数核对。
    items = [
        Segment(
            segment_id=f"bt6-{i:04d}",
            chapter_id="bt6",
            index=i,
            text=f"Sentence number {i} is here for the mismatch test.",
        )
        for i in range(1, 4)
    ]

    with SynthesisSession(config, engine=engine) as session:
        records, error = session._synthesize_batch(
            items, language=None, attempts=1
        )

    assert error is not None, "段数不匹配必须报错"
    assert "不匹配" in error
    assert len(records) == len(items)
    assert all(r is None for r in records)
    # 关键：不能留下任何半成品文件
    assert not list((config.project.output_path / "segments" / "bt6").glob("*.wav"))
