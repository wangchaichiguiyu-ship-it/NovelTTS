"""音频测试：响度归一、静音裁切、流式拼接、压缩命令构造。

**本文件不加载任何模型**：所有音频都是现场合成的正弦波 / 噪声。
需要 numpy / soundfile 的用例用 ``importorskip`` 守护。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noveltts import AUDIO_DTYPE, SAMPLE_RATE
from noveltts.audio_io import (
    apply_fade,
    audio_available,
    db_to_linear,
    iter_audio_blocks,
    join_gap_samples,
    linear_to_db,
    normalize_audio,
    peak_dbfs,
    probe_audio,
    read_audio,
    rms_dbfs,
    tail_is_quiet,
    trim_silence,
    write_audio,
)
from noveltts.config import AppConfig, AudioConfig, EncodeConfig, ParseConfig, ProjectConfig
from noveltts.encode import (
    _bitrate_to_bps,
    build_afconvert_command,
    build_ffmpeg_command,
    encode_audio,
    encoder_available,
    find_ffmpeg,
)

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")


# --------------------------------------------------------------------------- #
# 工具函数（纯数学，无需依赖）
# --------------------------------------------------------------------------- #


def test_db_conversions() -> None:
    assert db_to_linear(0) == pytest.approx(1.0)
    assert db_to_linear(-6) == pytest.approx(0.501, abs=0.01)
    assert linear_to_db(1.0) == pytest.approx(0.0)
    assert linear_to_db(0) == float("-inf")


def test_bitrate_parsing() -> None:
    assert _bitrate_to_bps("64k") == 64_000
    assert _bitrate_to_bps("128K") == 128_000
    assert _bitrate_to_bps("1m") == 1_000_000
    assert _bitrate_to_bps("96000") == 96_000
    assert _bitrate_to_bps("garbage") == 64_000


# --------------------------------------------------------------------------- #
# 音频度量与归一化
# --------------------------------------------------------------------------- #


def make_tone(seconds: float = 1.0, amplitude: float = 0.5, sr: int = SAMPLE_RATE):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_peak_and_rms_dbfs() -> None:
    tone = make_tone(amplitude=0.5)
    assert peak_dbfs(tone) == pytest.approx(-6.02, abs=0.1)
    assert rms_dbfs(tone) == pytest.approx(-9.03, abs=0.1)


def test_dbfs_of_empty_is_neg_inf() -> None:
    empty = np.zeros(0, dtype=np.float32)
    assert peak_dbfs(empty) == float("-inf")
    assert rms_dbfs(empty) == float("-inf")


def test_normalize_peak_reaches_target() -> None:
    """增益在 max_gain_db 范围内时应精确命中目标峰值。"""
    config = AudioConfig(normalize="peak", target_dbfs=-3.0, max_gain_db=20.0)
    result = normalize_audio(make_tone(amplitude=0.1), config)
    assert peak_dbfs(result) == pytest.approx(-3.0, abs=0.2)


def test_normalize_skips_digital_silence() -> None:
    """纯静音不能被“放大”——否则会把量化噪声抬起来。"""
    config = AudioConfig(normalize="peak", target_dbfs=-1.5, max_gain_db=40.0)
    silent = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    result = normalize_audio(silent, config)
    assert not result.any()


def test_normalize_never_clips() -> None:
    """即使目标峰值很激进，也不能把样本推过 1.0。"""
    config = AudioConfig(normalize="peak", target_dbfs=6.0, max_gain_db=40)
    result = normalize_audio(make_tone(amplitude=0.5), config)
    assert float(np.max(np.abs(result))) <= 1.0


def test_normalize_respects_max_gain() -> None:
    """增益上限必须生效，否则会把底噪也放大。"""
    config = AudioConfig(normalize="peak", target_dbfs=-1.5, max_gain_db=3.0)
    quiet = make_tone(amplitude=0.001)
    result = normalize_audio(quiet, config)
    gain = float(np.max(np.abs(result)) / np.max(np.abs(quiet)))
    assert gain <= db_to_linear(3.0) + 1e-3


def test_normalize_none_is_identity() -> None:
    config = AudioConfig(normalize="none")
    tone = make_tone()
    assert np.array_equal(normalize_audio(tone, config), tone)


def test_normalize_empty_is_safe() -> None:
    config = AudioConfig(normalize="peak")
    empty = np.zeros(0, dtype=np.float32)
    assert len(normalize_audio(empty, config)) == 0


def test_normalize_rms_mode() -> None:
    config = AudioConfig(normalize="rms", target_rms_dbfs=-18.0)
    result = normalize_audio(make_tone(amplitude=0.05), config)
    assert rms_dbfs(result) == pytest.approx(-18.0, abs=1.0)


# --------------------------------------------------------------------------- #
# 裁静音 / 尾部检测 / 淡化
# --------------------------------------------------------------------------- #


def test_trim_silence_removes_padding() -> None:
    sr = SAMPLE_RATE
    silence = np.zeros(sr // 2, dtype=np.float32)
    loud = make_tone(0.5, amplitude=0.6)
    padded = np.concatenate([silence, loud, silence])
    trimmed = trim_silence(padded, threshold_dbfs=-50, sample_rate=sr, keep_ms=10)
    assert len(trimmed) < len(padded)
    assert peak_dbfs(trimmed) > -10


def test_trim_silence_keeps_all_silent_input() -> None:
    """整段静音时不能产出空数组（会让下游写文件失败）。"""
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert len(trim_silence(silent, sample_rate=SAMPLE_RATE)) == len(silent)


def test_tail_is_quiet_detects_silence() -> None:
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    assert tail_is_quiet(silence, sample_rate=SAMPLE_RATE) is True


def test_tail_is_quiet_detects_sound() -> None:
    assert tail_is_quiet(make_tone(1.0, amplitude=0.5), sample_rate=SAMPLE_RATE) is False


def test_tail_is_quiet_on_empty() -> None:
    assert tail_is_quiet(np.zeros(0, dtype=np.float32), sample_rate=SAMPLE_RATE) is True


def test_apply_fade_edges_are_zero() -> None:
    tone = np.ones(1000, dtype=np.float32)
    faded = apply_fade(tone, fade_in_samples=100, fade_out_samples=100)
    assert faded[0] == pytest.approx(0.0)
    assert faded[-1] == pytest.approx(0.0)
    assert faded[500] == pytest.approx(1.0)
    # 原数组不能被修改
    assert tone[0] == 1.0


def test_join_gap_samples_length() -> None:
    config = AudioConfig(min_gap_seconds=0.5, sample_rate=24_000)
    gap = join_gap_samples(config)
    assert len(gap) == 12_000
    assert not gap.any()


def test_join_gap_scale() -> None:
    config = AudioConfig(min_gap_seconds=0.4, sample_rate=24_000)
    assert len(join_gap_samples(config, scale=0.5)) == 4_800


# --------------------------------------------------------------------------- #
# 文件读写
# --------------------------------------------------------------------------- #


def test_write_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    tone = make_tone(0.5)
    write_audio(path, tone, SAMPLE_RATE)
    samples, sr = read_audio(path)
    assert sr == SAMPLE_RATE
    assert abs(len(samples) - len(tone)) <= 1
    assert peak_dbfs(samples) == pytest.approx(peak_dbfs(tone), abs=0.2)


def test_write_audio_creates_directories(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "a.wav"
    write_audio(path, make_tone(0.1), SAMPLE_RATE)
    assert path.is_file()


def test_written_file_is_pcm16(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    write_audio(path, make_tone(0.1), SAMPLE_RATE)
    info = probe_audio(path)
    assert info.subtype == AUDIO_DTYPE
    assert info.channels == 1
    assert info.sample_rate == SAMPLE_RATE


def test_probe_audio_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_audio(tmp_path / "nope.wav")


def test_iter_audio_blocks_covers_file(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    tone = make_tone(2.0)
    write_audio(path, tone, SAMPLE_RATE)
    blocks = list(iter_audio_blocks(path, block_seconds=0.25))
    assert len(blocks) > 1
    assert sum(len(b) for b in blocks) == len(tone)


def test_iter_audio_blocks_block_size() -> None:
    """块大小必须受控，这是低内存拼接的关键。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.wav"
        write_audio(path, make_tone(3.0), SAMPLE_RATE)
        for block in iter_audio_blocks(path, block_seconds=0.5):
            assert len(block) <= SAMPLE_RATE * 0.5 + 1


def test_audio_available_reports_ok() -> None:
    ok, detail = audio_available()
    assert ok, detail


# --------------------------------------------------------------------------- #
# 压缩：命令构造（不真正编码）
# --------------------------------------------------------------------------- #


def test_ffmpeg_command_m4a() -> None:
    command = build_ffmpeg_command(
        "in.wav", "out.m4a", EncodeConfig(format="m4a", bitrate="64k", sample_rate=24000, channels=1)
    )
    assert command[0] == "ffmpeg"
    assert "-c:a" in command and "aac" in command
    assert "-b:a" in command and "64k" in command
    assert "-ar" in command and "24000" in command
    assert "-ac" in command and "1" in command
    assert command[-1] == "out.m4a"
    assert command[-3] == "in.wav" or "in.wav" in command


def test_ffmpeg_command_mp3_and_opus() -> None:
    mp3 = build_ffmpeg_command("a.wav", "a.mp3", EncodeConfig(format="mp3", bitrate="96k"))
    assert "libmp3lame" in mp3
    opus = build_ffmpeg_command("a.wav", "a.opus", EncodeConfig(format="opus", bitrate="32k"))
    assert "libopus" in opus


def test_ffmpeg_command_flac_ignores_bitrate() -> None:
    command = build_ffmpeg_command("a.wav", "a.flac", EncodeConfig(format="flac", bitrate="64k"))
    assert "flac" in command
    assert "-b:a" not in command


def test_ffmpeg_command_copy_mode_keeps_stream() -> None:
    command = build_ffmpeg_command("a.wav", "a.m4a", EncodeConfig(format="copy"))
    assert "-c:a" in command and "copy" in command


def test_ffmpeg_command_is_list_not_shell_string() -> None:
    """必须是参数列表，避免文件名里的空格/引号出问题。"""
    command = build_ffmpeg_command(
        "my file with spaces.wav", "out put.m4a", EncodeConfig()
    )
    assert isinstance(command, list)
    assert all(isinstance(part, str) for part in command)
    assert "my file with spaces.wav" in command


def test_ffmpeg_command_includes_extra_args() -> None:
    config = EncodeConfig(extra_args=["-vn", "-map_metadata", "-1"])
    command = build_ffmpeg_command("a.wav", "a.m4a", config)
    assert "-vn" in command
    assert command.index("-vn") < command.index("a.m4a")


def test_afconvert_command() -> None:
    command = build_afconvert_command("a.wav", "a.m4a", EncodeConfig(bitrate="64k", channels=1))
    assert command[0] == "afconvert"
    assert "m4af" in command and "aac" in command
    assert "64000" in command


def test_encoder_available_has_result() -> None:
    ok, detail = encoder_available(EncodeConfig())
    assert isinstance(ok, bool)
    assert detail


def test_find_ffmpeg_returns_path_or_none() -> None:
    found = find_ffmpeg()
    assert found is None or Path(found).is_file()


@pytest.mark.skipif(find_ffmpeg() is None, reason="未安装 ffmpeg")
def test_encode_audio_end_to_end(tmp_path: Path) -> None:
    """真跑一次 ffmpeg（输入是合成正弦波，不涉及模型）。"""
    source = tmp_path / "chapter.wav"
    write_audio(source, make_tone(1.0), SAMPLE_RATE)

    result = encode_audio(source, config=None, encode_config=EncodeConfig(format="m4a", bitrate="48k"))
    assert result.output.is_file()
    assert result.output_bytes > 0
    assert result.output_bytes < result.source_bytes
    assert result.encoder == "ffmpeg"
    assert "->" in result.summary()


@pytest.mark.skipif(find_ffmpeg() is None, reason="未安装 ffmpeg")
def test_encode_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        encode_audio(tmp_path / "nope.wav", encode_config=EncodeConfig())


# --------------------------------------------------------------------------- #
# 拼接
# --------------------------------------------------------------------------- #


def build_config(tmp_path: Path, **audio_kwargs) -> AppConfig:
    config = AppConfig(
        project=ProjectConfig(root=str(tmp_path), work_dir="w", output_dir="o", cache_dir="c"),
        parse=ParseConfig(),
        audio=AudioConfig(**audio_kwargs),
    )
    config.project.resolve(tmp_path)
    config.project.ensure_dirs()
    return config


def test_concat_chapter_streams_and_gaps(tmp_path: Path) -> None:
    from noveltts.concat import concat_chapter
    from noveltts.models import Segment

    config = build_config(tmp_path, min_gap_seconds=0.2, join_fade_ms=5.0)
    segments = []
    paths = []
    for index in range(1, 4):
        segment = Segment(
            segment_id=f"c001-{index:04d}",
            chapter_id="c001",
            index=index,
            text=f"Segment number {index} ends here.",
        )
        path = tmp_path / f"{segment.segment_id}.wav"
        write_audio(path, make_tone(0.5), SAMPLE_RATE)
        segments.append(segment)
        paths.append(path)

    result = concat_chapter(
        config, segments, segment_paths=paths, tail_silence_seconds=0.1
    )
    assert result.output_path.is_file()
    assert result.segment_count == 3
    assert result.missing == []
    # 3 段 × 0.5s + 2 个间隔 + 尾部静音
    assert result.duration == pytest.approx(1.5 + 0.4 + 0.1, abs=0.15)

    info = probe_audio(result.output_path)
    assert info.sample_rate == SAMPLE_RATE
    assert info.channels == 1


def test_concat_reports_missing_segments(tmp_path: Path) -> None:
    from noveltts.concat import concat_chapter
    from noveltts.models import Segment

    config = build_config(tmp_path)
    segments = [
        Segment(segment_id="c001-0001", chapter_id="c001", index=1, text="Present."),
        Segment(segment_id="c001-0002", chapter_id="c001", index=2, text="Missing."),
    ]
    paths = [tmp_path / "present.wav", tmp_path / "missing.wav"]
    write_audio(paths[0], make_tone(0.3), SAMPLE_RATE)

    result = concat_chapter(config, segments, segment_paths=paths)
    assert result.missing == ["c001-0002"]
    assert result.segment_count == 1


def test_concat_raises_when_nothing_available(tmp_path: Path) -> None:
    from noveltts.concat import concat_chapter
    from noveltts.models import Segment

    config = build_config(tmp_path)
    segments = [Segment(segment_id="c001-0001", chapter_id="c001", index=1, text="x")]
    with pytest.raises(RuntimeError):
        concat_chapter(config, segments, segment_paths=[tmp_path / "nope.wav"])


def test_concat_rejects_empty_segments(tmp_path: Path) -> None:
    from noveltts.concat import concat_chapter

    config = build_config(tmp_path)
    with pytest.raises(ValueError):
        concat_chapter(config, [])


def test_concat_detects_length_mismatch(tmp_path: Path) -> None:
    from noveltts.concat import concat_chapter
    from noveltts.models import Segment

    config = build_config(tmp_path)
    segments = [Segment(segment_id="c001-0001", chapter_id="c001", index=1, text="x")]
    with pytest.raises(ValueError):
        concat_chapter(config, segments, segment_paths=[tmp_path / "a.wav", tmp_path / "b.wav"])


def test_concat_shorter_gap_when_tail_already_quiet(tmp_path: Path) -> None:
    """片段自带尾部静音时，插入的间隔应当更短。"""
    from noveltts.concat import _build_gaps
    from noveltts.models import Segment

    config = build_config(tmp_path, min_gap_seconds=0.4, tail_quiet_gain=0.25)

    loud = tmp_path / "loud.wav"
    write_audio(loud, make_tone(0.5, amplitude=0.5), SAMPLE_RATE)
    quiet_tail = tmp_path / "quiet.wav"
    write_audio(
        quiet_tail,
        np.concatenate([make_tone(0.4, amplitude=0.5), np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)]),
        SAMPLE_RATE,
    )

    segments = [
        Segment(segment_id="c001-0001", chapter_id="c001", index=1, text="Loud one."),
        Segment(segment_id="c001-0002", chapter_id="c001", index=2, text="Quiet tail."),
        Segment(segment_id="c001-0003", chapter_id="c001", index=3, text="Last one."),
    ]
    infos = [probe_audio(loud), probe_audio(quiet_tail), probe_audio(loud)]
    gaps = _build_gaps(
        config,
        list(zip(segments, [loud, quiet_tail, loud])),
        infos,
        base_gap=0.4,
        sample_rate=SAMPLE_RATE,
    )
    assert gaps[-1] is None            # 最后一段之后不插
    assert len(gaps[0]) == int(SAMPLE_RATE * 0.4)   # 尾部有声音 -> 完整间隔
    assert len(gaps[1]) < len(gaps[0])              # 尾部安静 -> 更短


def test_concat_shorter_gap_for_unfinished_sentence(tmp_path: Path) -> None:
    """片段不以句末标点结束时，间隔要缩短（语流更连贯）。

    Note:
        需要**至少三段**：最后一段之后不插间隔，所以两段时只剩一个间隔，
        无法同时对比「完整句」与「被切断的句子」。
    """
    from noveltts.concat import _build_gaps
    from noveltts.models import Segment

    config = build_config(tmp_path, min_gap_seconds=0.4)
    path = tmp_path / "a.wav"
    write_audio(path, make_tone(0.5, amplitude=0.5), SAMPLE_RATE)

    finished = Segment(segment_id="c001-0001", chapter_id="c001", index=1, text="Done.")
    unfinished = Segment(segment_id="c001-0002", chapter_id="c001", index=2, text="Not done")
    last = Segment(segment_id="c001-0003", chapter_id="c001", index=3, text="Last one.")
    infos = [probe_audio(path)] * 3

    gaps = _build_gaps(
        config,
        [(finished, path), (unfinished, path), (last, path)],
        infos,
        base_gap=0.4,
        sample_rate=SAMPLE_RATE,
    )
    assert len(gaps[0]) == int(SAMPLE_RATE * 0.4)   # 完整句 -> 完整间隔
    assert len(gaps[1]) < len(gaps[0])              # 被切断 -> 更短
    assert gaps[2] is None


def test_concat_from_manifest(tmp_path: Path) -> None:
    """按 manifest 拼接（不重新解析 EPUB）。"""
    from noveltts.concat import concat_from_manifest
    from noveltts.models import Segment
    from noveltts.synthesis import segments_index_path
    from noveltts.utils import ensure_dir, write_json

    config = build_config(tmp_path)
    chapter_dir = ensure_dir(config.project.output_path / "segments" / "c001")

    records = []
    for index in (1, 2):
        segment_id = f"c001-{index:04d}"
        path = chapter_dir / f"{segment_id}.wav"
        write_audio(path, make_tone(0.3), SAMPLE_RATE)
        records.append({
            "segment_id": segment_id,
            "audio_path": str(path.relative_to(config.project.root_path)),
            "duration": 0.3,
            "sample_rate": SAMPLE_RATE,
            "attempts": 1,
            "skipped": False,
            "error": None,
        })

    write_json(
        chapter_dir / "manifest.json",
        {
            "chapter_id": "c001",
            "title": "Test Chapter",
            "text_sha1": "x",
            "created_at": "now",
            "updated_at": "now",
            "sample_rate": SAMPLE_RATE,
            "segment_count": 2,
            "records": records,
            "settings": {},
        },
    )
    index_path = segments_index_path(config.project.work_path, "c001")
    ensure_dir(index_path.parent)
    write_json(index_path, {"c001-0001": "First one.", "c001-0002": "Second one."})

    result = concat_from_manifest(config, "c001")
    assert result.output_path.is_file()
    assert result.segment_count == 2
    assert result.duration > 0.5


def test_concat_from_manifest_missing(tmp_path: Path) -> None:
    from noveltts.concat import concat_from_manifest

    config = build_config(tmp_path)
    with pytest.raises(FileNotFoundError):
        concat_from_manifest(config, "c999")


def test_discover_chapters(tmp_path: Path) -> None:
    from noveltts.concat import discover_chapters
    from noveltts.utils import ensure_dir, write_json

    config = build_config(tmp_path)
    for chapter_id in ("c002", "c001"):
        directory = ensure_dir(config.project.output_path / "segments" / chapter_id)
        write_json(directory / "manifest.json", {"chapter_id": chapter_id})
    # 没有 manifest 的目录不算
    ensure_dir(config.project.output_path / "segments" / "c003")

    assert discover_chapters(config) == ["c001", "c002"]


def test_concat_all_chapters_merges_book(tmp_path: Path) -> None:
    from noveltts.concat import concat_all_chapters
    from noveltts.models import Segment
    from noveltts.synthesis import dump_segments_index
    from noveltts.utils import ensure_dir, write_json

    config = build_config(tmp_path)
    for chapter_id in ("c001", "c002"):
        directory = ensure_dir(config.project.output_path / "segments" / chapter_id)
        segment = Segment(
            segment_id=f"{chapter_id}-0001", chapter_id=chapter_id, index=1, text="One segment."
        )
        path = directory / f"{segment.segment_id}.wav"
        write_audio(path, make_tone(0.3), SAMPLE_RATE)
        write_json(
            directory / "manifest.json",
            {
                "chapter_id": chapter_id,
                "title": chapter_id,
                "text_sha1": "x",
                "created_at": "n",
                "updated_at": "n",
                "sample_rate": SAMPLE_RATE,
                "segment_count": 1,
                "records": [{
                    "segment_id": segment.segment_id,
                    "audio_path": str(path.relative_to(config.project.root_path)),
                    "duration": 0.3, "sample_rate": SAMPLE_RATE,
                    "attempts": 1, "skipped": False, "error": None,
                }],
                "settings": {},
            },
        )
        dump_segments_index([segment], work_path=config.project.work_path)

    results = concat_all_chapters(config, merge_into_single=True, output_name="book")
    assert len(results) == 3          # 两章 + 整本
    book = results[-1]
    assert book.chapter_id == "book"
    assert book.output_path.name == "book.wav"
    assert book.duration > 0.6


# --------------------------------------------------------------------------- #
# 语速调节（变速不变调）
# --------------------------------------------------------------------------- #


def _dominant_freq(samples, sr: int) -> float:
    """用 FFT 找主频，用来验证变速后音高不变。"""
    window = min(8192, len(samples))
    if window < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(samples[:window] * np.hanning(window)))
    return float(np.fft.rfftfreq(window, 1 / sr)[int(np.argmax(spec))])


def test_time_stretch_identity_when_rate_is_one() -> None:
    """rate=1 必须原样返回（不做任何计算，省时间也避免精度损失）。"""
    from noveltts.audio_io import time_stretch

    tone = make_tone(0.5)
    assert time_stretch(tone, rate=1.0, sample_rate=SAMPLE_RATE) is tone


def test_time_stretch_shortens_and_lengthens() -> None:
    from noveltts.audio_io import time_stretch

    tone = make_tone(1.0)
    fast = time_stretch(tone, rate=1.25, sample_rate=SAMPLE_RATE)
    slow = time_stretch(tone, rate=0.8, sample_rate=SAMPLE_RATE)
    assert len(fast) < len(tone) < len(slow)
    # 1.25 倍速 -> 约 0.8 倍时长
    assert len(fast) == pytest.approx(len(tone) / 1.25, rel=0.1)
    assert len(slow) == pytest.approx(len(tone) / 0.8, rel=0.1)


def test_time_stretch_preserves_pitch() -> None:
    """核心保证：变速**不变调**。

    这是"为什么不用简单重采样"的回归测试——重采样会把 220Hz 抬成 275Hz。
    """
    from noveltts.audio_io import time_stretch

    tone = make_tone(1.5)
    original = _dominant_freq(tone, SAMPLE_RATE)
    assert original == pytest.approx(220.0, abs=5.0)

    for rate in (0.8, 1.25, 1.5):
        stretched = time_stretch(tone, rate=rate, sample_rate=SAMPLE_RATE)
        pitch = _dominant_freq(stretched, SAMPLE_RATE)
        assert pitch == pytest.approx(original, rel=0.03), (
            f"rate={rate} 时主频从 {original:.1f}Hz 变成 {pitch:.1f}Hz，音高被改变了"
        )


def test_time_stretch_rejects_non_positive_rate() -> None:
    from noveltts.audio_io import time_stretch

    with pytest.raises(ValueError):
        time_stretch(make_tone(0.1), rate=0.0)
    with pytest.raises(ValueError):
        time_stretch(make_tone(0.1), rate=-1.5)


def test_time_stretch_handles_empty() -> None:
    from noveltts.audio_io import time_stretch

    empty = np.zeros(0, dtype=np.float32)
    assert len(time_stretch(empty, rate=1.5, sample_rate=SAMPLE_RATE)) == 0


def test_time_stretch_output_is_float32() -> None:
    from noveltts.audio_io import time_stretch

    out = time_stretch(make_tone(0.3), rate=1.2, sample_rate=SAMPLE_RATE)
    assert out.dtype == np.float32


def test_speed_factor_validation() -> None:
    from noveltts.audio_io import speed_factor_to_rate

    assert speed_factor_to_rate(1.25) == 1.25
    assert speed_factor_to_rate(0.8) == 0.8
    with pytest.raises(ValueError):
        speed_factor_to_rate(0)
