"""配置与通用工具测试。

重点覆盖实际踩过的坑：

* ``from __future__ import annotations`` 让 ``dataclasses.fields().type`` 变成
  字符串，导致嵌套配置保持成 dict（``config.project.resolve`` 直接 AttributeError）；
* YAML 缺失时内置极简解析器必须能读完整份 config.yaml。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noveltts.config import (
    AppConfig,
    AudioConfig,
    EncodeConfig,
    GenerationConfig,
    ProjectConfig,
    SegmentConfig,
    TextRulesConfig,
)
from noveltts.utils import (
    chunked,
    count_chars,
    count_words,
    deep_merge,
    ensure_dir,
    estimate_speech_seconds,
    format_duration,
    human_size,
    read_json,
    read_text,
    resolve_path,
    sha1_of,
    slugify,
    write_json,
    write_text,
)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (5, "00:05"), (65, "01:05"), (3600, "1:00:00"), (3725, "1:02:05")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB")],
)
def test_human_size(size: int, expected: str) -> None:
    assert human_size(size) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Loss 3: Crimson Rhapsody", "loss-3-crimson-rhapsody"),
        ("Intermission: I'll Have You Know", "intermission-ill-have-you-know"),
        ("", "chapter"),
        ("!!!", "chapter"),
    ],
)
def test_slugify(value: str, expected: str) -> None:
    assert slugify(value) == expected


def test_slugify_truncates() -> None:
    assert len(slugify("word " * 50, max_length=20)) <= 20


def test_count_words_mixed() -> None:
    assert count_words("hello world") == 2
    assert count_words("你好世界") == 4
    assert count_words("hello 世界") == 3


def test_count_chars_ignores_punctuation() -> None:
    assert count_chars("hi, there!") == 7
    assert count_chars("...") == 0


def test_estimate_speech_seconds_positive() -> None:
    assert estimate_speech_seconds("hello world") > 0


def test_sha1_is_stable_and_short() -> None:
    assert sha1_of("abc") == sha1_of("abc")
    assert len(sha1_of("abc")) == 12
    assert sha1_of("abc") != sha1_of("abd")


def test_chunked() -> None:
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunked([], 2)) == []


def test_chunked_rejects_zero() -> None:
    with pytest.raises(ValueError):
        list(chunked([1], 0))


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 9}, "c": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}


def test_resolve_path_expands_user() -> None:
    resolved = resolve_path("~/x", base="/tmp")
    assert resolved.is_absolute()
    assert "~" not in str(resolved)


# --------------------------------------------------------------------------- #
# 文件读写
# --------------------------------------------------------------------------- #


def test_write_and_read_text_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "a.txt"
    write_text(path, "你好\nworld\n")
    assert read_text(path) == "你好\nworld\n"


def test_read_text_strips_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhello")
    assert read_text(path) == "hello"


def test_write_text_is_atomic(tmp_path: Path) -> None:
    """写入过程中不留半截文件，也不留临时文件。"""
    path = tmp_path / "a.txt"
    write_text(path, "first")
    write_text(path, "second")
    assert read_text(path) == "second"
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def test_json_roundtrip_unicode(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    payload = {"title": "标题", "n": 3}
    write_json(path, payload)
    assert read_json(path) == payload
    assert "标题" in path.read_text(encoding="utf-8")  # 不被转义成 \uXXXX


def test_read_json_missing_returns_default(tmp_path: Path) -> None:
    assert read_json(tmp_path / "nope.json", default={"a": 1}) == {"a": 1}


def test_ensure_dir_idempotent(tmp_path: Path) -> None:
    first = ensure_dir(tmp_path / "x" / "y")
    second = ensure_dir(tmp_path / "x" / "y")
    assert first == second and first.is_dir()


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


def test_config_defaults_are_typed() -> None:
    """回归测试：嵌套配置必须构造成 dataclass，不能是 dict。"""
    config = AppConfig.from_dict({})
    assert isinstance(config.project, ProjectConfig)
    assert isinstance(config.text_rules, TextRulesConfig)
    assert isinstance(config.generation, GenerationConfig)
    assert isinstance(config.audio, AudioConfig)
    assert isinstance(config.encode, EncodeConfig)


def test_config_load_without_file_resolves_root(tmp_path: Path) -> None:
    config = AppConfig.load(None, root=tmp_path)
    assert config.project.root_path == tmp_path.resolve()
    assert config.project.output_path == (tmp_path / "output/audio").resolve()


def test_config_load_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "generation:\n  device: cpu\n  batch_size: 2\n"
        "segment:\n  max_chars: 90\n"
        "voice:\n  ref_audio: ref.wav\n",
        encoding="utf-8",
    )
    config = AppConfig.load(path)
    assert config.project.root_path == tmp_path.resolve()
    assert config.generation.device == "cpu"
    assert config.generation.batch_size == 2
    assert config.segment.max_chars == 90
    assert config.voice.ref_audio == "ref.wav"
    # 未在文件里出现的字段应保留默认值
    assert config.segment.min_chars == 24


def test_config_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AppConfig.load(tmp_path / "nope.yaml")


def test_config_overrides_win(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("generation:\n  device: cpu\n", encoding="utf-8")
    config = AppConfig.load(path, overrides={"generation": {"device": "mps"}})
    assert config.generation.device == "mps"


def test_config_unknown_key_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("generation:\n  device: cpu\nbogus:\n  a: 1\n", encoding="utf-8")
    config = AppConfig.load(path)
    assert config.generation.device == "cpu"


def test_config_none_section_falls_back_to_default() -> None:
    config = AppConfig.from_dict({"segment": None})
    assert isinstance(config.segment, SegmentConfig)


def test_config_rejects_non_mapping_section() -> None:
    config = AppConfig.from_dict({"segment": [1, 2, 3]})
    assert isinstance(config.segment, SegmentConfig)


def test_config_summary_contains_key_facts() -> None:
    summary = AppConfig.from_dict({}).summary()
    assert "模型" in summary
    assert "切分" in summary
    assert "音频" in summary


def test_mini_yaml_parser_handles_nested_and_lists() -> None:
    """没装 PyYAML 时，内置解析器要能读完整份配置。"""
    from noveltts.config import _mini_yaml_load

    text = """
project:
  source_dir: sources

parse:
  skip_title_patterns:
    - "^table of contents"
    - "^copyright"

generation:
  batch_size: 1
  temperature: null
  progress_every: 10

text_rules:
  abbreviations:
    "Mr.": "Mister"
"""
    data = _mini_yaml_load(text)
    assert data["project"]["source_dir"] == "sources"
    assert data["parse"]["skip_title_patterns"] == ["^table of contents", "^copyright"]
    assert data["generation"]["batch_size"] == 1
    assert data["generation"]["temperature"] is None
    assert data["text_rules"]["abbreviations"] == {"Mr.": "Mister"}


def test_mini_yaml_parser_handles_inline_containers() -> None:
    from noveltts.config import _mini_yaml_load

    data = _mini_yaml_load('a: [1, 2, 3]\nb: {x: 1, y: "two"}\nc: true\n')
    assert data["a"] == [1, 2, 3]
    assert data["b"] == {"x": 1, "y": "two"}
    assert data["c"] is True


def test_shipped_config_yaml_is_loadable(project_root: Path) -> None:
    """仓库里那份 config.yaml 必须能被完整加载。"""
    path = project_root / "config.yaml"
    assert path.is_file(), "缺少 config.yaml"
    config = AppConfig.load(path)
    assert config.generation.model_path
    assert config.voice.ref_audio
    assert config.text_rules.abbreviations
    assert config.segment.max_chars > config.segment.min_chars


def test_shipped_config_paths_are_sane(project_root: Path) -> None:
    config = AppConfig.load(project_root / "config.yaml")
    assert config.project.root_path == project_root.resolve()
    assert (config.project.root_path / config.voice.ref_audio).is_file(), "参考音频不存在"
    assert (config.project.root_path / config.generation.model_path).is_dir(), "本地模型目录不存在"


# --------------------------------------------------------------------------- #
# numba 缓存自动修复（librosa 导入失败的真实故障）
# --------------------------------------------------------------------------- #


def test_numba_cache_dir_respects_existing_env(monkeypatch) -> None:
    """用户已显式设置时不能覆盖。"""
    from noveltts import ensure_numba_cache_dir

    monkeypatch.setenv("NUMBA_CACHE_DIR", "/tmp/user-choice")
    assert ensure_numba_cache_dir() is None


def test_numba_cache_dir_skips_when_no_librosa(monkeypatch) -> None:
    """没装 librosa 的环境（只跑 parse/segment）不该被干扰。"""
    import importlib.util
    import sys

    from noveltts import ensure_numba_cache_dir

    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    # 让「librosa 不存在」这件事对 find_spec 成立
    monkeypatch.delitem(sys.modules, "librosa", raising=False)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "librosa":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert ensure_numba_cache_dir() is None


def test_numba_cache_dir_sets_env_when_pycache_unwritable(monkeypatch, tmp_path) -> None:
    """回归测试：librosa 的 __pycache__ 不可写时必须自动兜底。

    否则 numba 会在 `import librosa` 期间抛
    `RuntimeError: cannot cache function '__o_fold': no locator available`，
    导致 qwen_tts 完全无法导入、模型加载不起来。
    """
    import sys
    import types

    from noveltts import ensure_numba_cache_dir

    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)

    # 造一个假的已加载 librosa，其 __pycache__ 指向不可写位置
    fake_librosa = types.ModuleType("librosa")
    fake_librosa.__file__ = str(tmp_path / "core" / "__init__.py")  # 目录不存在 → 不可写
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    cache_dir = ensure_numba_cache_dir()
    assert cache_dir is not None, "应当设置 NUMBA_CACHE_DIR 兜底"
    import os

    assert os.environ["NUMBA_CACHE_DIR"] == cache_dir
    assert os.path.isdir(cache_dir), "兜底目录必须真实存在，否则 numba 仍然会失败"


def test_numba_cache_dir_skips_when_pycache_writable(monkeypatch, tmp_path) -> None:
    """__pycache__ 可写时不要多此一举（否则白白失去缓存复用）。"""
    import sys
    import types

    from noveltts import ensure_numba_cache_dir

    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    core = tmp_path / "core"
    (core / "__pycache__").mkdir(parents=True)

    fake_librosa = types.ModuleType("librosa")
    fake_librosa.__file__ = str(core / "__init__.py")
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    assert ensure_numba_cache_dir() is None
