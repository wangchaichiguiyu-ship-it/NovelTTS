"""CLI 测试。

只运行**不加载模型**的子命令（``info`` / ``parse`` / ``normalize`` / ``segment``
/ ``index`` / ``concat`` / ``encode``），以及 ``synth`` 的 ``--dry-run`` 路径
（用假引擎产正弦波）。

特别验证：``--help`` 与 ``info`` 绝不能触发 torch / qwen_tts 的导入——
这是「改配置不用等模型加载」的前提。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from noveltts.cli import build_parser, main, parse_segment_spec


@pytest.fixture
def workdir(tmp_path: Path, sample_epub: Path) -> Path:
    """准备一个可用的项目目录（含 sources/config.yaml/参考音频）。"""
    (tmp_path / "sources").mkdir(parents=True, exist_ok=True)
    target = tmp_path / "sources" / sample_epub.name
    target.write_bytes(sample_epub.read_bytes())

    # 参考音频：用 soundfile 造一个 1 秒的小 wav
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(24_000) / 24_000)).astype("float32")
    sf.write(str(tmp_path / "ref.wav"), tone, 24_000)

    (tmp_path / "config.yaml").write_text(
        "project:\n"
        "  source_dir: sources\n"
        "  work_dir: output/work\n"
        "  output_dir: output/audio\n"
        "  cache_dir: output/cache\n"
        "parse:\n"
        "  min_chapter_chars: 1\n"
        "voice:\n"
        "  ref_audio: ref.wav\n"
        "  ref_text: test reference\n"
        "generation:\n"
        "  model_path: dummy-model\n"
        "  device: cpu\n"
        "  num_retries: 0\n"
        "  progress_every: 0\n",
        encoding="utf-8",
    )
    return tmp_path



def _torch_newly_imported(action) -> bool:
    """执行 ``action``，返回它**是否在本进程内首次导入了 torch**。

    Note:
        不能简单地断言 ``"torch" not in sys.modules``：测试跑在同一个
        Python 进程里，只要**任何**一个更早的用例（例如 test_accelerator.py
        里的设备探测）导入过 torch，这个断言就会无辜失败。
        真正要验证的是"**这条命令有没有触发** torch 导入"，
        所以这里用前后快照对比。
    """
    import sys as _sys

    before = "torch" in _sys.modules
    action()
    after = "torch" in _sys.modules
    return after and not before


def run_cli(workdir: Path, *args: str) -> int:
    """在指定目录里跑一次 CLI。"""
    import os

    previous = Path.cwd()
    os.chdir(workdir)
    try:
        return main(["-c", "config.yaml", *args])
    finally:
        os.chdir(previous)


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #


def test_parser_requires_command() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None


@pytest.mark.parametrize(
    "command",
    ["info", "parse", "normalize", "segment", "index", "synth", "concat", "encode", "run"],
)
def test_all_subcommands_parse(command: str) -> None:
    args = build_parser().parse_args([command])
    assert args.command == command


def test_synth_has_dry_run_flag() -> None:
    args = build_parser().parse_args(["synth", "--dry-run"])
    assert args.dry_run is True


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("3", [3]),
        ("3,5", [3, 5]),
        ("8-12", [8, 9, 10, 11, 12]),
        ("3,5,8-10", [3, 5, 8, 9, 10]),
        ("12-8", [8, 9, 10, 11, 12]),      # 反向区间自动归一
        ("3,3,3", [3]),                    # 去重
        ("", []),
    ],
)
def test_parse_segment_spec(spec: str, expected: list[int]) -> None:
    assert parse_segment_spec(spec) == expected


def test_parse_segment_spec_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_segment_spec("a-b")


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "noveltts" in captured.out
    assert "synth" in captured.out


# --------------------------------------------------------------------------- #
# 不加载模型的命令
# --------------------------------------------------------------------------- #


def test_help_does_not_import_torch(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` 必须是零成本操作。"""
    with pytest.raises(SystemExit):
        assert not _torch_newly_imported(lambda: main(["--help"])), (
            "打印帮助不应该导入 torch"
        )


def test_info_runs(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "info") == 0
    out = capsys.readouterr().out
    assert "NovelTTS" in out
    assert "素材文件" in out
    assert "模型设置" in out


def test_info_json_output(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "info", "--json") == 0
    captured = capsys.readouterr()
    assert captured.out.strip(), f"stdout 为空；stderr={captured.err[:400]!r}"
    payload = json.loads(captured.out)
    assert payload["version"]
    assert payload["sources"]
    assert payload["model"]["path"] == "dummy-model"
    assert payload["audio_backend"]["ok"] is True


def test_parse_and_list(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "parse", "--list") == 0
    out = capsys.readouterr().out
    assert "Chapter One" in out
    assert (workdir / "output" / "work" / "chapters.json").is_file()


def test_parse_dump_text(workdir: Path) -> None:
    assert run_cli(workdir, "parse", "--dump-text") == 0
    text_dir = workdir / "output" / "work" / "text"
    assert text_dir.is_dir()
    assert any(text_dir.glob("c001.txt"))


def test_normalize_preview(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "normalize", "--chapter", "c001", "--limit", "3") == 0
    out = capsys.readouterr().out
    assert "洗稿统计" in out


def test_normalize_direct_text(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "normalize", "--text", "Vol. 8 came out in 2025.") == 0
    out = capsys.readouterr().out
    assert "Volume 8" in out


def test_segment_command(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "segment", "--chapter", "c001", "--preview", "3") == 0
    out = capsys.readouterr().out
    assert "片段总数" in out

    chapters = json.loads(
        (workdir / "output" / "work" / "chapters.json").read_text(encoding="utf-8")
    )
    assert chapters["chapters"][0]["segments"], "章节 JSON 里应带上 segments"


def test_index_command(workdir: Path) -> None:
    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "index", "--chapter", "c001") == 0
    index_path = workdir / "output" / "work" / "segments" / "c001.json"
    assert index_path.is_file()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload


# --------------------------------------------------------------------------- #
# dry-run 全链路（假引擎，不加载模型）
# --------------------------------------------------------------------------- #


def test_synth_dry_run(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "synth", "--chapter", "c001", "--dry-run") == 0
    capsys.readouterr()

    segments_dir = workdir / "output" / "audio" / "segments" / "c001"
    assert (segments_dir / "manifest.json").is_file()
    assert list(segments_dir.glob("c001-*.wav"))

    # dry-run 不该加载模型。这里只看「本次有没有**新**导入 torch」——
    # 整个测试跑在同一进程里，别的用例早已把 torch 导进来了。
    assert not _torch_newly_imported(
        lambda: run_cli(workdir, "synth", "--chapter", "c001", "--limit", "1", "--dry-run")
    ), "dry-run 绝不能触发 torch 导入"


def test_synth_dry_run_text_mode(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """一键生成一小段文本。"""
    assert run_cli(workdir, "synth", "--text", "Hello there. How are you?", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "adhoc" in out
    assert list((workdir / "output" / "audio" / "segments" / "adhoc").glob("*.wav"))


def test_synth_dry_run_limit(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "synth", "--chapter", "c001", "--limit", "1", "--dry-run") == 0
    capsys.readouterr()
    segments_dir = workdir / "output" / "audio" / "segments" / "c001"
    assert len(list(segments_dir.glob("c001-*.wav"))) == 1


def test_synth_dry_run_segments_spec(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "segment", "--chapter", "c001") == 0
    capsys.readouterr()
    assert run_cli(
        workdir, "synth", "--chapter", "c001", "--segments", "1,3", "--dry-run"
    ) == 0
    capsys.readouterr()
    segments_dir = workdir / "output" / "audio" / "segments" / "c001"
    names = sorted(p.name for p in segments_dir.glob("c001-*.wav"))
    assert names == ["c001-0001.wav", "c001-0003.wav"]


def test_concat_after_synth(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "synth", "--chapter", "c001", "--dry-run") == 0
    capsys.readouterr()

    code = run_cli(workdir, "concat", "--chapter", "c001")
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (workdir / "output" / "audio" / "c001.wav").is_file()


def test_concat_without_synth_fails(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(workdir, "parse") == 0
    capsys.readouterr()
    assert run_cli(workdir, "concat") == 1


def test_encode_after_concat(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from noveltts.encode import find_ffmpeg

    if find_ffmpeg() is None:
        pytest.skip("未安装 ffmpeg")

    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "synth", "--chapter", "c001", "--dry-run") == 0
    assert run_cli(workdir, "concat", "--chapter", "c001") == 0
    capsys.readouterr()

    assert run_cli(workdir, "encode", "--chapter", "c001") == 0
    assert (workdir / "output" / "audio" / "c001.m4a").is_file()


def test_run_dry_run_full_pipeline(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``run --dry-run`` 一键走完 切分 -> 合成 -> 拼接 -> 压缩。"""
    assert run_cli(workdir, "run", "--chapter", "c001", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "[合成]" in out
    assert "[拼接]" in out
    assert (workdir / "output" / "audio" / "c001.wav").is_file()


def test_run_resumes_without_regenerating(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """第二次 run 应该全部跳过，不再调用引擎。

    Note:
        进度与汇总信息走 **logging（stderr）**，结构化结果走 stdout，
        所以这里两块都要看。
    """
    assert run_cli(workdir, "run", "--chapter", "c001", "--dry-run") == 0
    capsys.readouterr()

    assert run_cli(workdir, "run", "--chapter", "c001", "--dry-run") == 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "跳过" in combined
    assert "新生成 0" in combined


# --------------------------------------------------------------------------- #
# 错误处理
# --------------------------------------------------------------------------- #


def test_missing_config_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import os

    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        code = main(["-c", "nope.yaml", "info"])
    finally:
        os.chdir(previous)
    assert code == 2


def test_missing_source_returns_error(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for path in (workdir / "sources").iterdir():
        path.unlink()
    assert run_cli(workdir, "info") == 0     # info 只是列目录，不算错
    assert run_cli(workdir, "parse") == 1    # parse 找不到素材应当失败


def test_unknown_chapter_returns_error(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(workdir, "parse") == 0
    assert run_cli(workdir, "synth", "--chapter", "c999", "--dry-run") == 1
