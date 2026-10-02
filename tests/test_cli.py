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


# --------------------------------------------------------------------------- #
# 本机覆盖配置（config.local.yaml）
# --------------------------------------------------------------------------- #
# 起因：用 tar 覆盖式部署代码时会把 config.yaml 一起冲掉，导致云端被改好的
# 模型绝对路径丢失（回退成 HuggingFace 仓库 id，而云端连不上 HF）。
# 因此把「机器相关设置」移到不入库的 config.local.yaml。


def test_local_config_overrides_main(workdir: Path) -> None:
    """config.local.yaml 应覆盖 config.yaml 里显式出现的字段。"""
    import os

    (workdir / "config.local.yaml").write_text(
        "generation:\n  model_path: /root/models/Qwen3-TTS-12Hz-1.7B-Base\n  batch_size: 8\n",
        encoding="utf-8",
    )
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        from noveltts.cli import _load_config, build_parser

        args = build_parser().parse_args(["info"])
        config = _load_config(args)
    finally:
        os.chdir(previous)

    assert config.generation.model_path == "/root/models/Qwen3-TTS-12Hz-1.7B-Base"
    assert config.generation.batch_size == 8
    # 本机文件没写的字段必须保持主配置的值
    assert config.generation.language == "dummy-model" or config.generation.language == "English"


def test_local_config_only_overrides_provided_fields(workdir: Path) -> None:
    """本机文件只写一行时，不能把其余字段重置成默认值。"""
    import os

    (workdir / "config.local.yaml").write_text(
        "generation:\n  batch_size: 12\n", encoding="utf-8"
    )
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        from noveltts.cli import _load_config, build_parser

        args = build_parser().parse_args(["info"])
        config = _load_config(args)
    finally:
        os.chdir(previous)

    assert config.generation.batch_size == 12
    # 主配置里的值不能被冲掉
    assert config.generation.model_path == "dummy-model"
    assert config.audio.speech_rate == 1.0


def test_local_config_absent_is_fine(workdir: Path) -> None:
    import os

    previous = Path.cwd()
    os.chdir(workdir)
    try:
        from noveltts.cli import _load_config, build_parser

        args = build_parser().parse_args(["info"])
        config = _load_config(args)
    finally:
        os.chdir(previous)
    assert config.generation.model_path == "dummy-model"


def test_explicit_config_ignores_local_config(workdir: Path) -> None:
    """显式 -c 指定配置时不应再叠加本机文件，避免串味。"""
    import os

    (workdir / "config.local.yaml").write_text(
        "generation:\n  batch_size: 99\n", encoding="utf-8"
    )
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        # 用 run_cli 走的是显式 -c config.yaml
        assert run_cli(workdir, "info") == 0
        from noveltts.cli import _load_config, build_parser

        args = build_parser().parse_args(["-c", "config.yaml", "info"])
        config = _load_config(args)
    finally:
        os.chdir(previous)
    assert config.generation.batch_size != 99


def test_local_config_unknown_key_warns_but_works(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    (workdir / "config.local.yaml").write_text(
        "generation:\n  batch_size: 5\n  not_a_real_option: 1\n", encoding="utf-8"
    )
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        from noveltts.cli import _load_config, build_parser

        args = build_parser().parse_args(["info"])
        config = _load_config(args)
    finally:
        os.chdir(previous)
    assert config.generation.batch_size == 5


def test_local_config_project_dirs_are_re_resolved(workdir: Path) -> None:
    """回归测试：本机覆盖 project 段后，派生的 *_path 必须重新解析。

    起因（实际 bug）：config.local.yaml 里改了 output_dir 指向数据盘，
    但 project.output_path 是 AppConfig.load 时算好并缓存的，只换
    output_dir 字段不会更新它——于是 info 仍显示老路径、产物依然写回系统盘。
    """
    import os

    target = workdir / "data-disk" / "audio"
    (workdir / "config.local.yaml").write_text(
        f"project:\n  output_dir: {target}\n", encoding="utf-8"
    )
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        from noveltts.cli import _load_config, build_parser

        args = build_parser().parse_args(["info"])
        config = _load_config(args)
    finally:
        os.chdir(previous)

    assert config.project.output_dir == str(target)
    # 派生属性必须跟着变，而不是停留在 project.root/output/audio
    assert config.project.output_path == target.resolve()
    assert config.project.output_path != (workdir / "output" / "audio").resolve()
    # 目录应已被创建
    assert config.project.output_path.is_dir()


def test_local_config_work_dir_override(workdir: Path) -> None:
    """work_dir 同样要重新解析（云端把中间产物也放数据盘）。"""
    import os

    target = workdir / "data-disk" / "work"
    (workdir / "config.local.yaml").write_text(
        f"project:\n  work_dir: {target}\n", encoding="utf-8"
    )
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        from noveltts.cli import _load_config, build_parser

        args = build_parser().parse_args(["info"])
        config = _load_config(args)
    finally:
        os.chdir(previous)

    assert config.project.work_path == target.resolve()


def test_command_line_beats_local_config(workdir: Path) -> None:
    """回归测试：命令行覆盖必须**高于** config.local.yaml。

    起因（实际 bug）：config.local.yaml 里写了 ``audio.speech_rate: 1.0``，
    于是 ``--rate 1.6`` 完全失效——耗时与产物时长都跟原速一模一样，
    manifest 里也记成 1.0。而文档承诺优先级是
    「内置默认 < config.yaml < config.local.yaml < 命令行」。

    根因：本机配置的叠加发生在命令行覆盖**之后**，于是后面的覆盖被前面盖掉。
    """
    import os

    (workdir / "config.local.yaml").write_text(
        "audio:\n  speech_rate: 1.0\n  min_gap_seconds: 0.4\n"
        "generation:\n  batch_size: 3\n",
        encoding="utf-8",
    )
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        from noveltts.cli import _load_config, build_parser

        # 不带 --rate：本机配置生效
        args = build_parser().parse_args(["synth", "--chapter", "c003"])
        plain = _load_config(args)
        assert plain.audio.speech_rate == 1.0
        assert plain.audio.min_gap_seconds == 0.4
        assert plain.generation.batch_size == 3

        # 带 --rate：命令行必须赢
        args = build_parser().parse_args(["synth", "--chapter", "c003", "--rate", "1.6"])
        overridden = _load_config(args)
        assert overridden.audio.speech_rate == 1.6, (
            "命令行 --rate 被 config.local.yaml 盖掉了"
        )
        # 命令行没提到的项仍来自本机配置
        assert overridden.audio.min_gap_seconds == 0.4
        assert overridden.generation.batch_size == 3

        # --batch-size / --device 同理
        args = build_parser().parse_args(
            ["synth", "--chapter", "c003", "--batch-size", "7", "--device", "cpu"]
        )
        cfg = _load_config(args)
        assert cfg.generation.batch_size == 7
        assert cfg.generation.device == "cpu"
        # 没被命令行覆盖的 speech_rate 仍来自本机配置
        assert cfg.audio.speech_rate == 1.0
    finally:
        os.chdir(previous)


def test_command_line_rate_validation(workdir: Path) -> None:
    """--rate 必须为正数。"""
    import os

    previous = Path.cwd()
    os.chdir(workdir)
    try:
        from noveltts.cli import _load_config, build_parser

        args = build_parser().parse_args(["synth", "--rate", "0"])
        with pytest.raises(ValueError):
            _load_config(args)
        args = build_parser().parse_args(["synth", "--rate", "-1"])
        with pytest.raises(ValueError):
            _load_config(args)
    finally:
        os.chdir(previous)
