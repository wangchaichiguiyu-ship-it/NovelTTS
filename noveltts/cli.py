"""命令行界面：``noveltts`` 的全部子命令。

子命令一览
----------

============================  ====================================================
命令                          作用
============================  ====================================================
``info``                      查看素材、章节与配置概览（**不加载模型**）
``parse``                     解析 EPUB/TXT -> 章节 JSON
``normalize``                 洗稿预览（看模型会被喂进去什么文本）
``segment``                   切分 -> 片段 JSON + 可读的片段清单
``index``                     生成「片段 id -> 文本」索引（失败重跑要用）
``synth``                     **调用模型**生成音频（章节 / 单段 / 前 N 段）
``concat``                    拼接片段 -> 章节 wav
``encode``                    压缩章节 wav -> m4a/mp3
``run``                       一键跑完整条流水线
============================  ====================================================

设计约定
--------

* **只有 ``synth`` 与 ``run`` 会加载模型**。其余命令都是纯 CPU 文本处理，
  可以放心反复运行、快速调试。
* ``synth`` / ``run`` 支持 ``--dry-run``：用假引擎产出正弦波，
  用来在没有 GPU、或不想烧算力的时候验证整条链路是否通畅。
* 所有路径默认相对 ``config.yaml`` 所在目录解析。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import AppConfig
from .utils import (
    format_duration,
    get_logger,
    human_size,
    read_json,
    setup_logging,
    write_json,
    write_text,
)

logger = get_logger("cli")

#: 默认配置文件（放在项目根目录，**入库**）。
DEFAULT_CONFIG_NAME = "config.yaml"

#: 本机覆盖配置（**不入库**）。
#:
#: 为什么需要这个：``config.yaml`` 是入库的、团队共享的，里面的
#: ``model_path`` 等值必须是「通用默认值」；而每台机器的路径都不一样
#: （本地 Mac 与云端实例的模型目录不同）。
#: 如果直接改 ``config.yaml``，那么每次用 git pull / 重新部署覆盖代码时，
#: 这些本机设置就会被冲掉——**这是实际踩过的坑**。
#:
#: 因此约定：机器相关的设置写进 ``config.local.yaml``，它以**深层覆盖**
#: 的方式盖在 ``config.yaml`` 之上，且被 .gitignore 排除。
LOCAL_CONFIG_NAME = "config.local.yaml"


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """构造 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        prog="noveltts",
        description="小说 -> 有声书流水线（Qwen3-TTS 声音克隆）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "典型用法：\n"
            "  noveltts info                       # 先看看有什么素材\n"
            "  noveltts parse --list               # 解析并打印章节\n"
            "  noveltts segment --chapter c002 --preview   # 看切分结果\n"
            "  noveltts synth --chapter c002 --limit 5     # 只生成 5 段试听\n"
            "  noveltts synth --text 'Hello there.'        # 单段生成\n"
            "  noveltts concat --chapter c002              # 拼接成章\n"
            "  noveltts encode --chapter c002              # 压缩成 m4a\n"
            "  noveltts run --chapter c002                 # 一键全流程\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"noveltts {__version__}")
    parser.add_argument(
        "-c", "--config", default=None, help=f"配置文件路径（默认 {DEFAULT_CONFIG_NAME}）"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="输出 DEBUG 级日志"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="只输出 WARNING 及以上日志"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<命令>")

    # ---------------------------------------------------------------- info
    info = subparsers.add_parser("info", help="查看素材、章节与配置概览（不加载模型）")
    _add_common_source_args(info)
    info.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本处理")

    # --------------------------------------------------------------- parse
    parse = subparsers.add_parser("parse", help="解析 EPUB/TXT，切出章节")
    _add_common_source_args(parse)
    parse.add_argument("--out", default=None, help="章节 JSON 输出路径")
    parse.add_argument("--list", action="store_true", help="在日志里列出所有章节")
    parse.add_argument("--dump-text", action="store_true", help="同时导出每章纯文本")

    # ----------------------------------------------------------- normalize
    normalize = subparsers.add_parser("normalize", help="洗稿预览（不写文件）")
    _add_common_source_args(normalize)
    normalize.add_argument("--chapter", default=None, help="章节 id / 序号 / 标题片段")
    normalize.add_argument("--text", default=None, help="直接对一段文本洗稿")
    normalize.add_argument("--limit", type=int, default=10, help="最多展示多少段（默认 10）")
    normalize.add_argument("--show-unchanged", action="store_true", help="也展示未改动的段落")

    # ------------------------------------------------------------- segment
    segment = subparsers.add_parser("segment", help="语意切分 -> 片段清单")
    _add_common_source_args(segment)
    segment.add_argument("--chapter", default=None, help="只切这一章；省略则切全书")
    segment.add_argument("--out", default=None, help="片段 JSON 输出路径")
    segment.add_argument("--preview", type=int, default=0, help="打印前 N 个片段")
    segment.add_argument("--no-normalize", action="store_true", help="跳过洗稿（保留原文）")

    # --------------------------------------------------------------- index
    index = subparsers.add_parser("index", help="生成片段文本索引（失败重跑依赖它）")
    _add_common_source_args(index)
    index.add_argument("--chapter", default=None, help="只处理这一章；省略则全部")

    # --------------------------------------------------------------- synth
    synth = subparsers.add_parser("synth", help="调用模型生成音频（会加载模型！）")
    _add_common_source_args(synth)
    synth.add_argument("--chapter", default=None, help="要合成的章节")
    synth.add_argument("--text", default=None, help="直接合成一段文本（一键生成一小段）")
    synth.add_argument(
        "--segments", default=None,
        help="只合成指定片段，如 '3,5,8-12'（配合 --chapter 使用）",
    )
    synth.add_argument("--limit", type=int, default=0, help="只合成前 N 段（0=全部）")
    synth.add_argument("--force", action="store_true", help="已存在的片段也重新生成")
    synth.add_argument("--no-title", action="store_true", help="不朗读章节标题")
    synth.add_argument("--language", default=None, help="覆盖语言（如 English/Chinese）")
    synth.add_argument(
        "--batch-size", default=None,
        help="批大小：auto 按显存自推，或写数字（Mac 16G 建议 1，云端 24G 可 4~8）",
    )
    synth.add_argument("--device", default=None, help="覆盖设备：auto/cuda/mps/cpu")
    synth.add_argument(
        "--rate", type=float, default=None,
        help="语速倍率（变速不变调）：1.0 原速，1.2 快 20%%，0.85 慢 15%%（建议 0.5~2.0）",
    )
    synth.add_argument(
        "--retry-failed", action="store_true", help="只重跑 manifest 里失败的片段"
    )
    synth.add_argument(
        "--dry-run", action="store_true",
        help="用假引擎跑通流程（不加载模型，产出正弦波，用于验证链路）",
    )

    # -------------------------------------------------------------- concat
    concat = subparsers.add_parser("concat", help="把片段拼接成章节 wav")
    _add_common_source_args(concat)
    concat.add_argument("--chapter", default=None, help="章节；省略则拼接所有已生成的章")
    concat.add_argument("--out", default=None, help="输出 wav 路径")
    concat.add_argument("--gap", type=float, default=None, help="段间停顿秒数覆盖")
    concat.add_argument("--merge-book", action="store_true", help="再把所有章合并成整本")
    concat.add_argument("--book-name", default="full_book", help="整本文件名（不含扩展名）")

    # -------------------------------------------------------------- encode
    encode = subparsers.add_parser("encode", help="压缩章节 wav -> m4a/mp3")
    _add_common_source_args(encode)
    encode.add_argument("--chapter", default=None, help="章节；省略则压缩所有 wav")
    encode.add_argument("--format", default=None, help="覆盖格式：m4a/mp3/opus/flac")
    encode.add_argument("--bitrate", default=None, help="覆盖码率，如 64k")
    encode.add_argument("--keep-wav", action="store_true", help="保留源 wav")

    # ----------------------------------------------------------------- run
    run = subparsers.add_parser("run", help="一键跑完整条流水线")
    _add_common_source_args(run)
    run.add_argument("--chapter", default=None, help="只处理这一章；省略则处理全书")
    run.add_argument("--limit", type=int, default=0, help="每章只合成前 N 段（试听）")
    run.add_argument("--force", action="store_true", help="强制重新生成音频")
    run.add_argument("--no-encode", action="store_true", help="跳过压缩")
    run.add_argument("--no-concat", action="store_true", help="跳过拼接")
    run.add_argument("--language", default=None, help="覆盖语言")
    run.add_argument("--device", default=None, help="覆盖设备：auto/cuda/mps/cpu")
    run.add_argument(
        "--batch-size", default=None,
        help="批大小：auto 按显存自推，或写数字（Mac 16G 建议 1，云端 24G 可 4~8）",
    )
    run.add_argument(
        "--rate", type=float, default=None,
        help="语速倍率（变速不变调）：1.0 原速，1.2 快 20%%，0.85 慢 15%%（建议 0.5~2.0）",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="用假引擎跑通全流程（不加载模型）"
    )

    return parser


def _add_common_source_args(parser: argparse.ArgumentParser) -> None:
    """给子命令加上「素材来源」相关参数。"""
    parser.add_argument("--source", default=None, help="素材文件（.epub/.txt）；省略则自动选择")
    parser.add_argument("--chapters-json", default=None, help="直接复用已解析的章节 JSON")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。

    Args:
        argv: 参数列表；``None`` 时取 ``sys.argv[1:]``。

    Returns:
        进程退出码（0 成功，非 0 失败）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO")
    setup_logging(level, force=True)

    try:
        config = _load_config(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    handlers = {
        "info": cmd_info,
        "parse": cmd_parse,
        "normalize": cmd_normalize,
        "segment": cmd_segment,
        "index": cmd_index,
        "synth": cmd_synth,
        "concat": cmd_concat,
        "encode": cmd_encode,
        "run": cmd_run,
    }
    handler = handlers[args.command]
    try:
        return handler(config, args)
    except KeyboardInterrupt:
        logger.warning("已被用户中断（已生成的片段已落盘，重跑会自动跳过）")
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        logger.error("%s", exc)
        if args.verbose:
            raise
        return 1


def _load_config(args: argparse.Namespace) -> AppConfig:
    """加载配置，并应用「本机覆盖」与命令行覆盖。

    优先级（低 -> 高）::

        内置默认值  <  config.yaml  <  config.local.yaml  <  命令行参数

    ``config.local.yaml`` 用来放**机器相关**的设置（模型绝对路径、设备、
    批大小等），它不入库，因此代码更新不会冲掉本机配置。

    Raises:
        FileNotFoundError: 显式指定的配置文件不存在。
        ValueError: 路径或参数非法。
    """
    config_path = args.config
    # 用默认路径查找时的基准目录（也是 config.local.yaml 的查找目录）
    default_root = Path.cwd()
    root: Path | None = None
    if config_path is None:
        candidate = default_root / DEFAULT_CONFIG_NAME
        if candidate.is_file():
            config_path = str(candidate)
            root = default_root
        else:
            root = default_root
    else:
        explicit = Path(config_path).expanduser().resolve()
        root = explicit.parent

    # --- 本机覆盖 ---
    # 只有「用默认路径找到 config.yaml」时才自动叠加，避免显式 -c 时意外串味。
    # 注意 root 在上面的两个分支里都要赋值，否则 config.yaml 存在时 root 会是
    # None，本机覆盖就永远找不到（这是刚写这个功能时踩的坑）。
    local_config: Path | None = None
    if args.config is None and root is not None:
        candidate = root / LOCAL_CONFIG_NAME
        if candidate.is_file():
            local_config = candidate

    overrides: dict[str, Any] = {}
    if getattr(args, "device", None):
        overrides.setdefault("generation", {})["device"] = args.device
    if getattr(args, "batch_size", None):
        # 允许写 "auto" 让程序按显存自推；纯数字字符串转成 int
        raw_batch = str(args.batch_size).strip().lower()
        overrides.setdefault("generation", {})["batch_size"] = (
            "auto" if raw_batch == "auto" else int(raw_batch)
        )
    if getattr(args, "language", None):
        overrides.setdefault("generation", {})["language"] = args.language
    if getattr(args, "rate", None) is not None:
        if args.rate <= 0:
            raise ValueError(f"--rate 必须 > 0，收到 {args.rate}")
        overrides.setdefault("audio", {})["speech_rate"] = float(args.rate)
    if getattr(args, "format", None):
        overrides.setdefault("encode", {})["format"] = args.format
    if getattr(args, "bitrate", None):
        overrides.setdefault("encode", {})["bitrate"] = args.bitrate
    if getattr(args, "keep_wav", False):
        overrides.setdefault("encode", {})["keep_wav"] = True

    config = AppConfig.load(config_path, overrides=overrides or None, root=root)
    if local_config is not None:
        # 本机覆盖：以「config.local.yaml 优先」的方式再叠一层
        local = AppConfig.load(
            local_config,
            overrides=overrides or None,
            root=root,
        )
        config = _overlay_config(config, local, local_config)
    config.project.ensure_dirs()
    logger.debug("配置来源：%s", config.config_path or "（内置默认值）")
    if local_config is not None:
        logger.debug("已叠加本机覆盖：%s", local_config)
    return config


def _overlay_config(base: AppConfig, local: AppConfig, local_path: Path) -> AppConfig:
    """把本机覆盖配置叠到主配置之上。

    只覆盖 ``config.local.yaml`` 里**显式出现**的字段，其余保持主配置的值。
    这样本机文件可以只写一行 ``model_path``，而不必复制整份配置。

    Args:
        base: 主配置（``config.yaml``）。
        local: 本机覆盖（``config.local.yaml``）。
        local_path: 本机文件路径（仅用于日志）。

    Returns:
        合并后的 :class:`~noveltts.config.AppConfig`（新对象，不修改入参）。
    """
    from dataclasses import asdict, fields, replace

    from .config import _load_yaml  # 局部导入：避免顶层循环依赖

    raw = _load_yaml(local_path)
    # 只保留本机文件里真实写了的字段，避免用默认值把主配置覆盖掉
    merged_sections: dict[str, Any] = {}
    for section in fields(AppConfig):
        name = section.name
        if name == "config_path" or name not in raw:
            continue
        provided = raw[name]
        if not isinstance(provided, dict):
            merged_sections[name] = provided
            continue
        current = asdict(getattr(base, name))
        filtered = {k: v for k, v in provided.items() if k in current}
        unknown = set(provided) - set(current)
        if unknown:
            logger.warning(
                "%s 的 %s 段有未知配置项，已忽略：%s",
                local_path.name,
                name,
                "、".join(sorted(unknown)),
            )
        merged_sections[name] = replace(getattr(base, name), **filtered)

    if not merged_sections:
        return base

    logger.info(
        "已叠加本机配置 %s（覆盖：%s）",
        local_path.name,
        "、".join(sorted(merged_sections)),
    )
    result = replace(base, **merged_sections)

    # 关键：project 段的 *_dir 被覆盖后必须**重新解析**一次。
    # 否则 project.output_path 等派生属性仍是旧值（它们在 AppConfig.load 里
    # 算好并缓存了）。这是本功能刚写时的实际 bug：改了 output_dir，
    # 但 noveltts info 仍显示老路径，产物照样写回系统盘。
    if "project" in merged_sections:
        result.project.resolve(result.project.root_path)

    result.config_path = base.config_path
    return result


# --------------------------------------------------------------------------- #
# 素材定位与解析辅助
# --------------------------------------------------------------------------- #


def _pick_source(config: AppConfig, args: argparse.Namespace) -> Path:
    """决定用哪个素材文件。"""
    from .epub_parser import find_source_files

    if getattr(args, "source", None):
        path = Path(args.source).expanduser()
        if not path.is_absolute():
            path = (config.project.root_path / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到素材文件：{path}")
        return path

    files = find_source_files(config.project.source_path)
    if not files:
        raise FileNotFoundError(
            f"{config.project.source_path} 里没有 .epub/.txt 文件。"
            "请把书放进该目录，或用 --source 指定。"
        )
    if len(files) > 1:
        # 优先 EPUB：它的目录（nav/ncx）能给出完整章节标题，
        # 而 TXT 里的标题常被 <br/> 拆成两行，只能切出「Loss 1:」这种半截标题。
        epubs = [f for f in files if f.suffix.lower() == ".epub"]
        chosen = epubs[0] if epubs else files[0]
        logger.info(
            "素材目录里有 %d 个文件，自动选择 %s（可用 --source 指定其它）",
            len(files),
            chosen.name,
        )
        return chosen
    return files[0]


def _chapters_json_path(config: AppConfig) -> Path:
    """章节 JSON 的默认路径。"""
    return config.project.work_path / "chapters.json"


def _segments_json_path(config: AppConfig, chapter_id: str | None = None) -> Path:
    """片段 JSON 的默认路径。"""
    if chapter_id:
        return config.project.work_path / "segments" / f"{chapter_id}.json"
    return config.project.work_path / "segments.json"


def _load_book(config: AppConfig, args: argparse.Namespace):
    """加载（或解析）整本书。

    优先复用 ``--chapters-json`` 或默认的 ``chapters.json``；都没有时才
    重新解析素材。这样 ``segment`` / ``synth`` 不必每次重跑解析。
    """
    from .models import Book

    explicit = getattr(args, "chapters_json", None)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (config.project.root_path / path).resolve()
        data = read_json(path)
        if not isinstance(data, dict):
            raise FileNotFoundError(f"章节 JSON 不存在或格式错误：{path}")
        logger.info("复用章节 JSON：%s", path)
        return Book.from_dict(data)

    default_path = _chapters_json_path(config)
    if default_path.is_file():
        data = read_json(default_path)
        if isinstance(data, dict):
            logger.debug("复用章节 JSON：%s", default_path)
            return Book.from_dict(data)

    return _parse_book(config, args)


def _parse_book(config: AppConfig, args: argparse.Namespace):
    """解析素材并写入章节 JSON。"""
    from .epub_parser import parse_book

    source = _pick_source(config, args)
    logger.info("解析素材：%s", source.name)
    book = parse_book(source, config.parse)
    write_json(_chapters_json_path(config), book.to_dict(include_segments=False))
    logger.info(
        "解析完成：%d 章 / %s 字 / 预估 %s",
        len(book.chapters),
        f"{book.total_chars:,}",
        format_duration(book.estimated_seconds),
    )
    return book


def _segment_book(config: AppConfig, book, *, normalize: bool = True) -> None:
    """切分整本书并落盘（章节 JSON 会带上 segments）。"""
    from .segmenter import TextSegmenter

    segmenter = TextSegmenter(config.segment, language=book.language)
    for chapter in book.chapters:
        segmenter.segment_chapter(chapter, normalize=normalize)

    write_json(_chapters_json_path(config), book.to_dict(include_segments=True))
    logger.info("切分完成，已写入 %s", _chapters_json_path(config))


def _ensure_segments(config: AppConfig, book, args: argparse.Namespace) -> None:
    """确保章节已经切分（没切就先切）。"""
    if all(chapter.segments for chapter in book.chapters):
        return
    logger.info("章节尚未切分，先执行切分…")
    _segment_book(config, book, normalize=not getattr(args, "no_normalize", False))


def _write_segment_index(config: AppConfig, chapters) -> list[Path]:
    """为若干章写「片段 id -> 文本」索引。"""
    from .synthesis import dump_segments_index

    paths: list[Path] = []
    for chapter in chapters:
        if not chapter.segments:
            continue
        paths.append(dump_segments_index(chapter.segments, work_path=config.project.work_path))
    return paths


def parse_segment_spec(spec: str) -> list[int]:
    """解析 ``'3,5,8-12'`` 形式的片段序号。"""
    result: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, _, end_text = part.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"片段区间格式错误：{part!r}") from exc
            if start > end:
                start, end = end, start
            result.extend(range(start, end + 1))
        else:
            try:
                result.append(int(part))
            except ValueError as exc:
                raise ValueError(f"片段序号格式错误：{part!r}") from exc
    return sorted(set(result))


# --------------------------------------------------------------------------- #
# 各子命令
# --------------------------------------------------------------------------- #


def cmd_info(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts info`` —— 概览，不加载模型。"""
    from .audio_io import audio_available
    from .concat import discover_chapters
    from .encode import encoder_available
    from .epub_parser import find_source_files

    sources = find_source_files(config.project.source_path)
    generated = discover_chapters(config)
    audio_ok, audio_reason = audio_available()
    encode_ok, encode_reason = encoder_available(config.encode)

    payload: dict[str, Any] = {
        "version": __version__,
        "config_path": str(config.config_path) if config.config_path else None,
        "project_root": str(config.project.root_path),
        "source_dir": str(config.project.source_path),
        "output_dir": str(config.project.output_path),
        "sources": [{"name": p.name, "size": p.stat().st_size} for p in sources],
        "generated_chapters": generated,
        "audio_backend": {"ok": audio_ok, "detail": audio_reason},
        "encoder": {"ok": encode_ok, "detail": encode_reason},
        "model": {
            "path": config.generation.model_path,
            "device": config.generation.device,
            "dtype": config.generation.dtype,
            "batch_size": config.generation.batch_size,
            "language": config.generation.language,
        },
        "segment": {
            "min_chars": config.segment.min_chars,
            "max_chars": config.segment.max_chars,
            "hard_max_chars": config.segment.hard_max_chars,
        },
    }

    # 若已有解析结果，把章节规模也带上
    chapters_json = _chapters_json_path(config)
    if chapters_json.is_file():
        data = read_json(chapters_json, {})
        if isinstance(data, dict):
            payload["book"] = {
                "title": data.get("title", ""),
                "chapter_count": data.get("chapter_count", 0),
                "total_chars": data.get("total_chars", 0),
                "estimated_seconds": data.get("estimated_seconds", 0),
                "path": str(chapters_json),
            }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"NovelTTS {__version__}")
    print(f"  项目根目录 : {payload['project_root']}")
    print(f"  配置文件   : {payload['config_path'] or '（内置默认值）'}")
    print(f"  素材目录   : {payload['source_dir']}")
    print(f"  输出目录   : {payload['output_dir']}")
    print()
    print("素材文件：")
    if sources:
        for item in payload["sources"]:
            print(f"  - {item['name']} ({human_size(item['size'])})")
    else:
        print("  （无）请把 .epub/.txt 放进素材目录")
    print()
    if "book" in payload:
        book = payload["book"]
        print("已解析的书：")
        print(f"  标题     : {book['title']}")
        print(f"  章节数   : {book['chapter_count']}")
        print(f"  总字数   : {book['total_chars']:,}")
        print(f"  预估时长 : {format_duration(book['estimated_seconds'])}")
        print()
    print(f"已生成音频的章节（{len(generated)}）：")
    print("  " + (", ".join(generated) if generated else "（还没有，先跑 noveltts synth）"))
    print()
    print("依赖检查：")
    print(f"  音频读写 : {'OK' if audio_ok else '不可用'} - {audio_reason}")
    print(f"  音频编码 : {'OK' if encode_ok else '不可用'} - {encode_reason}")
    print()
    print("模型设置（synth / run 才会真正加载）：")
    print(f"  路径     : {payload['model']['path']}")
    print(f"  设备     : {payload['model']['device']}  精度: {payload['model']['dtype']}")
    print(f"  批大小   : {payload['model']['batch_size']}  语言: {payload['model']['language']}")
    return 0


def cmd_parse(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts parse`` —— 解析并切章。"""
    book = _parse_book(config, args)

    if args.out:
        out = Path(args.out).expanduser()
        if not out.is_absolute():
            out = (config.project.root_path / out).resolve()
        write_json(out, book.to_dict(include_segments=False))
        logger.info("章节 JSON 已写入 %s", out)

    if args.list:
        print(f"\n《{book.title}》 共 {len(book.chapters)} 章\n")
        print(f"{'ID':<8}{'字数':>9}{'预估':>10}  标题")
        print("-" * 72)
        for chapter in book.chapters:
            print(
                f"{chapter.chapter_id:<8}{chapter.char_count:>9,}"
                f"{format_duration(chapter.estimated_seconds):>10}  {chapter.title}"
            )
        print("-" * 72)
        print(
            f"{'合计':<8}{book.total_chars:>9,}"
            f"{format_duration(book.estimated_seconds):>10}"
        )

    if args.dump_text:
        text_dir = config.project.work_path / "text"
        written = 0
        for chapter in book.chapters:
            name = f"{chapter.chapter_id}.txt"
            write_text(text_dir / name, chapter.text + "\n")
            written += 1
        logger.info("已导出 %d 个章节纯文本到 %s", written, text_dir)

    return 0


def cmd_normalize(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts normalize`` —— 洗稿预览。"""
    from .normalize import NormalizeReport, build_normalizer

    normalizer = build_normalizer(config.text_rules, language=config.text_rules.language)
    report = NormalizeReport()

    if args.text:
        pairs = [("(命令行文本)", args.text)]
    else:
        book = _load_book(config, args)
        chapter = book.get_chapter(args.chapter) if args.chapter else book.chapters[0]
        pairs = [(f"{chapter.chapter_id} 第{i + 1}段", p.text) for i, p in enumerate(chapter.paragraphs)]

    shown = 0
    print()
    for label, text in pairs:
        result = normalizer.normalize(text, report=report)
        if result.text == text and not args.show_unchanged:
            continue
        if shown >= args.limit:
            break
        shown += 1
        print(f"[{label}]")
        if result.text == text:
            print("  （未改动）")
            print(f"  {text[:160]}")
        else:
            print(f"  原: {text[:160]}")
            print(f"  新: {result.text[:160]}")
        print()

    print(report.summary())
    return 0


def cmd_segment(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts segment`` —— 语意切分。"""
    from .segmenter import summarize_segments

    book = _load_book(config, args)
    chapters = [book.get_chapter(args.chapter)] if args.chapter else book.chapters
    normalize = not args.no_normalize

    if args.chapter:
        _segment_book(config, book, normalize=normalize)
    else:
        _segment_book(config, book, normalize=normalize)

    if args.out:
        out = Path(args.out).expanduser()
        if not out.is_absolute():
            out = (config.project.root_path / out).resolve()
        payload = {
            "title": book.title,
            "language": book.language,
            "chapters": [
                {
                    "chapter_id": chapter.chapter_id,
                    "title": chapter.title,
                    "segments": [s.to_dict() for s in (chapter.segments or [])],
                }
                for chapter in chapters
            ],
        }
        write_json(out, payload)
        logger.info("片段 JSON 已写入 %s", out)

    # 每章写一份 id->文本 索引，供失败重跑与拼接判断停顿使用
    _write_segment_index(config, chapters)

    print()
    for chapter in chapters:
        segments = chapter.segments or []
        print(f"=== {chapter.chapter_id} {chapter.title}")
        print(summarize_segments(segments))
        if args.preview:
            for segment in segments[: args.preview]:
                print(f"  [{segment.index:4d}] {segment.kind:<9} {segment.char_count:>4}字 | {segment.text}")
        print()

    total = sum(len(c.segments or []) for c in chapters)
    logger.info("共 %d 个片段", total)
    return 0


def cmd_index(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts index`` —— 生成片段文本索引。"""
    book = _load_book(config, args)
    _ensure_segments(config, book, args)
    chapters = [book.get_chapter(args.chapter)] if args.chapter else book.chapters
    paths = _write_segment_index(config, chapters)
    for path in paths:
        logger.info("索引已写入 %s", path)
    return 0


def cmd_synth(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts synth`` —— 调用模型生成音频。"""
    from .synthesis import SynthesisSession

    # ---- 模式一：直接合成一段文本 ----
    if args.text:
        logger.info("合成单段文本（%d 字）", len(args.text))
        if args.dry_run:
            logger.warning("dry-run：将使用假引擎，产出正弦波而非真实语音")
        with SynthesisSession(
            config, dry_run=args.dry_run, device=args.device
        ) as session:
            records = session.synthesize_text(
                args.text, chapter_id="adhoc", language=args.language
            )
        for record in records:
            status = "失败" if record.error else ("跳过" if record.skipped else "完成")
            print(f"  {record.segment_id}: {status} {record.audio_path} "
                  f"({record.duration:.2f}s)" + (f" 错误: {record.error}" if record.error else ""))
        print(f"\n输出目录：{config.project.output_path / 'segments' / 'adhoc'}")
        return 0 if all(r.error is None for r in records) else 1

    # ---- 模式二：按章节合成 ----
    book = _load_book(config, args)
    _ensure_segments(config, book, args)

    if args.chapter:
        chapters = [book.get_chapter(args.chapter)]
    elif args.retry_failed:
        chapters = book.chapters
    else:
        chapters = book.chapters
        logger.info("未指定 --chapter，将处理全部 %d 章", len(chapters))

    _write_segment_index(config, chapters)

    if args.dry_run:
        logger.warning("dry-run：将使用假引擎，产出正弦波而非真实语音")

    only = parse_segment_spec(args.segments) if args.segments else None
    failures = 0

    with SynthesisSession(config, dry_run=args.dry_run, device=args.device) as session:
        logger.info("引擎：%s", getattr(session.engine, "describe", lambda: "?")())
        for chapter in chapters:
            if args.retry_failed:
                report = session.retry_failed(chapter.chapter_id, language=args.language)
            else:
                report = session.synthesize_chapter(
                    chapter,
                    force=args.force,
                    language=args.language,
                    include_title=not args.no_title,
                    limit=args.limit,
                    only_segments=only,
                )
            print(f"  {report.summary()}")
            failures += report.failed

    if failures:
        logger.warning(
            "共 %d 段失败。可运行 `noveltts synth --chapter <id> --retry-failed` 单独重跑。",
            failures,
        )
        return 1
    logger.info("全部完成。下一步：noveltts concat --chapter <id>")
    return 0


def cmd_concat(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts concat`` —— 拼接片段。"""
    from .concat import concat_all_chapters, concat_from_manifest, discover_chapters

    if args.chapter:
        result = concat_from_manifest(
            config, args.chapter, output_path=args.out, gap_seconds=args.gap
        )
        print(f"  {result.summary()}")
        print(f"  输出：{result.output_path}")
        if result.missing:
            print(f"  注意：缺失 {len(result.missing)} 段 -> {', '.join(result.missing[:8])}")
        return 0

    results = concat_all_chapters(config, merge_into_single=args.merge_book, output_name=args.book_name)
    if not results:
        logger.error(
            "没有可拼接的章节。请先运行 `noveltts synth`；"
            f"（查找目录：{config.project.output_path / 'segments'}）"
        )
        return 1
    for result in results:
        print(f"  {result.summary()}")
    return 0


def cmd_encode(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts encode`` —— 压缩章节音频。"""
    from .encode import encode_audio, encode_chapters, encoder_available

    ok, detail = encoder_available(config.encode)
    if not ok:
        logger.error("%s", detail)
        return 1

    if args.chapter:
        source = config.project.output_path / f"{args.chapter}.wav"
        if not source.is_file():
            logger.error("找不到 %s，请先运行 `noveltts concat --chapter %s`", source, args.chapter)
            return 1
        result = encode_audio(source, config=config)
        print(f"  {result.summary()}")
        print(f"  输出：{result.output}")
        return 0

    results = encode_chapters(config, keep_wav=args.keep_wav)
    if not results:
        logger.error("没有可压缩的 wav，请先运行 `noveltts concat`")
        return 1
    for result in results:
        print(f"  {result.summary()}")
    return 0


def cmd_run(config: AppConfig, args: argparse.Namespace) -> int:
    """``noveltts run`` —— 一键全流程：切分 -> 合成 -> 拼接 -> 压缩。"""
    from .concat import concat_from_manifest
    from .encode import encode_audio, encoder_available

    book = _load_book(config, args)
    _ensure_segments(config, book, args)
    chapters = [book.get_chapter(args.chapter)] if args.chapter else book.chapters
    _write_segment_index(config, chapters)

    print(f"\n{'=' * 64}")
    print(f"一键生成：{len(chapters)} 章 / "
          f"{sum(len(c.segments or []) for c in chapters)} 片段")
    print(f"{'=' * 64}\n")

    # ---------- 1) 合成 ----------
    from .synthesis import SynthesisSession

    if args.dry_run:
        logger.warning("dry-run：将使用假引擎，产出正弦波而非真实语音")

    failures = 0
    reports = []
    with SynthesisSession(config, dry_run=args.dry_run, device=args.device) as session:
        logger.info("引擎：%s", getattr(session.engine, "describe", lambda: "?")())
        for chapter in chapters:
            report = session.synthesize_chapter(
                chapter,
                force=args.force,
                language=args.language,
                limit=args.limit,
            )
            print(f"  [合成] {report.summary()}")
            reports.append(report)
            failures += report.failed

    # ---------- 2) 拼接 ----------
    wav_paths: list[Path] = []
    if not args.no_concat:
        print()
        for chapter in chapters:
            try:
                result = concat_from_manifest(config, chapter.chapter_id)
            except (FileNotFoundError, RuntimeError) as exc:
                logger.warning("跳过 %s 的拼接：%s", chapter.chapter_id, exc)
                continue
            print(f"  [拼接] {result.summary()}")
            wav_paths.append(result.output_path)

    # ---------- 3) 压缩 ----------
    if not args.no_encode and wav_paths:
        ok, detail = encoder_available(config.encode)
        print()
        if not ok:
            logger.warning("跳过压缩：%s", detail)
        else:
            for wav in wav_paths:
                try:
                    result = encode_audio(wav, config=config)
                except RuntimeError as exc:
                    logger.error("压缩 %s 失败：%s", wav.name, exc)
                    continue
                print(f"  [压缩] {result.summary()}")

    # ---------- 汇总 ----------
    total_audio = sum(r.audio_seconds for r in reports)
    total_elapsed = sum(r.elapsed_seconds for r in reports)
    print(f"\n{'=' * 64}")
    print(
        f"完成：音频 {format_duration(total_audio)} / 合成耗时 {format_duration(total_elapsed)}"
        + (f" / RTF {total_elapsed / total_audio:.2f}" if total_audio else "")
    )
    print(f"输出目录：{config.project.output_path}")
    print(f"{'=' * 64}\n")

    if failures:
        logger.warning("有 %d 段失败，可用 --retry-failed 重跑", failures)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
