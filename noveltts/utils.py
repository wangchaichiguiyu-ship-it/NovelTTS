"""通用工具：日志、文件读写、原子写、文本统计。

本模块只使用标准库，保证 ``import noveltts.utils`` 在任何环境下都是毫秒级。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

#: 记录一次 :func:`setup_logging` 是否已经完成，避免重复挂 handler。
_LOGGING_READY = False


def setup_logging(level: str | int = "INFO", *, force: bool = False) -> None:
    """配置根 logger。

    Args:
        level: 日志级别，接受 ``"INFO"`` / ``"DEBUG"`` 等字符串或 :mod:`logging` 常量。
        force: 为 ``True`` 时重新配置（重复调用 CLI 时会用到）。

    Notes:
        重复调用不会叠加 handler，因此不会出现日志打印两遍的问题。
    """
    global _LOGGING_READY
    if _LOGGING_READY and not force:
        logging.getLogger().setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _LOGGING_READY = True


def get_logger(name: str) -> logging.Logger:
    """返回带统一前缀的 logger。"""
    return logging.getLogger(f"noveltts.{name}")


# --------------------------------------------------------------------------- #
# 文件读写
# --------------------------------------------------------------------------- #


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """确保目录存在并返回 :class:`Path`（``exist_ok=True``）。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: str | os.PathLike[str], encoding: str = "utf-8") -> str:
    """读取文本文件，自动吞掉 UTF-8 BOM。"""
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.decode(encoding, errors="replace")


def write_text(
    path: str | os.PathLike[str],
    text: str,
    encoding: str = "utf-8",
) -> Path:
    """原子地写入文本文件。

    先写同目录临时文件再 ``os.replace``，避免中途 Ctrl-C 留下半截文件
    （对上千段音频的长时间跑批很重要）。
    """
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-", suffix=target.suffix)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """读取 JSON；文件不存在时返回 ``default`` 而不是抛异常。"""
    p = Path(path)
    if not p.is_file():
        return default
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """原子地写入 JSON（默认不转义非 ASCII，方便直接看中文）。"""
    return write_text(
        path,
        json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii) + "\n",
    )


def sha1_of(text: str) -> str:
    """返回文本的 SHA-1 前 12 位，用作片段文件名/缓存键。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def slugify(value: str, *, max_length: int = 60, fallback: str = "chapter") -> str:
    """把章节标题转换成安全的文件名片段。

    例：``"Loss 3: Crimson Rhapsody"`` -> ``"loss-3-crimson-rhapsody"``。
    """
    value = unicodedata.normalize("NFKD", value)
    value = value.replace("'", "").replace("\u2019", "")
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value).strip("-").lower()
    value = re.sub(r"-{2,}", "-", value)[:max_length].strip("-")
    return value or fallback


# --------------------------------------------------------------------------- #
# 文本统计
# --------------------------------------------------------------------------- #

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

#: 会被计入“字数”的字符（字母、数字、CJK）。
_COUNTABLE_RE = re.compile(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]")


def count_words(text: str) -> int:
    """估算朗读字数：拉丁文按词计，CJK 按字计。

    这个数字用于预估音频时长与切分长度，不追求语言学精确。
    """
    cjk_chars = len(_CJK_RE.findall(text))
    latin_words = len(_LATIN_WORD_RE.findall(text))
    return cjk_chars + latin_words


def count_chars(text: str) -> int:
    """返回“可朗读字符数”，忽略空白与标点，用于切分阈值判断。"""
    return len(_COUNTABLE_RE.findall(text))


def estimate_speech_seconds(text: str, chars_per_second: float = 14.0) -> float:
    """粗略估计一段文本的朗读时长（秒）。

    Args:
        text: 待朗读文本。
        chars_per_second: 每秒朗读的可朗读字符数，英文默认约 14。

    Returns:
        预估秒数。仅用于进度提示与容量规划。
    """
    return count_chars(text) / max(chars_per_second, 0.1)


def format_duration(seconds: float) -> str:
    """把秒数格式化为 ``HH:MM:SS`` 或 ``MM:SS``。"""
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def human_size(num_bytes: float) -> str:
    """把字节数格式化为人类可读字符串。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


# --------------------------------------------------------------------------- #
# 杂项
# --------------------------------------------------------------------------- #


def chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """把可迭代对象切成固定大小的列表（``size`` 必须为正）。"""
    if size < 1:
        raise ValueError(f"size 必须 >= 1，收到 {size}")
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，``override`` 优先；返回新字典，不修改入参。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(value: str | os.PathLike[str], *, base: str | os.PathLike[str] | None = None) -> Path:
    """把可能是相对路径的配置值解析成绝对路径。

    Args:
        value: 原始路径。
        base: 相对路径的基准目录；``None`` 时用当前工作目录。

    Returns:
        展开 ``~`` 之后的绝对路径（不做存在性检查）。
    """
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()
