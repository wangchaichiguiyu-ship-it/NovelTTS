"""数据结构：章节、段落、片段，以及它们的 JSON（反）序列化。

这些 dataclass 是整个流水线唯一的“通用语言”：

* :class:`Chapter` / :class:`Paragraph` —— 由 :mod:`noveltts.epub_parser` 产出。
* :class:`Segment` —— 由 :mod:`noveltts.segmenter` 产出，是送给 TTS 的最小单位。
* :class:`SegmentRecord` —— 由 :mod:`noveltts.synthesis` 维护，记录每段的落盘结果。

全部支持 ``to_dict`` / ``from_dict``，因此每一步的中间产物都可以落成 JSON
并在任意时刻重新加载（断点续跑、只重跑某一章等等）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from .utils import count_chars, count_words, estimate_speech_seconds

#: 片段类型。用于统计与挑段试听，不影响合成逻辑。
SegmentKind = Literal["narration", "dialogue", "heading", "meta"]


# --------------------------------------------------------------------------- #
# 解析层
# --------------------------------------------------------------------------- #


@dataclass
class Paragraph:
    """一个自然段。

    Attributes:
        text: 段落正文（已折叠空白、已解码 HTML 实体）。
        kind: 段落类型粗分类。
        source: 溯源信息，如 ``"OEBPS/Text/section-0011.html"``。
    """

    text: str
    kind: SegmentKind = "narration"
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paragraph":
        return cls(
            text=data["text"],
            kind=data.get("kind", "narration"),
            source=data.get("source", ""),
        )


@dataclass
class Chapter:
    """一本书里的一章（或序言/后记这样的独立单元）。

    Attributes:
        index: 在书内的顺序，从 1 开始。
        title: 章节标题。
        paragraphs: 自然段列表。
        chapter_id: 稳定标识，默认 ``c{index:03d}``；同时用作输出文件名片段。
        source: 来源文件（EPUB 内部路径或 TXT 的 section 名）。
        language: 该书主要语言（``en`` / ``zh`` / ``ja`` …），用于选洗稿规则。
        segments: 切分后的片段；``None`` 表示尚未切分。
    """

    index: int
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    chapter_id: str = ""
    source: str = ""
    language: str = "en"
    segments: list["Segment"] | None = None

    def __post_init__(self) -> None:
        if not self.chapter_id:
            self.chapter_id = f"c{self.index:03d}"

    # ---------------------------------------------------------------- 便捷属性
    @property
    def text(self) -> str:
        """把标题与所有段落拼回纯文本（标题在前）。"""
        parts = [self.title, *(p.text for p in self.paragraphs)]
        return "\n\n".join(part for part in parts if part)

    @property
    def char_count(self) -> int:
        """可朗读字符总数。"""
        return sum(count_chars(p.text) for p in self.paragraphs)

    @property
    def word_count(self) -> int:
        """估算字数。"""
        return sum(count_words(p.text) for p in self.paragraphs)

    @property
    def estimated_seconds(self) -> float:
        """预估朗读时长（秒）。"""
        return estimate_speech_seconds(self.text)

    def iter_paragraphs(self) -> Iterator[Paragraph]:
        """按顺序迭代段落。"""
        return iter(self.paragraphs)

    # ------------------------------------------------------------ 序列化
    def to_dict(self, *, include_segments: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "chapter_id": self.chapter_id,
            "title": self.title,
            "source": self.source,
            "language": self.language,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "estimated_seconds": round(self.estimated_seconds, 1),
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }
        if include_segments:
            data["segments"] = (
                None if self.segments is None else [s.to_dict() for s in self.segments]
            )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chapter":
        segments = data.get("segments")
        chapter = cls(
            index=int(data["index"]),
            title=data.get("title", ""),
            paragraphs=[Paragraph.from_dict(p) for p in data.get("paragraphs", [])],
            chapter_id=data.get("chapter_id", ""),
            source=data.get("source", ""),
            language=data.get("language", "en"),
        )
        if segments is not None:
            chapter.segments = [Segment.from_dict(s) for s in segments]
        return chapter


# --------------------------------------------------------------------------- #
# 切分层
# --------------------------------------------------------------------------- #


@dataclass
class Segment:
    """送给 TTS 模型的最小文本单位。

    Attributes:
        segment_id: 全局唯一 id，格式 ``{chapter_id}-{序号:04d}``，同时决定 wav 文件名。
        chapter_id: 所属章节。
        index: 章节内序号，从 1 开始（人读的序号）。
        text: 洗稿 + 切分后的最终文本，**直接喂给模型**。
        kind: 片段类型。
        char_count: 可朗读字符数。
        paragraph_index: 该片段主要来自第几个自然段（从 0 开始）。
        paragraph_span: 覆盖的自然段数量，跨段合并时 > 1。
        unit_count: 合并进本片段的句子数。
        source_text: 洗稿前的原文，便于对照排查“念错了”的原因。
    """

    segment_id: str
    chapter_id: str
    index: int
    text: str
    kind: SegmentKind = "narration"
    char_count: int = 0
    paragraph_index: int = 0
    paragraph_span: int = 1
    unit_count: int = 1
    source_text: str = ""

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = count_chars(self.text)
        if not self.source_text:
            self.source_text = self.text

    @property
    def word_count(self) -> int:
        return count_words(self.text)

    @property
    def estimated_seconds(self) -> float:
        return estimate_speech_seconds(self.text)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["word_count"] = self.word_count
        data["estimated_seconds"] = round(self.estimated_seconds, 1)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            segment_id=data["segment_id"],
            chapter_id=data.get("chapter_id", ""),
            index=int(data["index"]),
            text=data["text"],
            kind=data.get("kind", "narration"),
            char_count=int(data.get("char_count", 0)),
            paragraph_index=int(data.get("paragraph_index", 0)),
            paragraph_span=int(data.get("paragraph_span", 1)),
            unit_count=int(data.get("unit_count", 1)),
            source_text=data.get("source_text", ""),
        )


# --------------------------------------------------------------------------- #
# 合成层
# --------------------------------------------------------------------------- #


@dataclass
class SegmentRecord:
    """一段音频的生成结果。

    Attributes:
        segment_id: 对应 :class:`Segment`。
        audio_path: 相对项目根目录的 wav 路径（存相对路径，方便整个目录搬家）。
        duration: 音频时长（秒）。
        sample_rate: 采样率。
        attempts: 实际尝试次数（> 1 说明经历过重试）。
        skipped: ``True`` 表示本次运行因文件已存在而跳过。
        error: 失败原因；成功时为 ``None``。
    """

    segment_id: str
    audio_path: str
    duration: float = 0.0
    sample_rate: int = 0
    attempts: int = 1
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.duration > 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentRecord":
        return cls(
            segment_id=data["segment_id"],
            audio_path=data.get("audio_path", ""),
            duration=float(data.get("duration", 0.0)),
            sample_rate=int(data.get("sample_rate", 0)),
            attempts=int(data.get("attempts", 1)),
            skipped=bool(data.get("skipped", False)),
            error=data.get("error"),
        )


@dataclass
class ChapterAudioManifest:
    """一章的合成清单，落盘为 ``output/audio/<chapter_id>/manifest.json``。

    有了它，中断之后重跑可以精确知道“哪些段已经好了、哪些失败了”，
    而不需要去猜或者重新读所有 wav 的时长。
    """

    chapter_id: str
    title: str
    text_sha1: str
    created_at: str
    updated_at: str
    sample_rate: int
    segment_count: int
    records: list[SegmentRecord] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- 统计
    @property
    def done_count(self) -> int:
        return sum(1 for r in self.records if r.ok)

    @property
    def failed(self) -> list[SegmentRecord]:
        return [r for r in self.records if r.error]

    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.records if r.ok)

    def record_for(self, segment_id: str) -> SegmentRecord | None:
        for rec in self.records:
            if rec.segment_id == segment_id:
                return rec
        return None

    def upsert(self, record: SegmentRecord) -> None:
        """插入或替换一条记录（按 ``segment_id``）。"""
        for i, existing in enumerate(self.records):
            if existing.segment_id == record.segment_id:
                self.records[i] = record
                return
        self.records.append(record)

    # --------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "text_sha1": self.text_sha1,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sample_rate": self.sample_rate,
            "segment_count": self.segment_count,
            "done_count": self.done_count,
            "failed_count": len(self.failed),
            "total_duration": round(self.total_duration, 3),
            "settings": self.settings,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChapterAudioManifest":
        return cls(
            chapter_id=data["chapter_id"],
            title=data.get("title", ""),
            text_sha1=data.get("text_sha1", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            sample_rate=int(data.get("sample_rate", 0)),
            segment_count=int(data.get("segment_count", 0)),
            records=[SegmentRecord.from_dict(r) for r in data.get("records", [])],
            settings=data.get("settings", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ChapterAudioManifest | None":
        """从 JSON 载入；文件不存在或损坏时返回 ``None``。

        Note:
            ``json.JSONDecodeError`` 是 ``ValueError`` 的子类，但为了明确表达
            「损坏的清单不应该让整条流水线崩掉」，这里把它单独列出来。
        """
        import json

        from .utils import read_json

        try:
            data = read_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            from .utils import get_logger

            get_logger("models").warning("清单文件损坏，已忽略：%s（%s）", path, exc)
            return None

        if not isinstance(data, dict):
            return None
        try:
            return cls.from_dict(data)
        except (KeyError, TypeError, ValueError):
            return None


# --------------------------------------------------------------------------- #
# 书级别容器
# --------------------------------------------------------------------------- #


@dataclass
class Book:
    """解析结果：一本书的元信息 + 章节列表。"""

    title: str
    source_path: str
    language: str = "en"
    author: str = ""
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(c.char_count for c in self.chapters)

    @property
    def estimated_seconds(self) -> float:
        return sum(c.estimated_seconds for c in self.chapters)

    def get_chapter(self, key: str | int) -> Chapter:
        """按 ``chapter_id`` / 标题 / 序号取章。

        Args:
            key: ``"c003"``、``3``、``"Loss 3: Crimson Rhapsody"`` 或标题的唯一子串。

        Raises:
            KeyError: 找不到或匹配到多章。
        """
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            index = int(key)
            for chapter in self.chapters:
                if chapter.index == index:
                    return chapter
            raise KeyError(f"没有序号为 {index} 的章节")

        key_str = str(key).strip()
        for chapter in self.chapters:
            if chapter.chapter_id == key_str:
                return chapter
        exact = [c for c in self.chapters if c.title.strip().lower() == key_str.lower()]
        if len(exact) == 1:
            return exact[0]
        partial = [c for c in self.chapters if key_str.lower() in c.title.lower()]
        if len(partial) == 1:
            return partial[0]
        if not partial:
            raise KeyError(f"找不到章节 {key_str!r}")
        titles = ", ".join(repr(c.title) for c in partial[:5])
        raise KeyError(f"章节 {key_str!r} 匹配到多章：{titles}")

    def to_dict(self, *, include_segments: bool = True) -> dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "source_path": self.source_path,
            "language": self.language,
            "chapter_count": len(self.chapters),
            "total_chars": self.total_chars,
            "estimated_seconds": round(self.estimated_seconds, 1),
            "chapters": [c.to_dict(include_segments=include_segments) for c in self.chapters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Book":
        return cls(
            title=data.get("title", ""),
            author=data.get("author", ""),
            source_path=data.get("source_path", ""),
            language=data.get("language", "en"),
            chapters=[Chapter.from_dict(c) for c in data.get("chapters", [])],
        )
