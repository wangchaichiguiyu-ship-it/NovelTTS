"""EPUB / TXT 解析：把一本书拆成 :class:`~noveltts.models.Chapter` 列表。

只用标准库实现（``zipfile`` + ``xml.etree`` + ``html.parser``），刻意不依赖
``lxml`` / ``beautifulsoup4`` / ``ebooklib``——这几个包在你的 conda 环境里都没有，
而云端环境也可能装不上。

解析流程：

1. 读 ``META-INF/container.xml`` 找到 ``.opf``；
2. 从 ``.opf`` 读 ``<manifest>`` 与 ``<spine>``，得到**阅读顺序**的 XHTML 列表；
3. 读 EPUB3 的 ``nav``（``navtoc.xhtml``）或 EPUB2 的 ``toc.ncx``，拿到标题与锚点；
4. 按阅读顺序遍历文档、提取正文段落，把落在同一章节锚点区间内的段落归到一章；
5. 短单元（版权页/图片页）与目录类单元按配置过滤；
6. 兜底：如果 EPUB 的目录信息完全不可用，就退化为“一个 XHTML 文档 = 一章”，
   并用文档里的 ``<h1>/<h2>/<title>`` 猜标题。

TXT 输入（如 ``sources/*.txt``）走 :func:`parse_txt`：按行扫描，
把 ``Loss 1: X`` / ``Chapter 3`` / ``序章`` 这类标题行识别为切章点。
"""

from __future__ import annotations

import html
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from .config import ParseConfig
from .models import Book, Chapter, Paragraph
from .utils import count_chars, get_logger

logger = get_logger("parser")

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "epub": "http://www.idpf.org/2007/ops",
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
}

#: XHTML 里被视为“块级正文”的标签；遇到就换段。
_BLOCK_TAGS = {
    "p", "div", "br", "li", "blockquote", "section", "article", "h1", "h2",
    "h3", "h4", "h5", "h6", "tr", "td", "th", "figcaption", "dd", "dt", "pre",
}

#: 需要完全丢弃内容的标签（脚本、样式、导航）。
_DROP_TAGS = {"script", "style", "head", "title", "meta", "link", "svg", "nav"}

#: 用于猜标题的标签优先级。
_TITLE_TAGS = ("h1", "h2", "h3", "h4")

#: 章节标题最多可能被排版成几行（用于 :func:`_clean_heading_block`）。
_MAX_TITLE_LINES = 4

#: 判断段落是否为对话（开头是引号）。
_DIALOGUE_START_RE = re.compile(r"^[\"\u201c\u2018\u300c\u300e\u2014\-]+")

#: TXT 切章标题模式。
_TXT_HEADING_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:prologue|epilogue|foreword|afterword|interlude)\b.*$", re.I),
    re.compile(r"^(?:loss|chapter|part|book|volume|vol\.?|act|scene)\s+[0-9ivxlcdm]+\b.*$", re.I),
    re.compile(r"^(?:intermission|appendix|bonus)\b.*$", re.I),
    re.compile(r"^第[0-9一二三四五六七八九十百千]+[章回节卷篇部]\b.*$"),
    re.compile(r"^(?:序章|序言|后记|尾声|终章|番外|前言)\b.*$"),
    re.compile(r"^-{3,}\s*(?:loss|chapter|part)\b.*$", re.I),
)


# --------------------------------------------------------------------------- #
# 轻量 HTML -> 段落
# --------------------------------------------------------------------------- #


@dataclass
class _ExtractedDoc:
    """一个 XHTML 文档的提取结果。"""

    paragraphs: list[Paragraph] = field(default_factory=list)
    title: str = ""
    heading_ids: list[tuple[str, str]] = field(default_factory=list)
    """``(element_id, heading_text)``：文档内的小标题锚点，用于 EPUB 目录锚点对齐。"""

    @property
    def text_length(self) -> int:
        return sum(count_chars(p.text) for p in self.paragraphs)


class _XHTMLTextExtractor(HTMLParser):
    """把 XHTML 抽成纯文本段落。

    用 :class:`html.parser.HTMLParser` 而不是正则，因为要正确处理嵌套标签、
    未闭合标签和实体转义。``convert_charrefs=True`` 会自动解码 ``&amp;`` 之类。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[Paragraph] = []
        self.title: str = ""
        self.heading_ids: list[tuple[str, str]] = []

        self._buf: list[str] = []
        self._drop_depth = 0
        self._heading_tag: str | None = None
        self._heading_id: str | None = None
        self._in_title = False
        self._current_kind: str = "narration"

    # ------------------------------------------------------------- 标签事件
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}

        # ``<title>`` 要单独先处理：它也在 _DROP_TAGS 里（正文中当然不该出现），
        # 但我们**需要**它的文本来当章节标题。放在 _DROP_TAGS 判断之后处理的话，
        # 会因为 _drop_depth 递增而永远读不到内容（曾经导致回退切章时标题变成
        # 文件名 "a"/"b"）。
        if tag == "title":
            self._in_title = True
            self._drop_depth += 1
            return

        if tag in _DROP_TAGS:
            # <nav> 里通常还有真正的正文？不，nav 是目录，整块丢。
            self._drop_depth += 1
            return
        if self._drop_depth:
            return

        if tag == "img":
            # 图片页没有可读文本；保留一个空段落以便后续按长度过滤掉。
            return
        if tag in _TITLE_TAGS:
            self._flush()
            self._heading_tag = tag
            self._heading_id = attr_map.get("id") or None
            return
        if tag in _BLOCK_TAGS:
            self._flush()
        if tag == "br":
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if tag in _DROP_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if self._drop_depth:
            return
        if tag in _TITLE_TAGS and self._heading_tag == tag:
            text = self._flush()
            if text:
                self.heading_ids.append((self._heading_id or "", text))
                self.paragraphs.append(Paragraph(text=text, kind="heading"))
            self._heading_tag = None
            self._heading_id = None
            return
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._drop_depth:
            return
        self._buf.append(data)

    # --------------------------------------------------------------- 内部
    def _flush(self) -> str:
        """把缓冲区内容作为一个段落提交。"""
        raw = "".join(self._buf)
        self._buf = []
        text = normalize_whitespace(raw)
        if not text:
            return ""
        kind = "heading" if self._heading_tag else (
            "dialogue" if _DIALOGUE_START_RE.match(text) else "narration"
        )
        self.paragraphs.append(Paragraph(text=text, kind=kind))  # type: ignore[arg-type]
        return text

    def close(self) -> None:  # noqa: D102 - 继承文档
        super().close()
        self._flush()


def normalize_whitespace(text: str) -> str:
    """折叠空白：把各种空白字符统一成单空格，同时保留软连字符的“接词”语义。

    ``Campioni\\xadonships`` 这类带软连字符（U+00AD）的断词是 Seven Seas
    电子书的常见产物，这里直接删掉软连字符让词还原成 ``Championships``。
    """
    text = text.replace("\u00ad", "")          # soft hyphen：直接删除
    text = text.replace("\ufeff", "")          # BOM
    text = text.replace("\u00a0", " ")         # NBSP
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_document(xhtml: bytes | str) -> _ExtractedDoc:
    """把一个 XHTML/HTML 文档抽成段落列表。

    Args:
        xhtml: 原始字节或字符串。

    Returns:
        :class:`_ExtractedDoc`。解析失败时返回空结果而不是抛异常——
        电子书里常见的畸形 XHTML 不应该让整条流水线中断。
    """
    if isinstance(xhtml, bytes):
        text = _decode_bytes(xhtml)
    else:
        text = xhtml

    parser = _XHTMLTextExtractor()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - 单篇文档失败不应中断整本书
        logger.warning("HTML 解析异常（已尽力恢复）：%s", exc)

    doc = _ExtractedDoc(
        paragraphs=parser.paragraphs,
        title=normalize_whitespace(parser.title),
        heading_ids=parser.heading_ids,
    )
    return doc


def _decode_bytes(data: bytes) -> str:
    """按 ``<meta charset>`` 声明或 BOM 解码，默认 UTF-8。"""
    head = data[:2048].decode("ascii", errors="ignore")
    match = re.search(r'charset\s*=\s*["\']?([\w-]+)', head, re.I)
    encodings = [match.group(1)] if match else []
    encodings += ["utf-8", "utf-16", "gb18030", "big5", "latin-1"]
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# EPUB
# --------------------------------------------------------------------------- #


@dataclass
class _TocEntry:
    """目录条目。"""

    title: str
    href: str           # 相对 opf 的路径，已去掉 fragment
    fragment: str = ""  # ``#`` 后面的锚点 id


def parse_epub(path: str | Path, config: ParseConfig | None = None) -> Book:
    """解析 EPUB，返回按阅读顺序切好的章节。

    Args:
        path: ``.epub`` 文件路径。
        config: 切章配置；``None`` 时用默认值。

    Returns:
        :class:`~noveltts.models.Book`。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 不是合法的 EPUB（缺 container.xml / opf）。
    """
    config = config or ParseConfig()
    epub_path = Path(path)
    if not epub_path.is_file():
        raise FileNotFoundError(f"找不到 EPUB：{epub_path}")

    with zipfile.ZipFile(epub_path) as archive:
        names = set(archive.namelist())
        opf_path = _find_opf_path(archive, names)
        opf_dir = posixpath.dirname(opf_path)
        opf_root = _parse_xml(archive.read(opf_path))

        metadata = _read_metadata(opf_root)
        spine_hrefs = _read_spine(opf_root)
        manifest = _read_manifest(opf_root)
        toc_entries = _read_toc(archive, names, opf_dir, opf_root)

        # 把 spine 里的 idref 映射成实际路径
        doc_paths: list[str] = []
        for idref in spine_hrefs:
            href = manifest.get(idref)
            if href:
                doc_paths.append(_resolve_href(opf_dir, href))

        if not doc_paths:  # 兜底：没有可用 spine，就按文件名顺序遍历所有 XHTML
            logger.warning("EPUB spine 为空，回退到按文件名顺序扫描 XHTML")
            doc_paths = sorted(
                name for name in names if name.lower().endswith((".xhtml", ".html", ".htm"))
            )

        documents: dict[str, _ExtractedDoc] = {}
        for doc_path in doc_paths:
            if doc_path not in names:
                logger.debug("spine 指向的文件不存在，跳过：%s", doc_path)
                continue
            documents[doc_path] = extract_document(archive.read(doc_path))

    chapters = _assemble_chapters(
        documents=documents,
        toc_entries=toc_entries,
        opf_dir=opf_dir,
        config=config,
    )

    language = metadata.get("language") or config.default_language
    book = Book(
        title=metadata.get("title") or epub_path.stem,
        author=metadata.get("creator", ""),
        source_path=str(epub_path.resolve()),
        language=_normalize_language(language),
        chapters=chapters,
    )
    logger.info(
        "EPUB 解析完成：%s（%d 章，约 %d 字，预估 %s）",
        book.title, len(book.chapters), book.total_chars,
        _format_estimate(book.estimated_seconds),
    )
    return book


def _find_opf_path(archive: zipfile.ZipFile, names: set[str]) -> str:
    """定位 ``.opf`` 路径。"""
    container_name = "META-INF/container.xml"
    if container_name in names:
        root = _parse_xml(archive.read(container_name))
        for rootfile in root.iter():
            if rootfile.tag.endswith("rootfile"):
                full_path = rootfile.get("full-path")
                if full_path:
                    return full_path
    for name in names:
        if name.lower().endswith(".opf"):
            return name
    raise ValueError("EPUB 中找不到 .opf 文件，可能不是标准 EPUB")


def _parse_xml(data: bytes) -> ET.Element:
    """解析 XML，容忍未声明命名空间前缀之类的常见错误。"""
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        # 去掉 DOCTYPE 后重试（部分电子书带外部 DTD 引用）
        text = re.sub(rb"<!DOCTYPE[^>]*>", b"", data, count=1)
        return ET.fromstring(text)


def _read_metadata(opf_root: ET.Element) -> dict[str, str]:
    """读取 dc:title / dc:creator / dc:language。"""
    result: dict[str, str] = {}
    for element in opf_root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag in {"title", "creator", "language"} and element.text:
            result.setdefault(tag, element.text.strip())
    return result


def _read_manifest(opf_root: ET.Element) -> dict[str, str]:
    """``id -> href``。"""
    manifest: dict[str, str] = {}
    for element in opf_root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() != "item":
            continue
        item_id = element.get("id")
        href = element.get("href")
        if item_id and href:
            manifest[item_id] = href
    return manifest


def _read_spine(opf_root: ET.Element) -> list[str]:
    """按顺序返回 spine 中的 idref 列表。"""
    for element in opf_root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() != "spine":
            continue
        return [
            child.get("idref", "")
            for child in element
            if child.tag.rsplit("}", 1)[-1].lower() == "itemref"
        ]
    return []


def _read_toc(
    archive: zipfile.ZipFile,
    names: set[str],
    opf_dir: str,
    opf_root: ET.Element,
) -> list[_TocEntry]:
    """读取 EPUB3 ``nav`` 或 EPUB2 ``toc.ncx`` 目录。

    EPUB3 优先；两者都拿不到时返回空列表（后续退化为按文档切章）。
    """
    # --- EPUB3 nav ---
    nav_href: str | None = None
    manifest = _read_manifest(opf_root)
    for element in opf_root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() != "item":
            continue
        properties = (element.get("properties") or "").split()
        if "nav" in properties and element.get("href"):
            nav_href = element.get("href")
            break

    if nav_href:
        nav_path = _resolve_href(opf_dir, nav_href)
        if nav_path in names:
            entries = _parse_nav_document(archive.read(nav_path), posixpath.dirname(nav_path))
            if entries:
                return entries

    # --- EPUB2 ncx ---
    ncx_path: str | None = None
    spine = next(
        (e for e in opf_root.iter() if e.tag.rsplit("}", 1)[-1].lower() == "spine"), None
    )
    if spine is not None and spine.get("toc"):
        href = manifest.get(spine.get("toc", ""))
        if href:
            ncx_path = _resolve_href(opf_dir, href)
    if ncx_path is None:
        for name in names:
            if name.lower().endswith(".ncx"):
                ncx_path = name
                break
    if ncx_path and ncx_path in names:
        return _parse_ncx(archive.read(ncx_path), posixpath.dirname(ncx_path))

    logger.warning("EPUB 未提供可用目录（nav/ncx），将按文档边界切章")
    return []


def _parse_nav_document(data: bytes, base_dir: str) -> list[_TocEntry]:
    """解析 EPUB3 nav 文档里的 ``<nav epub:type="toc">``。"""
    root = _parse_xml(data)
    entries: list[_TocEntry] = []

    def walk(element: ET.Element) -> None:
        for child in element:
            tag = child.tag.rsplit("}", 1)[-1].lower()
            if tag == "a":
                href = child.get("href", "")
                title = "".join(child.itertext()).strip()
                if href:
                    path, _, fragment = href.partition("#")
                    entries.append(
                        _TocEntry(
                            title=normalize_whitespace(title),
                            href=_resolve_href(base_dir, path) if path else "",
                            fragment=fragment,
                        )
                    )
            else:
                walk(child)

    # 只找 toc 类型的 nav，避免把 landmarks 当成目录
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag != "nav":
            continue
        nav_type = ""
        for key, value in element.attrib.items():
            if key.rsplit("}", 1)[-1].lower() == "type":
                nav_type = value
        if nav_type in {"toc", "doc-toc"} or not entries:
            walk(element)
    return entries


def _parse_ncx(data: bytes, base_dir: str) -> list[_TocEntry]:
    """解析 EPUB2 ncx 的 ``navMap``（保留层级顺序）。"""
    root = _parse_xml(data)
    entries: list[_TocEntry] = []
    for point in root.iter():
        if point.tag.rsplit("}", 1)[-1].lower() != "navpoint":
            continue
        title = ""
        src = ""
        for child in point.iter():
            tag = child.tag.rsplit("}", 1)[-1].lower()
            if tag == "text" and child.text and not title:
                title = child.text.strip()
            elif tag == "content" and child.get("src") and not src:
                src = child.get("src", "")
        if not src:
            continue
        path, _, fragment = src.partition("#")
        entries.append(
            _TocEntry(
                title=normalize_whitespace(title),
                href=_resolve_href(base_dir, path) if path else "",
                fragment=fragment,
            )
        )
    return entries


def _resolve_href(base_dir: str, href: str) -> str:
    """把相对 href 解析成 zip 内的绝对路径（posix 风格）。"""
    href = href.split("#", 1)[0]
    if not href:
        return ""
    joined = posixpath.join(base_dir, href) if base_dir else href
    return posixpath.normpath(joined).lstrip("./")


# --------------------------------------------------------------------------- #
# 章节组装
# --------------------------------------------------------------------------- #


def _assemble_chapters(
    documents: dict[str, _ExtractedDoc],
    toc_entries: list[_TocEntry],
    opf_dir: str,
    config: ParseConfig,
) -> list[Chapter]:
    """把“文档序列 + 目录锚点”拼成章节列表。

    核心思路：目录条目给出 ``(文件, 锚点, 标题)``。按阅读顺序把每个条目的
    起点映射到 ``(文档序号, 文档内段落序号)``，相邻起点之间的所有段落就属于
    该条目对应的章节。目录不可用时，退化成“一个文档 = 一章”。
    """
    doc_order = list(documents.keys())
    doc_index = {path: i for i, path in enumerate(doc_order)}

    # 1) 把目录条目映射成 (doc_pos, para_pos, title)
    marks: list[tuple[int, int, str]] = []
    for entry in toc_entries:
        doc_pos = doc_index.get(entry.href)
        if doc_pos is None:
            continue
        para_pos = _locate_fragment(documents[entry.href], entry.fragment)
        marks.append((doc_pos, para_pos, entry.title))

    marks.sort(key=lambda item: (item[0], item[1]))
    # 去掉完全重复的起点
    deduped: list[tuple[int, int, str]] = []
    for mark in marks:
        if deduped and deduped[-1][0] == mark[0] and deduped[-1][1] == mark[1]:
            continue
        deduped.append(mark)
    marks = deduped

    if not marks:
        return _chapters_by_document(documents, config)

    # 2) 按起点切分
    #    起点列表需要补一个“全书结尾”哨兵
    total_docs = len(doc_order)
    markers = marks + [(total_docs, 0, "")]
    raw_chapters: list[tuple[str, list[Paragraph], str]] = []

    for i in range(len(markers) - 1):
        start_doc, start_para, title = markers[i]
        end_doc, end_para, _ = markers[i + 1]
        collected: list[Paragraph] = []
        sources: list[str] = []

        for doc_pos in range(start_doc, min(end_doc, total_docs - 1) + 1):
            doc = documents[doc_order[doc_pos]]
            paragraphs = doc.paragraphs
            lo = start_para if doc_pos == start_doc else 0
            hi = end_para if doc_pos == end_doc else len(paragraphs)
            collected.extend(paragraphs[lo:hi])
            if paragraphs[lo:hi]:
                sources.append(doc_order[doc_pos])

        # 目录条目自身的标题如果不在正文里，补一个 heading 段落作为开头
        title = title or (documents[doc_order[start_doc]].title if doc_order else "")
        if title and not _title_present(collected, title):
            collected.insert(0, Paragraph(text=title, kind="heading"))

        raw_chapters.append((title, collected, sources[0] if sources else ""))

    return _finalize_chapters(raw_chapters, config)


def _locate_fragment(doc: _ExtractedDoc, fragment: str) -> int:
    """把锚点 id 映射成文档内的段落序号。找不到就返回 0。"""
    if not fragment:
        return 0
    for index, (element_id, _text) in enumerate(doc.heading_ids):
        if element_id == fragment:
            # heading 段落本身在 paragraphs 里的位置需要现算
            return _paragraph_index_of_heading(doc, index)
    return 0


def _paragraph_index_of_heading(doc: _ExtractedDoc, heading_ordinal: int) -> int:
    """返回第 ``heading_ordinal`` 个 heading 段落所在的段落下标。"""
    seen = 0
    for index, paragraph in enumerate(doc.paragraphs):
        if paragraph.kind == "heading":
            if seen == heading_ordinal:
                return index
            seen += 1
    return 0


def _title_present(paragraphs: Iterable[Paragraph], title: str) -> bool:
    """判断标题是否已经作为段落出现过（避免重复插入）。"""
    normalized = _loose(title)
    if not normalized:
        return True
    for paragraph in paragraphs:
        if _loose(paragraph.text).startswith(normalized[:40]):
            return True
    return False


def _loose(text: str) -> str:
    """宽松比较用的归一化：小写、去标点空白。"""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.lower())


def _chapters_by_document(
    documents: dict[str, _ExtractedDoc],
    config: ParseConfig,
) -> list[Chapter]:
    """兜底切章：一个 XHTML 文档 = 一章。"""
    raw: list[tuple[str, list[Paragraph], str]] = []
    for path, doc in documents.items():
        if not doc.paragraphs:
            continue
        title = doc.title or _guess_title_from_paragraphs(doc.paragraphs) or Path(path).stem
        raw.append((title, list(doc.paragraphs), path))
    return _finalize_chapters(raw, config)


def _guess_title_from_paragraphs(paragraphs: list[Paragraph]) -> str:
    """从开头几个段落里猜标题。"""
    for paragraph in paragraphs[:4]:
        if paragraph.kind == "heading":
            return paragraph.text
    for paragraph in paragraphs[:2]:
        if len(paragraph.text) <= 60:
            return paragraph.text
    return ""


def _finalize_chapters(
    raw_chapters: list[tuple[str, list[Paragraph], str]],
    config: ParseConfig,
) -> list[Chapter]:
    """过滤短/无用单元、丢弃样板段落、决定是否并入上一章，并生成最终章节。"""
    skip_res = [re.compile(pattern, re.I) for pattern in config.skip_title_patterns]
    kept: list[tuple[str, list[Paragraph], str]] = []

    for title, paragraphs, source in raw_chapters:
        paragraphs = _drop_boilerplate(title, paragraphs, config)
        body_chars = sum(count_chars(p.text) for p in paragraphs if p.kind != "heading")

        if body_chars == 0:
            logger.debug("跳过纯图片/空单元：%s", title or source)
            continue

        is_nav_like = any(pattern.search(title.strip()) for pattern in skip_res) if title else False
        if config.drop_reference_sections and is_nav_like:
            logger.debug("跳过非正文单元：%s", title)
            continue

        if body_chars < config.min_chapter_chars:
            if config.merge_short_into_previous and kept:
                logger.debug("短单元 %r（%d 字）并入上一章", title, body_chars)
                kept[-1][1].extend(paragraphs)
                continue
            logger.debug("跳过过短单元 %r（%d 字）", title, body_chars)
            continue

        kept.append((title, list(paragraphs), source))

    chapters: list[Chapter] = []
    for ordinal, (title, paragraphs, source) in enumerate(kept, start=1):
        final_title = title or f"Chapter {ordinal}"
        chapters.append(
            Chapter(
                index=ordinal,
                title=final_title,
                paragraphs=_clean_heading_block(final_title, paragraphs),
                source=source,
                language=config.default_language,
            )
        )
    return chapters


def _drop_boilerplate(
    title: str,
    paragraphs: list[Paragraph],
    config: ParseConfig,
) -> list[Paragraph]:
    """按 ``exclude_paragraph_patterns`` 丢掉样板段落（版权、ISBN、许可声明…）。

    刻意保持**段落级**而非章节级：样板句子常常和正文挤在同一个 XHTML 文档里
    （例如后记末尾的版权声明），整章丢弃会误伤真正的正文。

    Important:
        如果某个单元的所有段落都命中规则，函数会**原样返回并给出警告**——
        宁可多读几句版权声明，也不能让一章凭空消失。
    """
    patterns = getattr(config, "exclude_paragraph_patterns", None)
    if not patterns:
        return paragraphs

    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.I))
        except re.error as exc:
            logger.warning("exclude_paragraph_patterns 正则非法，已跳过：%s（%s）", pattern, exc)
    if not compiled:
        return paragraphs

    kept: list[Paragraph] = []
    dropped = 0
    for paragraph in paragraphs:
        if paragraph.kind == "heading":
            kept.append(paragraph)
            continue
        if any(pattern.search(paragraph.text) for pattern in compiled):
            dropped += 1
            logger.debug("丢弃样板段落（%s）：%s", title or "无标题", paragraph.text[:70])
            continue
        kept.append(paragraph)

    if dropped and not kept:
        logger.warning(
            "单元 %r 的全部 %d 段都命中了样板规则；为避免整章消失将保留原文，"
            "请检查 parse.exclude_paragraph_patterns 是否过于宽泛",
            title,
            dropped,
        )
        return paragraphs

    if dropped:
        logger.debug("单元 %r 丢弃了 %d 段样板文字", title or "无标题", dropped)
    return kept


def _clean_heading_block(title: str, paragraphs: list[Paragraph]) -> list[Paragraph]:
    """清理章节开头的标题区，避免 TTS 把标题读两遍。

    电子书里标题的排版方式五花八门，这里是实际遇到过的四种：

    1. ``<h1>Intermission: Appeal to Emotion</h1>`` 之后再出现一份同名标题；
    2. ``<p>Loss 1:<br/>The Purest of Intentions</p>`` —— 标题被 ``<br/>`` 拆成两段；
    3. ``<h1>`` 缺失，标题只存在于目录，由 :func:`_assemble_chapters` 补了 heading
       段落，但正文里仍有原始标题；
    4. 标题与正文同处一段（``<p>Loss 1: The Purest...<br/>THE DAYS WERE...</p>``）。

    处理流程：

    * **第一步**：从开头连续的行里拼出一个候选标题（最多 3 行、累计 120 字符），
      若它足以覆盖 ``title``，就把这些行删掉，插入一个规范的 heading 段落。
      这一步同时解决了 1/2/3/4。
    * **第二步（兜底）**：候选拼不出标题时，退回“逐段消费标题字符”的老算法，
      只在标题被完整消费时才压缩，保证绝不误删正文。

    Important:
        两步都要求标题被**完整**匹配，否则原样返回。宁可多读一遍标题，
        也不能把正文吃掉。
    """
    if not paragraphs:
        return paragraphs

    target = _loose(title)
    if not target:
        return paragraphs

    # ---------------------------------------------------------------- 第一步
    # 把开头几行拼成候选标题。难点在于：正文段落也可能“碰巧”能按顺序匹配上
    # 标题里的字符（例如 "THE DAYS WERE..." 的若干字母在 "Loss 1: The Purest
    # of Intentions" 里恰好都能找到）。所以判定规则定得很死：
    #
    #   * 段落必须**完整地**落在标题的剩余字符里，一个字都不能超出；
    #   * 段落消费掉的标题长度必须超过其自身长度的一半——真正的标题片段会
    #     消费 ≈100%（标点空白不计入长度），而正文段只能消费到个位数比例，
    #     于是被判定为「正文，标题区到此结束」；
    #   * 消费长度为 0 的段落是 ``<br/>`` / 重复排版产生的标题残留
    #     （例如第二份 "Loss 1:"、"Foreword"），计入标题区但不重复朗读。
    #
    # 任何一条不满足即停止，绝不猜测、绝不吞正文。
    #
    # 另外要处理一个顺序陷阱：当 ``<h1>`` 里已经写全了标题（gain 直接覆盖
    # target），循环会在第一段就结束，后面的重复残留就漏掉了。因此这里用
    # `covered` 标志代替 break，让循环继续往下吸收残留，直到遇到正文为止。
    scan_limit = min(len(paragraphs), _MAX_TITLE_LINES + 2)
    consumed_upto = 0
    candidate_cursor = 0
    candidate_lines: list[str] = []
    covered = False

    for index, paragraph in enumerate(paragraphs[:scan_limit]):
        normalized = _loose(paragraph.text)
        if not normalized:
            consumed_upto = index + 1
            continue

        if covered:
            # 标题已经拼全，接下来只可能是重复的标题残留或正文。
            # 凡是已经出现在候选行里的内容都算残留，一并吸收；
            # 否则就是正文，标题区到此为止。
            if any(normalized in line or line in normalized for line in candidate_lines):
                consumed_upto = index + 1
                continue
            break

        position = candidate_cursor
        matched = True
        for char in normalized:
            position = target.find(char, position)
            if position < 0:
                matched = False
                break
            position += 1
        if not matched:
            break

        gained = position - candidate_cursor
        if gained == 0:
            # 标题还没拼全却出现零增益的行 -> 排版异常，保守退出
            break
        if gained * 2 < len(normalized):
            # 只消费到很小比例 -> 正文（或段落里混了正文），标题区结束
            break

        candidate_lines.append(normalized)
        candidate_cursor = position
        consumed_upto = index + 1
        if candidate_cursor >= len(target):
            covered = True

    if covered and consumed_upto > 0:
        head = Paragraph(text=title, kind="heading", source=paragraphs[0].source)
        return [head, *paragraphs[consumed_upto:]]

    # ---------------------------------------------------------------- 第二步
    cursor = 0
    consumed_upto = 0
    for index, paragraph in enumerate(paragraphs):
        normalized = _loose(paragraph.text)
        if not normalized:
            consumed_upto = index + 1
            continue

        position = cursor
        matched = True
        for char in normalized:
            position = target.find(char, position)
            if position < 0:
                matched = False
                break
            position += 1

        # 要求“该段落的大部分字符都落进标题”。否则像 "AFTER SCHOOL..." 这种正文
        # 会因为 'l'、'o'、's' 碰巧出现在 "Loss 1: The Purest of Intentions" 里
        # 而被误吞，导致标题区提前结束、正文被删。
        if not matched or (position - cursor) * 2 < len(normalized):
            break

        cursor = position
        consumed_upto = index + 1
        if cursor >= len(target):
            break

    if cursor < len(target) or consumed_upto == 0:
        return paragraphs

    head = Paragraph(text=title, kind="heading", source=paragraphs[0].source)
    return [head, *paragraphs[consumed_upto:]]


# --------------------------------------------------------------------------- #
# TXT
# --------------------------------------------------------------------------- #

_TXT_CHAPTER_RE = re.compile(
    r"^(?:"
    r"(?:loss|chapter|part|book|volume|act|intermission|prologue|epilogue|foreword|"
    r"afterword|interlude|appendix|bonus)\b.*"
    r"|第[0-9一二三四五六七八九十百千]+[章回节卷篇部].*"
    r"|(?:序章|序言|后记|尾声|终章|番外|前言|目录)"
    r")$",
    re.I,
)

#: 扉页/目录类标题。它们本身是**独立成段**的，需要被识别出来（而不是被当成
#: 正文粘到下一章上），随后由 ``parse_txt`` 按 ``skip_title_patterns`` 丢掉。
_TXT_FRONT_MATTER_RE = re.compile(
    r"^(?:table of contents|contents|color gallery|title page|copyrights?|credits|"
    r"cover|newsletter|also available|about the (?:author|series))\b.*$",
    re.I,
)


def is_heading_line(line: str) -> bool:
    """判断一行文本是否是章节标题。

    刻意保守：只认“章节关键词 + 短行”的组合，避免把 ``"Chapter one was over."``
    这种正文误判成标题。扉页/目录类标题（``Table of Contents``）也会被认出来，
    好让它们独立成段、随后被 :func:`parse_txt` 丢掉，而不是粘到下一章开头。
    """
    stripped = line.strip().strip("*_# ").strip()
    if not stripped or len(stripped) > 80:
        return False
    if _TXT_FRONT_MATTER_RE.match(stripped):
        return True
    if not _TXT_CHAPTER_RE.match(stripped):
        return False
    # 标题一般不以句末标点结束
    if stripped.endswith((".", "。", "!", "！", "?", "？", '"', "\u201d")):
        return False
    # 纯 "Chapter" 单词不算
    return len(stripped.split()) <= 12


def parse_txt(
    path: str | Path,
    config: ParseConfig | None = None,
    *,
    heading_re: str | None = None,
) -> Book:
    """解析纯文本小说。

    切章规则：
      * 默认用 :func:`is_heading_line` 识别标题行；
      * 传入 ``heading_re`` 时用自定义正则；
      * 一个标题行到下一个标题行之间算一章。

    Args:
        path: ``.txt`` 路径。
        config: 切章配置。
        heading_re: 自定义标题正则（整行匹配）。

    Returns:
        :class:`~noveltts.models.Book`。
    """
    config = config or ParseConfig()
    txt_path = Path(path)
    if not txt_path.is_file():
        raise FileNotFoundError(f"找不到文本文件：{txt_path}")

    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    matcher = re.compile(heading_re, re.I) if heading_re else None

    def hit(line: str) -> bool:
        return bool(matcher.match(line.strip())) if matcher else is_heading_line(line)

    # 1) 按标题行分段
    raw_chapters: list[tuple[str, list[Paragraph], str]] = []
    current_title = ""
    current_lines: list[str] = []

    def flush() -> None:
        paragraphs = _lines_to_paragraphs(current_lines)
        if paragraphs or current_title:
            raw_chapters.append((current_title, paragraphs, txt_path.name))

    for line in lines:
        if hit(line):
            flush()
            current_title = normalize_whitespace(line.strip().strip("*_# "))
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    # 丢掉「目录」「版权页」这类非正文单元，与 EPUB 的处理保持一致
    skip_res = [
        re.compile(pattern, re.I) for pattern in config.skip_title_patterns
    ] + [re.compile(r"^table of contents", re.I), re.compile(r"^目\s*录$")]

    def is_front_matter(title: str) -> bool:
        stripped = title.strip()
        if not stripped:
            return False
        return any(pattern.search(stripped) for pattern in skip_res)

    raw_chapters = [
        (title, paragraphs, source)
        for title, paragraphs, source in raw_chapters
        if not (config.drop_reference_sections and is_front_matter(title))
    ]
    if not raw_chapters:
        logger.warning("所有单元都被当作非正文过滤掉了，改为保留全部内容")
        raw_chapters = [(txt_path.stem, _lines_to_paragraphs(lines), txt_path.name)]

    # 2) 若识别不到任何标题，就整本作为一章
    meaningful = [c for c in raw_chapters if c[1]]
    if len(meaningful) <= 1:
        logger.warning("未识别到章节标题，整本作为单章处理")
        paragraphs = _lines_to_paragraphs(lines)
        raw_chapters = [(txt_path.stem, paragraphs, txt_path.name)]

    chapters = _finalize_chapters(raw_chapters, config)

    # TXT 里的标题常被排成多行（"Loss 1:" / "The Purest of Intentions"），
    # 这里把它们接回一个完整标题，否则界面上只会显示半截。
    for chapter in chapters:
        chapter.title = _join_multiline_title(chapter)

    book = Book(
        title=txt_path.stem,
        source_path=str(txt_path.resolve()),
        language=_normalize_language(config.default_language),
        chapters=chapters,
    )
    logger.info("TXT 解析完成：%d 章，约 %d 字", len(book.chapters), book.total_chars)
    return book


def _join_multiline_title(chapter: Chapter) -> str:
    """把被拆成多行的标题接回去。

    判定：标题以 ``:`` / ``-`` 结尾，或本身就是 ``Loss 1`` 这种「关键词+编号」，
    且紧随其后的段落很短、也不以句末标点结束，就把它并进标题。
    被并入标题的那一段会从正文里移除，避免 TTS 读两遍。
    """
    title = chapter.title.strip()
    if not title or not chapter.paragraphs:
        return title

    needs_continuation = title.endswith((":", "-", "\u2014")) or bool(
        re.match(r"^(?:Loss|Chapter|Part|Act|Book|Volume|Intermission)\s*\d*$", title, re.I)
    )
    if not needs_continuation:
        return title

    candidate = chapter.paragraphs[0]
    if candidate.kind == "heading":
        # 首段本身就是 heading，说明标题已经在正文里出现过
        return title
    if len(candidate.text) > 60:
        return title
    # 注意：这里不能因为候选段以 "?" 结尾就放弃——"Loss 4:" 的续行正是
    # "Enemy of Women Who?"。只排除明确的句子收尾标点。
    if candidate.text.endswith((".", "!", "\u3002", "\uff01", "\uff1f")):
        return title

    merged = f"{title.rstrip(':-\u2014 ').strip()}: {candidate.text}".strip()
    chapter.paragraphs = chapter.paragraphs[1:]
    return merged


def _lines_to_paragraphs(lines: list[str]) -> list[Paragraph]:
    """把行列表变成段落列表：空行分段，单行内的连续行合并为一段。"""
    paragraphs: list[Paragraph] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        text = normalize_whitespace(" ".join(buffer))
        buffer.clear()
        if not text:
            return
        kind = "dialogue" if _DIALOGUE_START_RE.match(text) else "narration"
        paragraphs.append(Paragraph(text=text, kind=kind))  # type: ignore[arg-type]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_buffer()
            continue
        if _looks_like_heading(stripped):
            flush_buffer()
            paragraphs.append(Paragraph(text=stripped, kind="heading"))
            continue
        buffer.append(stripped)
    flush_buffer()
    return paragraphs


def _looks_like_heading(line: str) -> bool:
    """段内标题（不如 :func:`is_heading_line` 严格：允许 "Loss 1: ..."）。"""
    return is_heading_line(line) or bool(re.match(r"^(?:Loss|Chapter)\s+\d+\s*:", line, re.I))


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #


def parse_book(
    path: str | Path,
    config: ParseConfig | None = None,
    *,
    heading_re: str | None = None,
) -> Book:
    """按扩展名自动选择解析器。

    Args:
        path: ``.epub`` / ``.txt`` / ``.md`` 路径。
        config: 切章配置。
        heading_re: 仅 TXT 生效的自定义标题正则。

    Raises:
        ValueError: 不支持的扩展名。
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".epub":
        return parse_epub(target, config)
    if suffix in {".txt", ".md", ".text"}:
        return parse_txt(target, config, heading_re=heading_re)
    raise ValueError(f"不支持的文件类型：{suffix}（支持 .epub / .txt / .md）")


def find_source_files(source_dir: str | Path) -> list[Path]:
    """列出素材目录里的 epub/txt（忽略 macOS 的 ``._`` 资源分叉文件）。"""
    directory = Path(source_dir)
    if not directory.is_dir():
        return []
    files = [
        p
        for p in sorted(directory.iterdir())
        if p.suffix.lower() in {".epub", ".txt", ".md"} and not p.name.startswith("._")
    ]
    return files


def _normalize_language(value: str) -> str:
    """把 ``en-US`` / ``eng`` 这类写法压成两字母代码。"""
    value = (value or "").strip().lower()
    if not value:
        return "en"
    if value.startswith("zh") or value in {"chi", "chs", "cht", "chinese"}:
        return "zh"
    if value.startswith("ja") or value in {"jpn", "japanese"}:
        return "ja"
    if value.startswith("en") or value in {"eng", "english"}:
        return "en"
    return value[:2]


def _format_estimate(seconds: float) -> str:
    from .utils import format_duration

    return format_duration(seconds)
