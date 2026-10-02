"""解析模块测试：EPUB / TXT 切章、标题去重、异常容错。"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from noveltts.config import ParseConfig
from noveltts.epub_parser import (
    extract_document,
    find_source_files,
    is_heading_line,
    normalize_whitespace,
    parse_book,
    parse_epub,
    parse_txt,
)
from noveltts.models import Book, Chapter, Paragraph


# --------------------------------------------------------------------------- #
# 文本抽取
# --------------------------------------------------------------------------- #


def test_normalize_whitespace_removes_soft_hyphen() -> None:
    """Seven Seas 电子书常用软连字符断词，必须还原成一个词。"""
    assert normalize_whitespace("Champi\u00adonships") == "Championships"
    assert normalize_whitespace("  a\u00a0\u00a0b  ") == "a b"


def test_extract_document_paragraphs_and_heading() -> None:
    html = """<html><head><title>My Title</title></head><body>
    <h1>Chapter One</h1>
    <p>First paragraph.</p>
    <p>Second one. With two sentences.</p>
    <p><img src="x.jpg"/></p>
    <script>var x = 1;</script>
    </body></html>"""
    doc = extract_document(html)
    texts = [p.text for p in doc.paragraphs]
    assert "Chapter One" in texts
    assert "First paragraph." in texts
    assert "Second one. With two sentences." in texts
    assert "var x = 1;" not in " ".join(texts)
    assert doc.title == "My Title"


def test_extract_document_decodes_entities() -> None:
    doc = extract_document("<p>A &amp; B &lt;tag&gt; &#8212; dash</p>")
    assert doc.paragraphs[0].text == "A & B <tag> \u2014 dash"


def test_extract_document_handles_broken_html() -> None:
    """畸形 XHTML 不能让整条流水线挂掉。"""
    doc = extract_document("<p>Unclosed paragraph <b>bold <p>next")
    assert any("Unclosed paragraph" in p.text for p in doc.paragraphs)


def test_extract_document_marks_dialogue() -> None:
    doc = extract_document('<p>\u201cHello there.\u201d</p><p>Narration.</p>')
    assert doc.paragraphs[0].kind == "dialogue"
    assert doc.paragraphs[1].kind == "narration"


def test_extract_document_accepts_bytes_with_charset() -> None:
    raw = '<meta charset="utf-8"/><p>中文段落</p>'.encode("utf-8")
    doc = extract_document(raw)
    assert doc.paragraphs[0].text == "中文段落"


# --------------------------------------------------------------------------- #
# EPUB
# --------------------------------------------------------------------------- #


def test_parse_epub_basic(sample_epub: Path) -> None:
    book = parse_epub(sample_epub, ParseConfig(min_chapter_chars=1))
    assert book.title == "Test Novel"
    assert book.author == "Test Author"
    assert book.language == "en"
    assert len(book.chapters) == 2
    assert book.chapters[0].title == "Chapter One: The Start"
    assert book.chapters[1].title == "Chapter Two: The End"


def test_parse_epub_drops_image_only_section(sample_epub: Path) -> None:
    """纯图片页没有可读文本，必须被丢掉。"""
    book = parse_epub(sample_epub, ParseConfig(min_chapter_chars=1))
    assert all("Image" not in c.title for c in book.chapters)


def test_parse_epub_chapter_ids_are_sequential(sample_epub: Path) -> None:
    book = parse_epub(sample_epub, ParseConfig(min_chapter_chars=1))
    assert [c.chapter_id for c in book.chapters] == ["c001", "c002"]


def test_parse_epub_heading_is_deduplicated(sample_epub: Path) -> None:
    """``<h1>`` 里的标题不能在正文里再读一遍。"""
    book = parse_epub(sample_epub, ParseConfig(min_chapter_chars=1))
    chapter = book.chapters[0]
    headings = [p.text for p in chapter.paragraphs if p.kind == "heading"]
    assert headings == ["Chapter One: The Start"]
    body = " ".join(p.text for p in chapter.paragraphs if p.kind != "heading")
    assert body.count("Chapter One") == 0


def test_parse_epub_preserves_body_text(sample_epub: Path) -> None:
    book = parse_epub(sample_epub, ParseConfig(min_chapter_chars=1))
    text = book.chapters[0].text
    assert "1500 meter race" in text
    assert "Mr. Smith nodded." in text


def test_parse_epub_min_chars_filter(sample_epub: Path) -> None:
    """阈值设得比章节还大时，所有章节都会被过滤。"""
    book = parse_epub(sample_epub, ParseConfig(min_chapter_chars=100_000))
    assert book.chapters == []


def test_parse_epub_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_epub(tmp_path / "nope.epub")


def test_parse_epub_not_an_epub(tmp_path: Path) -> None:
    bad = tmp_path / "bad.epub"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("random.txt", "hello")
    with pytest.raises(ValueError):
        parse_epub(bad)


def test_parse_epub_falls_back_when_no_toc(tmp_path: Path) -> None:
    """没有 nav/ncx 时退化为「一个文档 = 一章」，而不是报错。"""
    import zipfile as zf

    path = tmp_path / "noToc.epub"
    container = (
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>'
        "</rootfiles></container>"
    )
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        "<metadata/><manifest>"
        '<item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="b" href="b.xhtml" media-type="application/xhtml+xml"/>'
        "</manifest><spine><itemref idref=\"a\"/><itemref idref=\"b\"/></spine></package>"
    )
    page = (
        '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head>'
        "<title>{title}</title></head><body><p>{body}</p></body></html>"
    )
    with zf.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", opf)
        archive.writestr("a.xhtml", page.format(title="Part A", body="A" * 400))
        archive.writestr("b.xhtml", page.format(title="Part B", body="B" * 400))

    book = parse_epub(path, ParseConfig(min_chapter_chars=10))
    assert len(book.chapters) == 2
    assert {c.title for c in book.chapters} == {"Part A", "Part B"}


def test_parse_epub_book_metadata_roundtrip(sample_epub: Path) -> None:
    book = parse_epub(sample_epub, ParseConfig(min_chapter_chars=1))
    restored = Book.from_dict(book.to_dict())
    assert restored.title == book.title
    assert len(restored.chapters) == len(book.chapters)
    assert restored.chapters[0].title == book.chapters[0].title


# --------------------------------------------------------------------------- #
# TXT
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Loss 1: The Purest of Intentions", True),
        ("Chapter 3", True),
        ("Intermission: Appeal to Emotion", True),
        ("Epilogue: Onward to Adulthood", True),
        ("Afterword", True),
        ("第三章 风起", True),
        ("序章", True),
        # 正文不能被误判
        ("Chapter one was over.", False),
        ("The loss was enormous and it hurt a great deal.", False),
        ("", False),
        ("A" * 200, False),
    ],
)
def test_is_heading_line(line: str, expected: bool) -> None:
    assert is_heading_line(line) is expected


def test_parse_txt_splits_and_joins_titles(sample_txt: Path) -> None:
    book = parse_txt(sample_txt, ParseConfig(min_chapter_chars=1))
    titles = [c.title for c in book.chapters]
    # "Loss 1:" + "The First Loss" 应该合成一个完整标题
    assert "Loss 1: The First Loss" in titles
    assert "Intermission: A Short Break" in titles


def test_parse_txt_drops_table_of_contents(sample_txt: Path) -> None:
    book = parse_txt(sample_txt, ParseConfig(min_chapter_chars=1))
    assert all("Table of Contents" not in c.title for c in book.chapters)


def test_parse_txt_custom_heading_regex(tmp_path: Path) -> None:
    path = tmp_path / "custom.txt"
    path.write_text(
        "== Alpha ==\nSome text here that is long enough.\n\n== Beta ==\nMore text here.\n",
        encoding="utf-8",
    )
    book = parse_txt(path, ParseConfig(min_chapter_chars=1), heading_re=r"^==.*==$")
    assert len(book.chapters) == 2


def test_parse_txt_no_headings_becomes_single_chapter(tmp_path: Path) -> None:
    path = tmp_path / "flat.txt"
    path.write_text("Just one long paragraph.\n\nAnd another one.", encoding="utf-8")
    book = parse_txt(path, ParseConfig(min_chapter_chars=1))
    assert len(book.chapters) == 1


def test_parse_book_dispatches_by_extension(sample_epub: Path, sample_txt: Path) -> None:
    assert parse_book(sample_epub, ParseConfig(min_chapter_chars=1)).chapters
    assert parse_book(sample_txt, ParseConfig(min_chapter_chars=1)).chapters


def test_parse_book_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "x.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_book(path)


def test_find_source_files_ignores_appledouble(tmp_path: Path) -> None:
    (tmp_path / "book.epub").write_bytes(b"x")
    (tmp_path / "._book.epub").write_bytes(b"x")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    (tmp_path / "cover.jpg").write_bytes(b"x")
    names = {p.name for p in find_source_files(tmp_path)}
    assert names == {"book.epub", "notes.md"}


def test_find_source_files_missing_dir(tmp_path: Path) -> None:
    assert find_source_files(tmp_path / "nope") == []


# --------------------------------------------------------------------------- #
# 标题清理（回归测试：曾把正文误删）
# --------------------------------------------------------------------------- #


def test_clean_heading_block_variants() -> None:
    from noveltts.epub_parser import _clean_heading_block

    # 形态 1：h1 已完整 + 正文里又有一份同名标题
    result = _clean_heading_block(
        "Loss 1: The Purest of Intentions",
        [
            Paragraph(text="Loss 1: The Purest of Intentions", kind="heading"),
            Paragraph(text="Loss 1:"),
            Paragraph(text="The Purest of Intentions"),
            Paragraph(text="THE DAYS WERE GETTING LONGER."),
        ],
    )
    assert [p.kind for p in result] == ["heading", "narration"]
    assert result[1].text.startswith("THE DAYS")

    # 形态 2：标题被 <br/> 拆成两行
    result = _clean_heading_block(
        "Intermission: Appeal to Emotion",
        [
            Paragraph(text="Intermission:"),
            Paragraph(text="Appeal to Emotion"),
            Paragraph(text="AFTER SCHOOL, IN THE COUNCIL ROOM."),
        ],
    )
    assert result[0].kind == "heading"
    assert result[1].text.startswith("AFTER SCHOOL")

    # 形态 3：标题与正文无关时**绝不**改动
    untouched = [Paragraph(text="Some unrelated body text that goes on and on.")]
    assert _clean_heading_block("Chapter 5", untouched) == untouched

    # 形态 4：短正文不会被误吞
    kept = [Paragraph(text="Loss 4:"), Paragraph(text="Enemy of Women Who?"), Paragraph(text="Body.")]
    result = _clean_heading_block("Loss 4: Enemy of Women Who?", kept)
    assert result[0].kind == "heading"
    assert result[-1].text == "Body."


# --------------------------------------------------------------------------- #
# 样板文字过滤（版权页 / ISBN / 许可声明）
# --------------------------------------------------------------------------- #


def test_drop_boilerplate_removes_copyright_lines() -> None:
    from noveltts.epub_parser import _drop_boilerplate

    paragraphs = [
        Paragraph(text="Chapter One", kind="heading"),
        Paragraph(text="The library was quiet that morning."),
        Paragraph(text="Copyright \u00a9 2024 by Someone. All rights reserved."),
        Paragraph(text="ISBN 978-4-06-123456-7"),
        Paragraph(
            text="No part of this book may be reproduced without the prior "
            "written permission of the publisher."
        ),
        Paragraph(text="She found the book at last."),
    ]
    kept = _drop_boilerplate("Chapter One", paragraphs, ParseConfig())
    texts = [p.text for p in kept]
    assert "The library was quiet that morning." in texts
    assert "She found the book at last." in texts
    assert len(kept) == 3, texts


def test_drop_boilerplate_keeps_heading_always() -> None:
    """标题即使命中规则也要保留，否则章节会失去可读的起点。"""
    from noveltts.epub_parser import _drop_boilerplate

    paragraphs = [
        Paragraph(text="Copyright Page", kind="heading"),
        Paragraph(text="The first real sentence of the story."),
    ]
    kept = _drop_boilerplate("Copyright Page", paragraphs, ParseConfig())
    assert kept[0].kind == "heading"


def test_drop_boilerplate_never_empties_a_unit() -> None:
    """全部命中时保留原文——宁可多读一句，也不能让一章消失。"""
    from noveltts.epub_parser import _drop_boilerplate

    paragraphs = [Paragraph(text="Copyright \u00a9 2024. All rights reserved.")]
    kept = _drop_boilerplate("Front", paragraphs, ParseConfig())
    assert len(kept) == 1


def test_drop_boilerplate_disabled_when_list_empty() -> None:
    from noveltts.epub_parser import _drop_boilerplate

    paragraphs = [Paragraph(text="Copyright \u00a9 2024. All rights reserved.")]
    config = ParseConfig(exclude_paragraph_patterns=[])
    assert _drop_boilerplate("X", paragraphs, config) == paragraphs


def test_drop_boilerplate_survives_bad_regex() -> None:
    from noveltts.epub_parser import _drop_boilerplate

    paragraphs = [Paragraph(text="Real content here.")]
    config = ParseConfig(exclude_paragraph_patterns=["([unclosed"])
    assert _drop_boilerplate("X", paragraphs, config) == paragraphs


def test_drop_boilerplate_keeps_ordinary_prose() -> None:
    """默认规则不能误伤正常小说行文。"""
    from noveltts.epub_parser import _drop_boilerplate

    ordinary = [
        Paragraph(text="All the rights she had were reserved for herself alone."),
        Paragraph(text="He printed the page and handed it over."),
        Paragraph(text="The permission slip was signed by her mother."),
        Paragraph(text="This book is a work of her own making, she liked to say."),
    ]
    kept = _drop_boilerplate("Chapter", ordinary, ParseConfig())
    assert len(kept) == len(ordinary), [p.text for p in kept]
