"""切分模块测试。

覆盖实际踩过的坑：

* 短句 ``"Hello there."`` 被误判成标题，导致整段不再合并；
* 引号内的句号把台词劈成两半；
* 缩写点号（``Mr.`` / ``5 p.m.``）被当成句末；
* ``paragraph_span`` 算错。
"""

from __future__ import annotations

import pytest

from noveltts.config import SegmentConfig
from noveltts.models import Chapter, Paragraph
from noveltts.segmenter import (
    TextSegmenter,
    _has_unclosed_quote,
    classify_kind,
    hard_wrap,
    split_long_sentence,
    split_sentences,
    summarize_segments,
)


# --------------------------------------------------------------------------- #
# 句级切分
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 基本切分
        ("One. Two. Three.", ["One.", "Two.", "Three."]),
        # 缩写点号不是句末
        ("Mr. Smith left. Dr. Jones stayed.", ["Mr. Smith left.", "Dr. Jones stayed."]),
        ("It cost $1.5 million. He agreed.", ["It cost $1.5 million.", "He agreed."]),
        ("He was 6 ft. tall. Really.", ["He was 6 ft. tall.", "Really."]),
        ("They met at 5 p.m. Then left.", ["They met at 5 p.m. Then left."]),
        ("The U.S. team won. A.M. practice was early.", ["The U.S. team won.", "A.M. practice was early."]),
        # 引号配对时保护内部标点
        ('He said, "Stop. Now." Then he left.', ['He said, "Stop. Now."', "Then he left."]),
        ('"I am here," she said. "And I am staying."',
         ['"I am here," she said.', '"And I am staying."']),
        ('"Hello." She left.', ['"Hello."', "She left."]),
        # 弯引号同样受保护（此处引号包住整句，句号在引号内）
        ("\u201cStop. Now.\u201d Then she left.", ["\u201cStop. Now.\u201d", "Then she left."]),
        # 引号不配对时放弃保护（不能吞掉整段）
        ('He said, "Stop. Now. Then he left.', ['He said, "Stop.', "Now.", "Then he left."]),
        # 省略号
        ("Wait... what?!  No way!!!", ["Wait...", "what?!", "No way!!!"]),
        ("She paused. \u2026Then continued.", ["She paused.", "\u2026Then continued."]),
        # 空输入
        ("", []),
        ("   ", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_split_sentences_returns_no_empty_strings() -> None:
    for sentence in split_sentences("A.  B.   C."):
        assert sentence.strip() == sentence
        assert sentence


def test_long_sentence_without_punctuation_stays_whole() -> None:
    text = "A single long sentence without any punctuation at all that keeps going"
    assert split_sentences(text) == [text]


# --------------------------------------------------------------------------- #
# 长句再切分
# --------------------------------------------------------------------------- #


def test_split_long_sentence_uses_clause_boundaries() -> None:
    sentence = (
        "Exiting the arcade to the north, across from Hirokoji Dori, led to a quieter "
        "street, not serene exactly, because the bones of old shops still lingered, "
        "though most had been repainted, and a few new spots were scattered between them."
    )
    pieces = split_long_sentence(sentence, 80)
    assert len(pieces) > 1
    assert all(len(p) <= 120 for p in pieces)
    assert " ".join(pieces).replace("  ", " ") == sentence


def test_split_long_sentence_noop_when_short() -> None:
    assert split_long_sentence("Short one.", 80) == ["Short one."]


def test_hard_wrap_respects_limit() -> None:
    text = " ".join(["word"] * 100)
    pieces = hard_wrap(text, 40)
    assert len(pieces) > 1
    assert all(len(p) <= 60 for p in pieces)


def test_hard_wrap_splits_unspaced_cjk() -> None:
    text = "一" * 300
    pieces = hard_wrap(text, 50)
    assert len(pieces) > 1
    assert "".join(pieces) == text


# --------------------------------------------------------------------------- #
# 类型判断
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"Huh?"', "dialogue"),          # 短的引号句必须是对话，不能是标题
        ('"I"', "dialogue"),
        ('He said, "bye."', "dialogue"),  # 以闭引号收尾也算对话
        ("Chapter One", "heading"),
        ("Foreword", "heading"),
        ("The quick brown fox jumped over the lazy dog and kept running.", "narration"),
        ("", "narration"),
    ],
)
def test_classify_kind(text: str, expected: str) -> None:
    assert classify_kind(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [('"I told you that', True), ('"I told you that"', False), ("plain", False),
     ("\u201copen", True), ("\u201cclosed\u201d", False)],
)
def test_has_unclosed_quote(text: str, expected: bool) -> None:
    assert _has_unclosed_quote(text) is expected


# --------------------------------------------------------------------------- #
# 章节切分
# --------------------------------------------------------------------------- #


def make_chapter(paragraphs: list[tuple[str, str]], chapter_id: str = "c001") -> Chapter:
    """用 (类型, 文本) 列表构造一个测试章节。"""
    return Chapter(
        index=1,
        title="Test",
        chapter_id=chapter_id,
        paragraphs=[Paragraph(text=text, kind=kind) for kind, text in paragraphs],  # type: ignore[arg-type]
    )


def test_short_sentence_does_not_collapse_paragraph() -> None:
    """回归测试：短句曾被判成标题，导致整段只切出一个片段。"""
    chapter = make_chapter([
        ("narration", "Hello there. Boom goes everything. Final line here."),
        ("narration", "Another paragraph entirely. With a second sentence."),
    ])
    segments = TextSegmenter(SegmentConfig(max_chars=40), language="en").segment_chapter(
        chapter, normalize=False
    )
    assert len(segments) > 2
    assert all(s.kind != "heading" for s in segments)


def test_heading_becomes_its_own_segment() -> None:
    chapter = make_chapter([
        ("heading", "Loss 1: The Purest of Intentions"),
        ("narration", "The days were getting longer again."),
    ])
    segments = TextSegmenter(SegmentConfig(), language="en").segment_chapter(chapter, normalize=False)
    assert segments[0].kind == "heading"
    assert segments[0].text == "Loss 1: The Purest of Intentions"
    assert all(s.kind != "heading" for s in segments[1:])


def test_dialogue_and_narration_are_separated() -> None:
    chapter = make_chapter([
        ("narration", 'She looked up slowly. "What do you want?" He shrugged.'),
    ])
    segments = TextSegmenter(SegmentConfig(max_chars=40), language="en").segment_chapter(
        chapter, normalize=False
    )
    kinds = [s.kind for s in segments]
    # 不应出现 narration 与 dialogue 被塞进同一个片段
    for segment in segments:
        assert segment.kind in {"narration", "dialogue", "heading"}


def test_segment_ids_and_indices_are_sequential() -> None:
    chapter = make_chapter([
        ("narration", "First sentence here. Second sentence here. Third one too."),
    ])
    segments = TextSegmenter(SegmentConfig(max_chars=30), language="en").segment_chapter(
        chapter, normalize=False
    )
    assert [s.index for s in segments] == list(range(1, len(segments) + 1))
    assert [s.segment_id for s in segments] == [
        f"c001-{i:04d}" for i in range(1, len(segments) + 1)
    ]


def test_respects_max_chars() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    chapter = make_chapter([("narration", text)])
    config = SegmentConfig(max_chars=60, hard_max_chars=100)
    segments = TextSegmenter(config, language="en").segment_chapter(chapter, normalize=False)
    assert all(s.char_count <= config.hard_max_chars for s in segments)


def test_cross_paragraph_merge_only_for_unclosed_quote() -> None:
    """一句话被拆到两段时合并；两个各自闭合的引号段**不**合并。"""
    merged = make_chapter([
        ("narration", '"I suppose we do submit club budget proposals, but those are approved'),
        ("narration", 'by teachers anyway," she said.'),
    ])
    segments = TextSegmenter(SegmentConfig(), language="en").segment_chapter(merged, normalize=False)
    assert any(s.paragraph_span == 2 for s in segments)

    separate = make_chapter([
        ("dialogue", '"Hey, anything you want, I\'ll\u2014"'),
        ("dialogue", '"D-don\'t you dare," Komari snapped.'),
    ])
    segments = TextSegmenter(SegmentConfig(), language="en").segment_chapter(
        separate, normalize=False
    )
    assert all(s.paragraph_span == 1 for s in segments)


def test_split_on_paragraph_disables_merging() -> None:
    chapter = make_chapter([
        ("narration", '"I suppose we submit proposals'),
        ("narration", 'anyway," she said.'),
    ])
    config = SegmentConfig(split_on_paragraph=True)
    segments = TextSegmenter(config, language="en").segment_chapter(chapter, normalize=False)
    assert all(s.paragraph_span == 1 for s in segments)


def test_normalization_applied_and_source_kept() -> None:
    chapter = make_chapter([("narration", "Vol. 8 is out.")])
    segments = TextSegmenter(SegmentConfig(), language="en").segment_chapter(chapter, normalize=True)
    assert segments[0].text == "Volume 8 is out."
    assert segments[0].source_text == "Vol. 8 is out."


def test_normalization_can_be_skipped() -> None:
    chapter = make_chapter([("narration", "Vol. 8 is out.")])
    segments = TextSegmenter(SegmentConfig(), language="en").segment_chapter(chapter, normalize=False)
    assert segments[0].text == "Vol. 8 is out."


def test_empty_paragraphs_are_skipped() -> None:
    chapter = make_chapter([("narration", "   "), ("narration", "Real text here."), ("narration", "")])
    segments = TextSegmenter(SegmentConfig(), language="en").segment_chapter(chapter, normalize=False)
    assert len(segments) == 1
    assert segments[0].text == "Real text here."


def test_segment_text_helper() -> None:
    segmenter = TextSegmenter(SegmentConfig(), language="en")
    segments = segmenter.segment_text("First. Second.", chapter_id="adhoc", normalize=False)
    assert segments[0].chapter_id == "adhoc"
    assert all(s.segment_id.startswith("adhoc-") for s in segments)


def test_segment_text_with_title() -> None:
    segmenter = TextSegmenter(SegmentConfig(), language="en")
    segments = segmenter.segment_text("Body text.", chapter_id="t", title="My Title", normalize=False)
    assert segments[0].kind == "heading"
    assert segments[0].text == "My Title"


def test_chapter_segments_are_cached_on_chapter() -> None:
    chapter = make_chapter([("narration", "Some text goes here.")])
    segmenter = TextSegmenter(SegmentConfig(), language="en")
    segments = segmenter.segment_chapter(chapter, normalize=False)
    assert chapter.segments is segments


def test_paragraph_span_within_bounds() -> None:
    chapter = make_chapter([
        ("narration", "First paragraph with several sentences. Another one follows here."),
        ("narration", "Second paragraph is short."),
    ])
    segments = TextSegmenter(SegmentConfig(max_chars=40), language="en").segment_chapter(
        chapter, normalize=False
    )
    total = len(chapter.paragraphs)
    for segment in segments:
        assert 1 <= segment.paragraph_span
        assert segment.paragraph_index + segment.paragraph_span <= total


def test_overlong_sentence_is_split() -> None:
    long_clause = (
        "The quiet morning settled over the empty library like a blanket, "
        "and the shelves stood still, and the dust drifted slowly, and nothing "
        "at all happened for a very long while, which was exactly what she wanted."
    )
    chapter = make_chapter([("narration", long_clause)])
    config = SegmentConfig(max_chars=80, hard_max_chars=100)
    segments = TextSegmenter(config, language="en").segment_chapter(chapter, normalize=False)
    assert len(segments) > 1
    assert all(s.char_count <= config.hard_max_chars for s in segments)


def test_summarize_segments() -> None:
    chapter = make_chapter([("narration", "One sentence. Two sentences.")])
    segments = TextSegmenter(SegmentConfig(), language="en").segment_chapter(chapter, normalize=False)
    summary = summarize_segments(segments)
    assert "片段总数" in summary
    assert "字符数" in summary


def test_summarize_empty() -> None:
    assert summarize_segments([]) == "没有片段"
