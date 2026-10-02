"""洗稿模块测试。

重点覆盖实际踩过的坑：

* ``B29`` 里的 ``29`` 被误改（正则 ``|`` 把断言切开了）；
* ``No. Not everything.`` 被改成 ``Number.``；
* ``1990-1995`` 被当成普通数字各读一遍；
* 破折号一律变逗号，导致 ``busy—`` 结尾多一个逗号。
"""

from __future__ import annotations

import pytest

from noveltts.config import TextRulesConfig
from noveltts.normalize import (
    NormalizeReport,
    TextNormalizer,
    build_normalizer,
    chinese_to_int,
    decimal_to_words,
    detect_language,
    int_to_words,
    normalize_text,
    roman_to_int,
    to_ordinal_words,
)


@pytest.fixture
def normalizer() -> TextNormalizer:
    return build_normalizer(language="en")


# --------------------------------------------------------------------------- #
# 数词转换
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "zero"),
        (7, "seven"),
        (15, "fifteen"),
        (42, "forty-two"),
        (100, "one hundred"),
        (105, "one hundred five"),
        (1000, "one thousand"),
        (1500, "one thousand five hundred"),
        (2025, "two thousand twenty-five"),
        (1_200_000, "one million two hundred thousand"),
        (3_500_000_000, "three billion five hundred million"),
        (-5, "minus five"),
    ],
)
def test_int_to_words(value: int, expected: str) -> None:
    assert int_to_words(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, "first"), (2, "second"), (3, "third"), (11, "eleventh"),
     (21, "twenty-first"), (22, "twenty-second"), (100, "one hundredth")],
)
def test_to_ordinal_words(value: int, expected: str) -> None:
    assert to_ordinal_words(value) == expected


@pytest.mark.parametrize(
    ("value", "expected", "language"),
    [("1.5", "one point five", "en"), ("1,500.25", "one thousand five hundred point two five", "en"),
     ("1.5", "1点五", "zh")],
)
def test_decimal_to_words(value: str, expected: str, language: str) -> None:
    assert decimal_to_words(value, language=language) == expected


@pytest.mark.parametrize(
    ("roman", "expected"),
    [("IV", 4), ("VII", 7), ("X", 10), ("MCMLXXXIV", 1984), ("IIII", None), ("ABC", None)],
)
def test_roman_to_int(roman: str, expected: int | None) -> None:
    assert roman_to_int(roman) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("三十", 30), ("一百零五", 105), ("两千", 2000), ("一万", 10000), ("10", 10), ("abc", None)],
)
def test_chinese_to_int(text: str, expected: int | None) -> None:
    assert chinese_to_int(text) == expected


# --------------------------------------------------------------------------- #
# 语言探测
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The quick brown fox jumps over the lazy dog.", "en"),
        ("今天天气很好，我们去公园散步吧。", "zh"),
        ("これは日本語のテキストです。", "ja"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


# --------------------------------------------------------------------------- #
# 英文规则（逐条覆盖实际坑点）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # --- 缩写 ---
        ("Mr. Tanaka and Dr. Smith met Prof. Yamada.", "Mister Tanaka and Doctor Smith met Professor Yamada."),
        ("Vol. 8 came out in 2025.", "Volume 8 came out in two thousand twenty-five."),
        ("It was 1.5 million yen, approx.", "It was one point five million yen, approximately."),
        ("etc. and so on.", "et cetera and so on."),
        ("See No. 5 on page 12.", "See Number 5 on page twelve."),
        # No. 作为否定词时不能展开（曾经把台词改成 "Number. Not everything."）
        ('"No. Not everything, at least."', '"No. Not everything, at least."'),
        ('"Wh-what? No. Her preschool is doing a party."', '"Wh-what? No. Her preschool is doing a party."'),
        # --- 数字区间 ---
        ("And that was how we lost 870-120.", "And that was how we lost eight hundred seventy to one hundred twenty."),
        ("1990-1995 was a great era.", "1990 to 1995 was a great era."),
        # --- 专有名词保护 ---
        ("Class 2-C's round began, vs. class 2-E.", "Class 2-C's round began, versus class 2-E."),
        ("H2O and B29 are not numbers 2 or 29.", "H2O and B29 are not numbers 2 or twenty-nine."),
        ("National Route 1 highway.", "National Route 1 highway."),
        ("A well-known 3-second rule and a 1500-meter race.", "A well-known 3-second rule and a 1500-meter race."),
        ("The 400-meter final at 3:30 p.m.", "The 400-meter final at 3:30 p.m."),
        # --- 金额 ---
        ("It cost $1,500.00 exactly.", "It cost one thousand five hundred dollars exactly."),
        ("Then $2.50 more.", "Then two point five zero dollars more."),
        # --- 序数 / 罗马数字 ---
        ("1st place in the 2025 awards.", "first place in the two thousand twenty-five awards."),
        ("Chapter VII of Vol. IV.", "Chapter seven of Volume four."),
        ("The score was 3-2 in the 87th minute.", "The score was three to two in the eighty-seventh minute."),
        # --- 量词数字 ---
        ("He had 3 apples and 20 percent of the pie.", "He had 3 apples and twenty percent of the pie."),
        ("He ran 10 kilometers in 45 minutes.", "He ran ten kilometers in forty-five minutes."),
        # --- 标点 ---
        ("Wait... what?!", "Wait… what?!"),
        ("She said \u201cHello\u201d and left.", 'She said "Hello" and left.'),
        # --- 破折号四种情形 ---
        ("I mean\u2014argh!", "I mean, argh!"),
        ("busy\u2014She left.", "busy. She left."),
        ('for me to\u2014"', 'for me to"'),
        # 破折号后有空格再接大写 -> 是「顿一下继续同一句」，保留破折号；
        # 注意弯引号 \u2019 会被 quotes 规则统一成直引号，这是刻意的
        ("not why I\u2019m\u2014 Well, yes.", "not why I'm\u2014 Well, yes."),
        ("north\u2014across from Dori\u2014led on.", "north, across from Dori, led on."),
    ],
)
def test_english_rules(normalizer: TextNormalizer, source: str, expected: str) -> None:
    assert normalizer.normalize(source).text == expected


def test_single_digit_left_alone(normalizer: TextNormalizer) -> None:
    """个位数单独出现时保留阿拉伯数字（模型读它没有困难）。"""
    assert normalizer.normalize("I have 2 cats and 3 dogs.").text == "I have 2 cats and 3 dogs."


def test_disabled_rules_leave_text_alone() -> None:
    """关掉规则后文本必须逐字不变。"""
    rules = TextRulesConfig(
        enabled=True,
        abbreviations={},
        numeric_context_abbreviations={},
        ordinal_numbers=False,
        year_range=False,
        number_followed_by_noun=False,
        number_ranges=False,
        roman_numerals=False,
        dashes=False,
        ellipsis=False,
        quotes="keep",
        unicode_normalize="none",
        collapse_whitespace=False,
        strip_markup=False,
        custom_replacements={},
    )
    normalizer = TextNormalizer(rules, language="en")
    source = 'Vol. 8, 1st place \u2014 "quoted"... 870-120.'
    assert normalizer.normalize(source).text == source


def test_master_switch_disables_everything() -> None:
    normalizer = TextNormalizer(TextRulesConfig(enabled=False), language="en")
    source = 'Vol. 8 \u2014 "quoted"... 1st'
    assert normalizer.normalize(source).text == source


def test_custom_replacements_run_last() -> None:
    """自定义规则优先级最高。"""
    rules = TextRulesConfig(custom_replacements={r"\bNukumizu\b": "Nukumizu-san"})
    normalizer = build_normalizer(rules, language="en")
    assert "Nukumizu-san" in normalizer.normalize("Nukumizu walked in.").text


def test_url_and_email_are_protected(normalizer: TextNormalizer) -> None:
    """URL / 邮箱里的数字绝不能被改写。"""
    source = "See https://example.com/v1/page-1500 or mail a1@b2.com now."
    assert normalizer.normalize(source).text == source


def test_report_counts_and_samples(normalizer: TextNormalizer) -> None:
    report = normalizer.normalize("Vol. 8 and Vol. 9 came out.").report
    assert report.rules["abbreviation"].count == 2
    assert report.changed is True


def test_report_merges_across_calls(normalizer: TextNormalizer) -> None:
    report = NormalizeReport()
    normalizer.normalize("Vol. 8.", report=report)
    normalizer.normalize("Vol. 9.", report=report)
    assert report.rules["abbreviation"].count == 2
    assert "共" in report.summary()


def test_normalize_many_returns_report(normalizer: TextNormalizer) -> None:
    results, report = normalizer.normalize_many(["Vol. 8.", "1st place."])
    assert len(results) == 2
    assert report.total >= 2


def test_normalize_text_helper() -> None:
    assert normalize_text("Vol. 8.", language="en") == "Volume 8."


def test_empty_and_whitespace_input(normalizer: TextNormalizer) -> None:
    assert normalizer.normalize("").text == ""
    assert normalizer.normalize("   ").text == "   "


def test_bad_protect_pattern_is_skipped() -> None:
    """非法正则不应该炸掉整条流水线。"""
    rules = TextRulesConfig(protect_patterns=["([unclosed"])
    normalizer = TextNormalizer(rules, language="en")
    assert "Vol. 8" in normalizer.normalize("Vol. 8 here.").text


def test_bad_custom_replacement_is_skipped() -> None:
    rules = TextRulesConfig(custom_replacements={"(bad": "x"})
    normalizer = TextNormalizer(rules, language="en")
    assert normalizer.normalize("hello").text == "hello"


# --------------------------------------------------------------------------- #
# 中文规则
# --------------------------------------------------------------------------- #


def test_chinese_measure_words() -> None:
    normalizer = build_normalizer(language="zh")
    assert "30岁" in normalizer.normalize("他今年三十岁。").text


def test_chinese_phone_digits_spaced() -> None:
    normalizer = build_normalizer(language="zh")
    assert "1 3 8 0 0 1 3 8 0 0 0" in normalizer.normalize("电话是13800138000。").text


def test_auto_language_detection_picks_rules() -> None:
    normalizer = build_normalizer(TextRulesConfig(language="auto"))
    assert "30岁" in normalizer.normalize("他今年三十岁。").text
    assert normalizer.normalize("Vol. 8.").text == "Volume 8."
