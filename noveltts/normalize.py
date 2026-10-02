"""洗稿：把 TTS 模型念不对的缩写、数字、符号改写成口语化文字。

这是整条流水线里最“讲究”的一环：改对了听感自然，改错了会破坏原文。
因此本模块的设计原则是**保守 + 可审计 + 可回滚**：

* **保守**：只在高度确定的场景下改写（``Vol. 8``、``20 percent``、``1st``），
  拿不准的一律不动。比如 ``2-C``（班级名）保持原样——模型读 "two C" 是自然的，
  而改成 "two dash C" 反而糟糕。
* **可审计**：每次改写都通过 :class:`Replacer` 记录 规则名 / 次数 / 样例，
  跑完 ``noveltts normalize`` 会打印一张统计表，你可以据此调 config.yaml。
* **可回滚**：每个 :class:`~noveltts.models.Segment` 都保留 ``source_text``，
  方便对照“模型念错的那一句，原文到底是什么”。

实现上刻意**不用** ``inflect`` / ``num2words`` 这类第三方库，原因有二：
一是你的 conda 环境里没装，二是云端装包经常卡在网络上。所以这里自带了
完整的小型英文数词转换器（支持到 10^15），以及中文数字/月份/量词规则。

流水线顺序（顺序很重要，见 :func:`normalize_text`）：

1. 保护不需要改写的片段（URL、邮箱、ISBN、纯 ASCII 缩写白名单）；
2. 缩写展开 ``Vol.`` -> ``Volume``；
3. 序数词 ``1st`` -> ``first``（必须在基数词之前）；
4. 年份区间 ``1990-1995`` -> ``1990 to 1995``；
5. 数字 + 量名词 ``3 apples`` -> ``three apples``；
6. 大数字 ``1500`` -> ``fifteen hundred`` / ``1,500,000`` -> ``1.5 million``；
7. 罗马数字 ``Vol. IV`` -> ``Volume four``；
8. 破折号 ``A—B`` -> 逗号停顿；
9. 省略号 / 引号 / 空白 / Unicode 归一化；
10. 中文规则（``三十`` -> ``30``、电话式数字逐位念）；
11. 用户自定义规则（优先级最高，最后执行）。

每一步的具体规则都可以在 ``config.yaml`` 的 ``text_rules`` 段里单独关掉。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Iterable

from .config import TextRulesConfig
from .utils import get_logger

logger = get_logger("normalize")


# --------------------------------------------------------------------------- #
# 保护占位符
# --------------------------------------------------------------------------- #

#: 占位符前后缀：用私用区字符，正常文本里绝不会出现，因此不会撞车。
_PROTECT_OPEN = "\ue000"
_PROTECT_CLOSE = "\ue001"

_PROTECT_TOKEN_RE = re.compile(f"{_PROTECT_OPEN}(\\d+){_PROTECT_CLOSE}")

#: 默认需要原样保护、完全不参与改写的片段。
_DEFAULT_PROTECT_PATTERNS: tuple[str, ...] = (
    r"<[^>]{1,80}>",                        # 残留的富文本标记
    r"https?://\S+|www\.\S+",               # URL
    r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",         # 邮箱
    r"\b\d{1,5}(?:-\d{1,7}){2,3}[\dxX]\b",  # ISBN-10 / ISBN-13
    r"\d{1,2}:\d{2}(?::\d{2})?",            # 时刻 3:30 / 12:05:09（拆分反而更糟）
    r"\d+[-‑]\s?[A-Za-z]{1,12}\b",          # 连字符复合词 400-meter / 3-second
    r"\d+\s?[ap]\.?m\.?(?![A-Za-z])",       # 5 p.m. / 7 am（保留让模型读 "p m"）
)


class Protector:
    """临时占位器：把不想被后续规则碰到的片段藏起来，处理完再还原。

    用私用区字符 + 序号做占位符（``\\ue0000\\ue001``），因为这类字符在
    小说正文里几乎不可能出现，且正则里能被安全匹配。

    本模块用它解决两个关键问题：

    1. **保护原文**：URL / 邮箱 / ISBN 等片段不参与任何改写；
    2. **隔离生成物**：数字转成单词后（``1500`` -> ``fifteen hundred``），
       生成物里不能再被后续数字规则二次改写。因此所有产生文本的规则都
       通过 :meth:`emit` 把结果先存起来、只把占位符放回文本。

    Example:
        >>> p = Protector()
        >>> p.protect("see https://x.dev now", [r"https?://\\S+"])
        'see \\ue0000\\ue001 now'
        >>> p.restore("see \\ue0000\\ue001 now")
        'see https://x.dev now'
    """

    def __init__(self) -> None:
        self._store: list[str] = []

    def protect(self, text: str, patterns: Iterable[str]) -> str:
        """把所有命中 ``patterns`` 的片段替换成占位符。"""
        for pattern in patterns:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                logger.warning("保护规则正则非法，已跳过：%s（%s）", pattern, exc)
                continue
            text = compiled.sub(lambda m: self.store(m.group(0)), text)
        return text

    def store(self, value: str) -> str:
        """存入一段文本并返回它的占位符。"""
        self._store.append(value)
        return f"{_PROTECT_OPEN}{len(self._store) - 1}{_PROTECT_CLOSE}"

    def mark(self, value: str) -> str:
        """存入一段**已经确定不需要改写**的文本，并标记为「不被后续规则处理」。

        与 :meth:`emit` 的区别：``mark`` 不计入统计报告（因为它不是一次
        改写，只是原样保留）。典型用途是数字区间 ``870-120``——既不能被
        拆开改写，也不该在统计里出现。

        实现上仍然存成占位符，这样替换前后**不改变文本长度量级**，
        后续正则的匹配位置不会错位。
        """
        if not value:
            return value
        return self.store(value)

    def emit(self, value: str, report: "NormalizeReport", rule: str, before: str) -> str:
        """记录一次改写，返回该改写结果的占位符。

        这是所有“会生成新文本”的规则必须使用的出口：返回占位符而不是明文，
        可以让生成物免受后续规则影响。
        """
        report.hit(rule, before, value)
        return self.store(value)

    def restore(self, text: str) -> str:
        """把占位符换回原文。"""
        if not self._store:
            return text
        # 生成物本身可能包含占位符（嵌套），循环还原直到稳定
        for _ in range(4):
            replaced = _PROTECT_TOKEN_RE.sub(lambda m: self._store[int(m.group(1))], text)
            if replaced == text:
                break
            text = replaced
        return text


# --------------------------------------------------------------------------- #
# 改写统计
# --------------------------------------------------------------------------- #


@dataclass
class RuleStat:
    """单条规则的命中统计。"""

    name: str
    count: int = 0
    samples: list[tuple[str, str]] = field(default_factory=list)
    """``(原文, 改写后)`` 样例，最多 :data:`_MAX_SAMPLES` 条。"""

    def add(self, before: str, after: str) -> None:
        self.count += 1
        if len(self.samples) < _MAX_SAMPLES:
            self.samples.append((before, after))


_MAX_SAMPLES = 5


@dataclass
class NormalizeReport:
    """一次洗稿的完整统计报告。

    Attributes:
        rules: 规则名 -> :class:`RuleStat`。
        changed: 文本是否发生变化。
    """

    rules: dict[str, RuleStat] = field(default_factory=dict)
    changed: bool = False

    def hit(self, name: str, before: str, after: str) -> str:
        """记录一次命中并返回 ``after``（便于在 ``re.sub`` 里内联使用）。"""
        self.rules.setdefault(name, RuleStat(name)).add(before, after)
        return after

    @property
    def total(self) -> int:
        return sum(stat.count for stat in self.rules.values())

    def merge(self, other: "NormalizeReport") -> None:
        """合并另一份报告（批量处理多段时用）。"""
        self.changed = self.changed or other.changed
        for name, stat in other.rules.items():
            target = self.rules.setdefault(name, RuleStat(name))
            target.count += stat.count
            for sample in stat.samples:
                if len(target.samples) < _MAX_SAMPLES:
                    target.samples.append(sample)

    def summary(self) -> str:
        """人类可读的多行摘要。"""
        if not self.rules:
            return "洗稿统计：没有命中任何规则（原文未改动）"
        lines = [f"洗稿统计：共 {self.total} 处改写"]
        for name, stat in sorted(self.rules.items(), key=lambda kv: -kv[1].count):
            lines.append(f"  {name:<26s} {stat.count:>6d} 处")
            for before, after in stat.samples[:2]:
                lines.append(f"      {before!r} -> {after!r}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 英文数词
# --------------------------------------------------------------------------- #

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)
_SCALES = ("", "thousand", "million", "billion", "trillion", "quadrillion")

_ORDINAL_WORDS = {
    "one": "first", "two": "second", "three": "third", "four": "fourth",
    "five": "fifth", "six": "sixth", "seven": "seventh", "eight": "eighth",
    "nine": "ninth", "ten": "tenth", "eleven": "eleventh", "twelve": "twelfth",
    "thirteen": "thirteenth", "fourteen": "fourteenth", "fifteen": "fifteenth",
    "sixteen": "sixteenth", "seventeen": "seventeenth", "eighteen": "eighteenth",
    "nineteen": "nineteenth", "twenty": "twentieth", "thirty": "thirtieth",
    "forty": "fortieth", "fifty": "fiftieth", "sixty": "sixtieth",
    "seventy": "seventieth", "eighty": "eightieth", "ninety": "ninetieth",
    "hundred": "hundredth", "thousand": "thousandth", "million": "millionth",
    "billion": "billionth", "trillion": "trillionth",
}


def int_to_words(number: int) -> str:
    """把整数转成英文单词（支持到 10^15 量级，足够小说使用）。

    Examples:
        >>> int_to_words(0)
        'zero'
        >>> int_to_words(42)
        'forty-two'
        >>> int_to_words(1500)
        'one thousand five hundred'
        >>> int_to_words(1200000)
        'one million two hundred thousand'
    """
    if number < 0:
        return f"minus {int_to_words(-number)}"
    if number < 20:
        return _ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        head = f"{_ONES[hundreds]} hundred"
        return head if remainder == 0 else f"{head} {int_to_words(remainder)}"

    # 按 3 位一组从高位往低处理
    for scale_index in range(len(_SCALES) - 1, 1, -1):
        scale_value = 1000 ** scale_index
        if number >= scale_value:
            head, remainder = divmod(number, scale_value)
            head_words = f"{int_to_words(head)} {_SCALES[scale_index]}"
            if remainder == 0:
                return head_words
            return f"{head_words} {int_to_words(remainder)}"

    thousands, remainder = divmod(number, 1000)
    head = f"{int_to_words(thousands)} thousand"
    return head if remainder == 0 else f"{head} {int_to_words(remainder)}"


def to_ordinal_words(number: int) -> str:
    """把整数转成英文序数词。

    Examples:
        >>> to_ordinal_words(1)
        'first'
        >>> to_ordinal_words(21)
        'twenty-first'
        >>> to_ordinal_words(100)
        'one hundredth'
    """
    words = int_to_words(number)
    parts = words.split(" ")
    last = parts[-1]
    if "-" in last:  # forty-two -> forty-second
        head, tail = last.rsplit("-", 1)
        parts[-1] = f"{head}-{_ORDINAL_WORDS.get(tail, tail)}"
    else:
        parts[-1] = _ORDINAL_WORDS.get(last, last)
    return " ".join(parts)


def number_to_words(number: int, *, language: str = "en") -> str:
    """语言感知的整数转写。中文返回阿拉伯数字（交给模型念），英文返回单词。"""
    if language == "zh":
        # 中文直接把阿拉伯数字留在文本里更稳：模型对「1500」的读法比
        # 「一千五百」更自然，也避免了「两/二」的语感选择问题。
        return str(number)
    return int_to_words(number)


def decimal_to_words(text: str, *, language: str = "en") -> str:
    """把 ``1.5`` / ``1,500.25`` 这类小数读成文字。

    英文：``1.5`` -> ``one point five``；整数部分按千分位规则读。
    中文：``1.5`` -> ``一点五``（更符合口语）。
    """
    cleaned = text.replace(",", "")
    if language == "zh":
        integer_part, _, fraction = cleaned.partition(".")
        digits = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
                  "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}
        head = str(int(integer_part)) if integer_part.isdigit() else integer_part
        tail = "".join(digits.get(char, char) for char in fraction)
        return f"{head}点{tail}" if tail else head

    integer_part, _, fraction = cleaned.partition(".")
    head = int_to_words(int(integer_part)) if integer_part.isdigit() else integer_part
    if not fraction:
        return head
    tail = " ".join(_ONES[int(char)] if char.isdigit() else char for char in fraction)
    return f"{head} point {tail}"


def estimate_number_words(value: int) -> int:
    """估算一个数字读出来大概几个词（用于判断要不要缩写）。"""
    return len(int_to_words(value).split())


# --------------------------------------------------------------------------- #
# 正则片段
# --------------------------------------------------------------------------- #

_DIGIT = r"\d"

#: 数字模式。
#:
#: 每个分支末尾的 ``(?![A-Za-z0-9])`` 非常关键：它防止把 ``870-120`` 的
#: ``870`` 与 ``120`` 拼成一个数字 ``870120`` 读成 "eight hundred seventy
#: thousand one hundred twenty"。数字后面紧跟字母或数字时，说明它是某个
#: 标识符的一部分（``B29``、``2C``），应当整体跳过。
_NUMBER = (
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![A-Za-z0-9])"
    r"|\d+(?:\.\d+)?(?![A-Za-z0-9])"
)

#: 带边界约束的数字模式。**必须**用非捕获组把 ``_NUMBER`` 的可选支包起来，
#: 否则 ``|`` 会把整个模式切成两支，导致后行/先行断言只约束到其中一支
#: （典型症状：``B29`` 里的 ``29`` 被单独抓出来改写成 ``Btwenty-nine``）。
_BOUNDED_NUMBER = r"(?<![A-Za-z0-9])(?:" + _NUMBER + r")"

#: ``3 apples`` 里能安全跟在数字后面的量名词（只列常见且无歧义的）。
_NOUN_AFTER_NUMBER = (
    "percent|per cent|percentage|"
    "meters?|metres?|kilometers?|kilometres?|centimeters?|centimetres?|"
    "miles?|yards?|feet|foot|inches?|"
    "seconds?|minutes?|hours?|days?|weeks?|months?|years?|decades?|centuries?|"
    "points?|goals?|runs?|wins?|losses?|laps?|rounds?|sets?|matches?|games?|"
    "people|person|students?|teachers?|members?|players?|characters?|"
    "yen|dollars?|cents?|euros?|pounds?|"
    "times|occasions?|copies|pages?|chapters?|volumes?|episodes?|seasons?|"
    "degrees?|liters?|litres?|grams?|kilograms?|kilos?|tons?|"
    "floors?|stories|storeys|steps?|seats?|rooms?|beds?|cars?|bikes?"
)

#: 这些缩写展开后**不能**补句号：它们几乎只作头衔，不承担句末标点的作用。
_NON_TERMINAL_ABBREVIATIONS: frozenset[str] = frozenset(
    {"Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Jr", "Sr", "Vol", "vol", "Ch", "chap", "Ed", "ed"}
)

#: 序数后缀
_ORDINAL_SUFFIX = r"(?:st|nd|rd|th)"

#: 罗马数字里**永不转换**的写法：``I`` 是英文代词、``V`` 是常见字母，
#: 它们作为罗马数字 1/5 的概率远低于被误伤的概率。
_AMBIGUOUS_ROMAN: frozenset[str] = frozenset({"I", "V", "L", "C", "D", "M"})

#: 罗马数字（严格校验，避免把 "I" 这种代词误伤）
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(roman: str) -> int | None:
    """把罗马数字转成整数；非法输入返回 ``None``。

    Examples:
        >>> roman_to_int("IV")
        4
        >>> roman_to_int("MCMLXXXIV")
        1984
    """
    text = roman.upper()
    if not text or any(char not in _ROMAN_VALUES for char in text):
        return None
    total = 0
    previous = 0
    for char in reversed(text):
        value = _ROMAN_VALUES[char]
        total += -value if value < previous else value
        previous = max(previous, value)
    # 反向校验：转回去必须一致，否则是 "IIII" 这类非规范写法
    return total if int_to_roman(total) == text else None


def int_to_roman(number: int) -> str:
    """整数转罗马数字（``roman_to_int`` 的反向校验用）。"""
    if number <= 0 or number >= 4000:
        return ""
    table = (
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
        (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
        (5, "V"), (4, "IV"), (1, "I"),
    )
    result = []
    for value, symbol in table:
        while number >= value:
            result.append(symbol)
            number -= value
    return "".join(result)


_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def chinese_to_int(text: str) -> int | None:
    """中文数字 -> 整数（支持 ``三十`` / ``一百零五`` / ``两千`` / ``一万``）。

    Examples:
        >>> chinese_to_int("三十")
        30
        >>> chinese_to_int("一百零五")
        105
    """
    if not text:
        return None
    # 纯阿拉伯数字直接返回
    if text.isdigit():
        return int(text)

    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            section += (number or 1) * unit
            number = 0
        elif char == "万":
            section = (section + number) * 10_000
            total += section
            section = 0
            number = 0
        elif char == "亿":
            total = (total + section + number) * 100_000_000
            section = 0
            number = 0
        else:
            return None
    return total + section + number


# --------------------------------------------------------------------------- #
# 主清洗器
# --------------------------------------------------------------------------- #


class TextNormalizer:
    """洗稿器。

    Args:
        rules: 规则配置（来自 ``config.yaml`` 的 ``text_rules`` 段）。
        language: 生效语言；``"en"`` / ``"zh"`` / ``"ja"`` / ``"auto"``。

    Example:
        >>> normalizer = TextNormalizer(language="en")
        >>> normalizer.normalize("Vol. 8 came out in 2025.").text
        'Volume eight came out in two thousand twenty-five.'
    """

    def __init__(self, rules: TextRulesConfig | None = None, *, language: str = "en") -> None:
        self.rules = rules or TextRulesConfig()
        self.language = language

    # ------------------------------------------------------------------ 入口
    def normalize(self, text: str, *, report: NormalizeReport | None = None) -> "NormalizedText":
        """清洗一段文本。

        Args:
            text: 原始文本。
            report: 传入已有的报告则累加统计；``None`` 时新建。

        Returns:
            :class:`NormalizedText`，含 ``text`` 与 ``report``。
        """
        report = report or NormalizeReport()
        original = text
        if not self.rules.enabled or not text.strip():
            return NormalizedText(text=text, report=report)

        language = self._resolve_language(text)

        # --- 1) 保护片段 ---
        protector = Protector()
        patterns = list(_DEFAULT_PROTECT_PATTERNS)
        patterns += list(self.rules.protect_patterns)
        text = protector.protect(text, patterns)

        # --- 2) Unicode 归一化 ---
        if self.rules.unicode_normalize in {"NFC", "NFKC"}:
            text = unicodedata.normalize(self.rules.unicode_normalize, text)

        # --- 3) 富文本残留 ---
        if self.rules.strip_markup:
            text = self._strip_markup(text, report)

        # --- 4) 缩写 ---
        text = self._expand_abbreviations(text, report)

        # --- 5) 数字相关（顺序敏感，且全部通过 protector.emit 隔离） ---
        #     金额 -> 序数 -> 年份区间 -> 数字区间 -> 量词数字 -> 裸数字 -> 罗马数字
        text = self._convert_money(text, report, language, protector)
        if self.rules.ordinal_numbers:
            text = self._convert_ordinals(text, report, language, protector)
        if self.rules.year_range:
            text = self._convert_year_ranges(text, report, protector)
        if self.rules.number_ranges:
            text = self._convert_number_ranges(text, report, language, protector)
        if self.rules.number_followed_by_noun:
            text = self._convert_measured_numbers(text, report, language, protector)
        text = self._convert_bare_numbers(text, report, language, protector)
        if self.rules.roman_numerals:
            text = self._convert_roman_numerals(text, report)

        # --- 6) 符号与标点 ---
        if self.rules.dashes:
            text = self._convert_dashes(text, report)
        if self.rules.ellipsis:
            text = self._convert_ellipsis(text, report)
        if self.rules.quotes != "keep":
            text = self._convert_quotes(text, report)
        text = self._strip_control_chars(text, report)

        # --- 7) 中文规则 ---
        if self.rules.cjk_numbers and language in {"zh", "ja"}:
            text = self._convert_cjk_numbers(text, report)

        # --- 8) 自定义规则（最高优先级） ---
        text = self._apply_custom(text, report)

        # --- 9) 空白折叠 ---
        if self.rules.collapse_whitespace:
            text = self._collapse_whitespace(text)

        # --- 10) 还原保护片段与生成物 ---
        text = protector.restore(text)

        report.changed = report.changed or text != original
        return NormalizedText(text=text, report=report)

    def normalize_many(self, texts: Iterable[str]) -> tuple[list[str], NormalizeReport]:
        """批量清洗；返回 ``(结果列表, 汇总报告)``。"""
        report = NormalizeReport()
        results = [self.normalize(text, report=report).text for text in texts]
        return results, report

    # ------------------------------------------------------------- 内部：语言
    def _resolve_language(self, text: str) -> str:
        """按配置与文本内容决定用哪套规则。"""
        configured = (self.rules.language or "auto").lower()
        if configured in {"en", "zh", "ja"}:
            return configured
        return detect_language(text)

    # --------------------------------------------------------- 内部：各条规则
    def _strip_markup(self, text: str, report: NormalizeReport) -> str:
        """去掉残留的 HTML/富文本标记。"""
        def repl(match: re.Match[str]) -> str:
            return report.hit("strip_markup", match.group(0), "")

        return re.sub(r"</?[A-Za-z][A-Za-z0-9]{0,9}(?:\s[^<>]{0,120})?/?>", repl, text)

    def _expand_abbreviations(self, text: str, report: NormalizeReport) -> str:
        """展开缩写表。

        分两类处理（**顺序很重要**，先处理受约束的那类）：

        1. ``numeric_context_abbreviations``：只有在后面紧跟数字时才展开。
           ``No. 5`` -> ``Number 5``；``"No. Not everything."`` 保持不动。
        2. ``abbreviations``：无条件展开（``Mr.`` / ``Vol.`` / ``etc.`` …）。

        规则按“长键优先”执行，避免短键抢走长键的匹配。两个细节：

        * 键里的空格允许匹配任意空白，因此 ``No.`` 与 ``No .`` 都能命中；
        * 如果缩写出现在句末（``... approximately.``），展开后要补回句号，
          否则句子会被两个空格粘在一起，模型的停顿会丢失。
        """
        # --- 1) 受数字上下文约束的缩写 ---
        for source, target in (self.rules.numeric_context_abbreviations or {}).items():
            escaped = r"\s*".join(re.escape(part) for part in source.split())
            pattern = rf"(?<![A-Za-z0-9]){escaped}(?=\s*\d)(?![A-Za-z0-9])"

            def repl_numeric(match: re.Match[str], _target: str = target) -> str:
                return report.hit("abbreviation_numeric", match.group(0), _target)

            text = re.sub(pattern, repl_numeric, text)

        # --- 2) 无条件展开 ---
        if not self.rules.abbreviations:
            return text

        for source in sorted(self.rules.abbreviations, key=len, reverse=True):
            target = self.rules.abbreviations[source]
            # 把配置里的键转成宽松正则：中间的空白匹配任意空白
            escaped = r"\s*".join(re.escape(part) for part in source.split())
            pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
            # 头衔类缩写（Mr. / Dr. / St. …）永远不结句，不能补句号
            is_title = source.rstrip(".") in _NON_TERMINAL_ABBREVIATIONS
            source_ends_sentence = source.rstrip().endswith(".") and not is_title

            def repl(
                match: re.Match[str],
                _target: str = target,
                _eos: bool = source_ends_sentence,
            ) -> str:
                raw = match.group(0)
                replacement = _target
                if _eos and not replacement.endswith("."):
                    # 原文里的点号可能同时充当句号：后面紧跟大写词或直接结束
                    # （"... approximately. I left" / "... approximately."）时补回句号；
                    # 后面是小写词（"... approx. 5 minutes"）时不补。
                    tail = text[match.end():match.end() + 3]
                    if not tail:
                        replacement += "."
                    elif tail[0].isspace():
                        following = tail.lstrip()
                        if following[:1].isupper():
                            replacement += "."
                    elif tail[0] in "\"'”’":
                        replacement += "."
                return report.hit("abbreviation", raw, replacement)

            # 大小写敏感：配置里写 "No." 才会命中 "No."，小写的 "no."（口语否定词）
            # 不会被误改成 "number."。想同时匹配大小写就在配置里写两条。
            text = re.sub(pattern, repl, text)
        return text

    def _convert_ordinals(
        self, text: str, report: NormalizeReport, language: str, protector: Protector
    ) -> str:
        """``1st`` / ``22nd`` / ``3rd`` -> 序数词。"""
        if language not in {"en"}:
            return text

        def repl(match: re.Match[str]) -> str:
            value = int(match.group(1))
            return protector.emit(to_ordinal_words(value), report, "ordinal", match.group(0))

        return re.sub(rf"\b(\d{{1,4}}){_ORDINAL_SUFFIX}\b", repl, text, flags=re.I)

    def _convert_year_ranges(
        self, text: str, report: NormalizeReport, protector: Protector
    ) -> str:
        """``1990-1995`` / ``1990–1995`` -> ``1990 to 1995``。

        只处理两侧都是 **合理年份**（1000~2199）的情况。像 ``870-120`` 这种
        比分不会被误判；而 ``1500-1600`` 会被当成年份区间，这也符合上下文。
        """
        pattern = r"\b(1\d{3}|20\d{2}|21\d{2})\s*[-–—]\s*(1\d{3}|20\d{2}|21\d{2})\b"

        def repl(match: re.Match[str]) -> str:
            written = f"{match.group(1)} to {match.group(2)}"
            return protector.emit(written, report, "year_range", match.group(0))

        return re.sub(pattern, repl, text)

    def _convert_number_ranges(
        self, text: str, report: NormalizeReport, language: str, protector: Protector
    ) -> str:
        """``870-120`` 这类“数字-数字”读成 ``A to B``。

        比分、赛程、区间在小说里很常见，模型直接读连字符容易含糊或漏读。
        但只在两侧都是 1~4 位纯数字、且不是年份区间、不是小数、不是负数时才转换，
        因此 ``2-C``、``3-E``、``ISBN``、``3-2`` 之外的标识符都不受影响。
        """
        if language != "en":
            return text

        # 数字的两个分支同样要带 `(?![A-Za-z0-9])`，否则 `1,500-2,000` 会先被
        # 逗号分支吃掉一半（`1,` 被当作数字），产出 "1,five hundred" 这种垃圾。
        plain = r"\d{1,3}(?![A-Za-z0-9])"
        grouped = r"\d{1,3}(?:,\d{3})+(?![A-Za-z0-9])"
        other = r"\d+(?![A-Za-z0-9])"
        pattern = (
            r"(?<![A-Za-z0-9.])(" + grouped + r"|" + other + r")\s*[-–]\s*"
            r"(" + grouped + r"|" + other + r")(?![A-Za-z0-9])"
        )

        def repl(match: re.Match[str]) -> str:
            left = int(match.group(1).replace(",", ""))
            right = int(match.group(2).replace(",", ""))
            written = f"{int_to_words(left)} to {int_to_words(right)}"
            return protector.emit(written, report, "number_range", match.group(0))

        return re.sub(pattern, repl, text)

    def _convert_measured_numbers(
        self, text: str, report: NormalizeReport, language: str, protector: Protector
    ) -> str:
        """``3 apples`` -> ``three apples``（仅限白名单量名词，防止误伤专有名词）。"""
        if language != "en":
            return text

        pattern = rf"(?<![\dA-Za-z])(\d{{1,3}}(?:,\d{{3}})*)\s+({_NOUN_AFTER_NUMBER})\b"

        def repl(match: re.Match[str]) -> str:
            value = int(match.group(1).replace(",", ""))
            written = f"{int_to_words(value)} {match.group(2)}"
            return protector.emit(written, report, "measured_number", match.group(0))

        return re.sub(pattern, repl, text, flags=re.I)

    def _convert_money(
        self, text: str, report: NormalizeReport, language: str, protector: Protector
    ) -> str:
        """货币金额：``$1,500`` -> ``one thousand five hundred dollars``。

        必须在数字规则之前执行，否则 ``$1,500.00`` 会被拆成
        ``$one thousand five hundred point zero zero`` 这种读起来很傻的结果。
        """
        if language != "en":
            return text

        symbols = {"$": "dollars", "£": "pounds", "€": "euros", "¥": "yen"}
        symbol_class = re.escape("".join(symbols))
        pattern = rf"([{symbol_class}])\s?(\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{1,2}})?)"

        def repl(match: re.Match[str]) -> str:
            symbol, amount = match.group(1), match.group(2)
            unit = symbols[symbol]
            value = float(amount.replace(",", ""))
            if abs(value - round(value)) < 1e-9:
                written = f"{int_to_words(int(round(value)))} {unit}"
            else:
                # 有小数分位时才读 point，避免 "1500.00" 读成 "point zero zero"
                written = f"{decimal_to_words(amount, language='en')} {unit}"
            return protector.emit(written, report, "money", match.group(0))

        return re.sub(pattern, repl, text)

    def _convert_bare_numbers(
        self, text: str, report: NormalizeReport, language: str, protector: Protector
    ) -> str:
        """处理剩下的裸数字。

        策略：
          * 小数 / 千分位 -> :func:`decimal_to_words`；
          * 大整数 -> 口语化大数（``1500`` -> ``fifteen hundred``，
            ``2000000`` -> ``two million``，``1200000`` -> ``1.2 million``）；
          * 其余整数 -> 英文单词 / 中文保留阿拉伯数字。

        Note:
            前后紧邻字母或数字的场合一律跳过（``(?<![A-Za-z0-9])`` 而不是
            ``\\b``，因为 ``\\b`` 会让 ``B29`` 里的 ``29`` 被单独抓出来）。
            因此 ``2-C``（班级名）、``H2O``、``B29``、``Route 1`` 这类专有名词
            都不会被拆开改写。

        Note:
            数字区间（``870-120``）会被整体跳过，交给
            :meth:`_convert_number_ranges` 处理。**这一点很重要**：
            如果把 ``870`` 与 ``120`` 分别改写再还原，两个结果会直接贴在一起
            变成 ``870120``，读出来就是 "eight hundred seventy-one hundred
            twenty"——一个谁都听不懂的怪数字。
        """
        range_guard = self._range_guard(text, language)

        # 这里刻意**不用** re.sub 的返回值直接拼结果：re.sub 会把「返回原文」
        # 的匹配也重新写进输出，于是被跳过的 ``120`` 会被写两遍
        # （"870-120" -> "870-120-120"）。改成手工拼接后，落在区间内部的部分
        # 由区间那一次的替换统一输出，不会重复。
        pieces: list[str] = []
        cursor = 0

        for match in re.finditer(_BOUNDED_NUMBER, text):
            start, end = match.start(), match.end()

            if start < cursor:
                continue  # 落在已处理过的区间内部

            if start in range_guard:
                # 这是一个「数字-数字」区间：整段原样保留（存成占位符，
                # 保证后续规则与长度计算都不受影响）
                range_end = range_guard[start]
                pieces.append(text[cursor:start])
                pieces.append(protector.mark(text[start:range_end]))
                cursor = range_end
                continue

            raw = match.group(0)
            replacement = self._rewrite_bare_number(raw, report, language, protector)
            if replacement == raw:
                continue  # 不改写就交给主循环的原文切片

            pieces.append(text[cursor:start])
            pieces.append(replacement)
            cursor = end

        pieces.append(text[cursor:])
        return "".join(pieces)

    def _rewrite_bare_number(
        self, raw: str, report: NormalizeReport, language: str, protector: Protector
    ) -> str:
        """改写一个独立的数字；不需要改写时原样返回 ``raw``。"""
        if not self._separators_ok(raw):
            return raw
        if "," in raw or "." in raw:
            written = decimal_to_words(raw, language=language)
            return protector.emit(written, report, "decimal_number", raw)

        value = int(raw)
        if language == "en" and value < 10 and len(raw) == 1:
            # 个位数单独出现（列表编号、楼层等）保留阿拉伯数字更自然：
            # 模型读 "3" 和读 "three" 一样准，改动只会增加噪音。
            return raw
        written = self._format_large_number(value, language)
        if written == raw:
            return raw
        return protector.emit(written, report, "number", raw)

    @staticmethod
    def _range_guard(text: str, language: str) -> dict[int, int]:
        """找出「数字-数字」区间的位置，供 :meth:`_convert_bare_numbers` 跳过。

        区间只在英文下才需要特殊处理（中文本来就保留阿拉伯数字）。这样做的
        收益是**幂等**：无论 :meth:`_convert_number_ranges` 开着还是关着，
        ``870-120`` 都不会被拆成两个数字各自改写后再拼起来。

        Returns:
            ``{位置下标: 该位置所在区间的结束下标（不含）}``。

        Note:
            为什么要把区间内**每一个**数字位置都作为键？
            因为 :data:`_BOUNDED_NUMBER` 的正则引擎是从左往右找第一个能匹配
            的位置，而 ``870-120`` 里真正能匹配成功的起点取决于上下文
            （``870`` 与 ``120`` 都会命中）。与其去猜正则从哪开始，不如让
            区间内任意一个起点都能查到同一个终点：

            * 命中 ``870`` 的起点 -> 跳过整个区间；
            * 命中 ``120`` 的起点 -> 同样跳过整个区间。

            曾经写成「只登记第一个数字位置」，结果 ``120`` 那一半仍然被单独
            改写，产出 ``870-120-one hundred twenty`` 这种废话。

        Note:
            只认**纯整数**区间。``1.5-2.5`` 这种带小数点的不是区间
            （更可能是范围值），两个小数各自按小数规则处理。

            后方断言**不能**排除句点，否则 ``870-120.``（区间出现在句末）
            会匹配不上——这是实际踩过的坑。
        """
        if language != "en":
            return {}

        number = r"\d+"
        pattern = rf"(?<![A-Za-z0-9.])(?:{number})[-–](?:{number})(?![A-Za-z0-9])"
        spans: dict[int, int] = {}
        for match in re.finditer(pattern, text):
            for position in range(match.start(), match.end()):
                spans[position] = match.end()
        return spans

    @staticmethod
    def _separators_ok(raw: str) -> bool:
        """校验千分位写法是否合法。

        用来兜住正则退化的输出：``1,`` / ``1,,000`` / ``1,00`` 这类都不是
        合法的数字写法，宁可原样保留也不要产出 "one point ..." 这种垃圾。
        """
        if raw.startswith(",") or raw.endswith(","):
            return False
        integer = raw.split(".", 1)[0]
        for position, part in enumerate(integer.split(",")):
            if position == 0:
                if not part.isdigit():
                    return False
            elif len(part) != 3 or not part.isdigit():
                return False
        return True

    def _format_large_number(self, value: int, language: str) -> str:
        """把整数按口语习惯写出来。

        * ``1500`` -> ``fifteen hundred``（比 ``one thousand five hundred`` 更口语）；
        * ``2000000`` -> ``two million``；
        * ``1200000`` -> ``1.2 million``（百万以上用「数字 + million」更自然）；
        * 中文 -> 保留阿拉伯数字（模型自己会念对）。
        """
        if language != "en":
            return str(value)

        threshold = max(1000, int(self.rules.large_numbers))

        if value >= 1_000_000:
            for scale_index in range(len(_SCALES) - 1, 1, -1):
                scale_value = 1000 ** scale_index
                if value >= scale_value:
                    quotient = value / scale_value
                    if abs(quotient - round(quotient)) < 1e-9:
                        head = int_to_words(int(round(quotient)))
                    else:
                        head = f"{quotient:.1f}".rstrip("0").rstrip(".")
                    return f"{head} {_SCALES[scale_index]}"

        if value >= threshold and value % 100 == 0 and value < 10_000:
            # 1500 -> fifteen hundred；注意 1200 读 twelve hundred 更常见
            return f"{int_to_words(value // 100)} hundred"

        return int_to_words(value)

    def _convert_roman_numerals(self, text: str, report: NormalizeReport) -> str:
        """``Vol. IV`` / ``Chapter VII`` 里的罗马数字转成英文数词。

        只处理明确带上下文的罗马数字（前面有 Volume/Chapter/Part/Act 等词）。
        两道保险：

        * **不收录 ``No.`` / ``Number`` 作上下文**——``No. I'm too old`` 里的
          ``I`` 是代词，不是数字 1；
        * **``I`` 与 ``V`` 永不转换**——它们是英文里最常见的代词和字母，
          作为罗马数字 1/5 出现的概率远低于被误伤的概率。

        Note:
            上下文用捕获组而不是后行断言，因为后行断言要求定宽，
            而 ``Vol\\.?`` 这种可选点号是变宽的。
        """
        context = (
            r"(Volume|Vol\.?|Chapter|Chap\.?|Part|Pt\.?|Act|Scene|Book|"
            r"Section|Sec\.?|Mark|Grade)"
        )
        pattern = rf"\b{context}\s+([IVXLCDM]{{1,7}})\b"

        def repl(match: re.Match[str]) -> str:
            roman = match.group(2)
            if roman in _AMBIGUOUS_ROMAN:
                return match.group(0)
            value = roman_to_int(roman)
            if value is None:
                return match.group(0)
            return report.hit(
                "roman_numeral", match.group(0), f"{match.group(1)} {int_to_words(value)}"
            )

        return re.sub(pattern, repl, text)

    def _convert_dashes(self, text: str, report: NormalizeReport) -> str:
        """处理破折号。

        这是小说里最容易改错的一类符号，所以按“**破折号后面接什么**”分类处理：

        ====================  ==================  ====================================
        原文                  结果                理由
        ====================  ==================  ====================================
        ``mean—argh``         ``mean, argh``     被打断，逗号给出自然停顿
        ``busy—She left``    ``busy. She left`` 这里是句末，句号才对
        ``busy—``（段末）      ``busy``           拖长音，不加标点最自然
        ``I—we``              ``I—we``           单词被切断，保留破折号
        ``——``（中文）          ``，``             中文长破折号
        ====================  ==================  ====================================

        数字之间的 ``-`` 一律不动（比分、区间交给 :meth:`_convert_number_ranges`），
        单词内部的 ``-`` 也不动（``well-known``）。
        """
        # --- 中文长破折号 -> 逗号 ---
        text = re.sub(
            r"——+",
            lambda m: report.hit("dash_cjk", m.group(0), "，"),
            text,
        )

        dash = "\u2014\u2013"  # em dash / en dash

        # --- 情况 1：段末/句末的拖长音 -> 直接删掉 ---
        #     "busy—" / "to—”" 这种是声音被拖长后中断，加任何标点都不如直接
        #     让模型在句尾自然收住。
        text = re.sub(
            rf"(?<=[A-Za-z])\s*[{dash}]+\s*(?=$|[\n\"'\u201d\u2019])",
            lambda m: report.hit("dash_trailing", m.group(0), ""),
            text,
        )

        # --- 情况 2：破折号**直接**接大写字母 -> 其实是句末，用句号 ---
        #     例："busy—She left"
        text = re.sub(
            rf"(?<=[A-Za-z])[{dash}]+(?=[A-Z])",
            lambda m: report.hit("dash_sentence", m.group(0), ". "),
            text,
        )

        # --- 情况 3：破折号后面有空格再接大写 -> 是「话说到一半又接上」，用逗号 ---
        #     例："That's not why I'm— Well, yes, I might have been."
        #     这里不是句末：说话人只是顿了一下继续同一句，用句号会把语气切断。
        text = re.sub(
            rf"(?<=[A-Za-z])\s+[{dash}]+\s+(?=[A-Z])",
            lambda m: report.hit("dash_resume", m.group(0), ", "),
            text,
        )

        # --- 情况 4：后面接小写字母 -> 打断或插入语，用逗号 ---
        text = re.sub(
            rf"(?<=[A-Za-z])\s*[{dash}]+\s*(?=[a-z])",
            lambda m: report.hit("dash", m.group(0), ", "),
            text,
        )

        return text

    def _convert_ellipsis(self, text: str, report: NormalizeReport) -> str:
        """``...`` / ``。。。`` -> 单个省略号（模型对单字符省略号的停顿更稳定）。"""
        text = re.sub(
            r"\.{3,}", lambda m: report.hit("ellipsis", m.group(0), "…"), text
        )
        text = re.sub(
            r"。{3,}", lambda m: report.hit("ellipsis_cjk", m.group(0), "……"), text
        )
        return text

    def _convert_quotes(self, text: str, report: NormalizeReport) -> str:
        """统一引号。

        * ``straight``：全部转成 ``"`` / ``'``。模型在英文下对直引号最稳；
        * ``cjk``：转成「」『』（中文小说推荐）。
        """
        mapping: dict[str, str]
        if self.rules.quotes == "cjk":
            mapping = {
                "\u201c": "\u300c", "\u201d": "\u300d",
                "\u2018": "\u300e", "\u2019": "\u300f",
                '"': "\u300c", "'": "\u300e",
            }
        else:
            mapping = {
                "\u201c": '"', "\u201d": '"',
                "\u2018": "'", "\u2019": "'",
                "\u300c": '"', "\u300d": '"',
                "\u300e": '"', "\u300f": '"',
            }

        def repl(match: re.Match[str]) -> str:
            char = match.group(0)
            target = mapping[char]
            if char == target:
                return char
            return report.hit("quote", char, target)

        return re.sub("[" + re.escape("".join(mapping)) + "]", repl, text)

    def _strip_control_chars(self, text: str, report: NormalizeReport) -> str:
        """删除控制字符与零宽字符（电子书里常见，会让模型读错或卡住）。"""
        def repl(match: re.Match[str]) -> str:
            return report.hit("control_char", match.group(0), "")

        return re.sub(r"[\u0000-\u0008\u000b-\u001f\u007f\u200b-\u200f\u2028\u2029\ufeff]", repl, text)

    def _convert_cjk_numbers(self, text: str, report: NormalizeReport) -> str:
        """中文数字规则。

        1. 中文数字 + 量词 -> 阿拉伯数字（``三十岁`` -> ``30岁``）；
        2. 电话/编号型长串 -> 逐位念（``13800138000`` -> ``1 3 8 0 …``）；
        3. ``第X`` 序数保留中文（``第三`` 模型念得很好，不必改）。
        """
        measure = "|".join(re.escape(key) for key in self.rules.cjk_measure_pairs)
        if measure:
            pattern = rf"([零〇一二两三四五六七八九十百千万亿]{{1,8}})(?=({measure}))"

            def repl_measure(match: re.Match[str]) -> str:
                value = chinese_to_int(match.group(1))
                if value is None:
                    return match.group(0)
                return report.hit("cjk_number", match.group(0), str(value))

            text = re.sub(pattern, repl_measure, text)

        # 电话式数字：连续 7 位以上，逐位用空格隔开更清晰
        def repl_phone(match: re.Match[str]) -> str:
            digits = match.group(0)
            return report.hit("cjk_phone", digits, " ".join(digits))

        text = re.sub(r"(?<!\d)\d{7,}(?!\d)", repl_phone, text)
        return text

    def _apply_custom(self, text: str, report: NormalizeReport) -> str:
        """执行用户在 config 里写的自定义规则（最后执行，优先级最高）。"""
        for pattern, replacement in self.rules.custom_replacements.items():
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                logger.warning("自定义规则正则非法，已跳过：%s（%s）", pattern, exc)
                continue

            def repl(match: re.Match[str], _replacement: str = replacement) -> str:
                expanded = match.expand(_replacement)
                return report.hit("custom", match.group(0), expanded)

            text = compiled.sub(repl, text)
        return text

    def _collapse_whitespace(self, text: str) -> str:
        """折叠空白：多空格变一个，去掉行首行尾空格。"""
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()


@dataclass
class NormalizedText:
    """清洗结果。"""

    text: str
    report: NormalizeReport

    def __str__(self) -> str:  # pragma: no cover - 便捷输出
        return self.text


# --------------------------------------------------------------------------- #
# 语言探测
# --------------------------------------------------------------------------- #

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str, *, default: str = "en") -> str:
    """极简语言探测：按字符占比判断 ``zh`` / ``ja`` / ``en``。

    只看“哪种字符更多”，不做统计模型——对小说这种单一语言长文本足够。
    """
    sample = text[:4000]
    kana = len(_KANA_RE.findall(sample))
    cjk = len(_CJK_CHAR_RE.findall(sample))
    latin = len(_LATIN_RE.findall(sample))

    if kana and kana * 5 > cjk:
        return "ja"
    if cjk > latin:
        return "zh"
    if latin:
        return "en"
    return default


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #

#: 默认缩写表：这些都是实际听测里模型容易念错或念得别扭的。
#:
#: 两条重要约定：
#:
#: 1. **大小写敏感**。``No.``（Number）与小写 ``no.``（口语否定词）含义完全不同，
#:    所以表里只放无歧义的大小写形式。想同时匹配就写两条。
#: 2. **宁缺毋滥**。多义缩写（``p.`` / ``min.`` / ``v.`` …）默认不收，
#:    因为它们在没有上下文时更容易改错。要用请在 ``config.yaml`` 里显式添加。
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    # 头衔（永远不会是句末标点的承担者）
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "St.": "Saint",
    "Jr.": "Junior",
    "Sr.": "Senior",
    # 常见拉丁缩写
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "vs.": "versus",
    "approx.": "approximately",
    # 书籍/出版/编号（语义明确的才收；``No.`` 走数字上下文规则）
    "Vol.": "Volume",
    "Ch.": "Chapter",
    # 度量（全小写且无歧义的）
    "km": "kilometers",
    "cm": "centimeters",
    "mm": "millimeters",
    "kg": "kilograms",
    "lbs": "pounds",
    "hrs": "hours",
    "hr": "hour",
    "secs": "seconds",
    # 符号
    "&": "and",
    "+": "plus",
    "=": "equals",
    "%": "percent",
    "@": "at",
}


def build_normalizer(
    rules: TextRulesConfig | None = None,
    *,
    language: str = "en",
    use_default_abbreviations: bool = True,
) -> TextNormalizer:
    """构造一个合并了默认缩写表的清洗器。

    Args:
        rules: 用户规则；缩写表会与 :data:`DEFAULT_ABBREVIATIONS` 合并，
            用户配置优先。
        language: 生效语言。
        use_default_abbreviations: 是否启用内置缩写表。

    Returns:
        配置好的 :class:`TextNormalizer`。
    """
    if rules is None:
        rules = TextRulesConfig(language=language)

    if use_default_abbreviations:
        merged = dict(DEFAULT_ABBREVIATIONS)
        merged.update(rules.abbreviations or {})
        rules = replace(rules, abbreviations=merged)

    return TextNormalizer(rules, language=language)


def normalize_text(
    text: str,
    rules: TextRulesConfig | None = None,
    *,
    language: str = "en",
) -> str:
    """一行式快捷函数：清洗文本并返回结果字符串。"""
    return build_normalizer(rules, language=language).normalize(text).text
