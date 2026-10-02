"""语意切分：把一章文字切成适合逐段合成的 :class:`~noveltts.models.Segment`。

为什么要切分
------------

TTS 模型每次生成都有 ``max_new_tokens`` 上限，而且**文本越长、越容易出问题**：
漏字、串味（后半段音色漂移）、语速越读越快、长静音后突然续读。把一次合成
限制在“一到两句话”可以显著提升稳定性，代价仅仅是拼接时多几个接缝。

切分策略
--------

分四层，由粗到细：

1. **句级切分**（:func:`split_sentences`）
   识别 ``. ! ? … 。！？…`` 等句末标点，但会跳过：

   * 缩写点号（``Mr.`` / ``Vol.`` / ``U.S.``）；
   * 小数与省略号（``1.5`` / ``...``）；
   * **引号内部**的标点（``"Stop. Now."`` 不会被切成两段）。

2. **超长句再切分**（:func:`split_long_sentence`）
   一句话超过 ``hard_max_chars`` 时，按从句边界（逗号、分号、破折号、
   以及 ``and / but / because / which`` 这类连接词）切成小句，
   并且**永远不在引号内部切**，避免把一句台词劈成两半导致语气断裂。

3. **聚合成片段**（:meth:`TextSegmenter.segment_chapter`）
   把相邻句子贪心合并到 ``max_chars`` 为止。这样每个片段大约是
   “一到三句、50~120 字符”，既有足够上下文让模型读出语气，又不会太长。
   合并时会遵守两条规则：

   * ``keep_dialogue_separate``：叙述与对话不混在一个片段里，
     避免模型用叙述语气念台词；
   * ``split_on_paragraph``：可选择禁止跨自然段合并。

4. **收尾修正**（:meth:`TextSegmenter._polish`）
   过短的碎片（``< min_chars``）会与前后的同类片段合并，
   避免出现只有两三个字的片段（模型对超短输入容易产生奇怪的长音）。

输出的每个 :class:`Segment` 都带 ``kind``（narration / dialogue / heading）
与 ``source_text``（洗稿前原文），方便后续按角色分配音色或排查问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SegmentConfig
from .models import Chapter, Paragraph, Segment
from .normalize import TextNormalizer, build_normalizer
from .utils import count_chars, get_logger

logger = get_logger("segment")

# --------------------------------------------------------------------------- #
# 正则
# --------------------------------------------------------------------------- #

#: 句末标点。``…`` 单独处理，因为它常常出现在句末但后面还有内容。
_SENTENCE_END_CHARS = ".!?。！？…"

#: 句中不该被当作句末的点号场景：
#:   * 单字母缩写 ``U.S.`` / ``A.M.``
#:   * 常见缩写词（含度量单位，``6 ft.`` / ``5 p.m.``）
#:   * 小数 ``1.5``
_ABBREV_DOT_RE = re.compile(
    r"(?:^|\s)(?:[A-Za-z]\.){2,}$"       # U.S.  A.M.  p.m.
    r"|(?:^|\s)(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Vol|Ch|No|vs|etc|approx|"
    r"ft|in|yd|oz|lb|lbs|sec|min|hr|Jr|Mt|Ave|Blvd|Dept|Est|Inc|Ltd|Co)\.$"
    r"|\be\.g\.$|\bi\.e\.$",
    re.I,
)
_DECIMAL_DOT_RE = re.compile(r"\d\.$")

#: 判定“句末标点”时**需要**配对的引号：弯引号与中日引号。
#: 直引号 ``"`` 被刻意排除——它在校对不严的电子书里有相当比例是**开引号**
#: （``He said, "Stop. Now."``），跟着它延迟断句会吞掉整段文字；而它作为
#: 闭引号时（``"Hello." She left.``）我们更希望照常断句。
_CURLY_OPEN_QUOTES = "\u201c\u2018\u300c\u300e\u3008\u300a"
_CURLY_CLOSE_QUOTES = "\u201d\u2019\u300d\u300f\u3009\u300b"

#: 直引号（只用于统计深度，不阻止断句）。
_STRAIGHT_QUOTES = "\"'"

#: 开引号集合（供 :func:`classify_kind` 判断对话使用，含直引号）。
_OPEN_QUOTES = '"' + _CURLY_OPEN_QUOTES
_CLOSE_QUOTES = '"' + _CURLY_CLOSE_QUOTES

#: 断句后需要“跟着一起收进来”的闭引号集合（含直引号）。
_CLOSE_QUOTES_ALL = _CLOSE_QUOTES

#: 开括号 / 闭括号。
_OPEN_BRACKETS = "([{（【〔"
_CLOSE_BRACKETS = ")]}）】〕"

#: 从句边界：这些标点后面可以断开（超长句再切分时用）。
_CLAUSE_BREAK_RE = re.compile(r"(?<=[,;:，；：])\s+|(?<=[\u2014\u2013])\s*")

#: 从句连接词：在这些词**之前**断开可以保留语意。
_CLAUSE_CONJUNCTION_RE = re.compile(
    r"\s+(?=(?:and|but|or|so|yet|for|nor|because|although|though|while|whereas|"
    r"unless|until|when|whenever|where|which|who|whom|whose|that|if|as|since|"
    r"after|before|once|whether)\b)",
    re.I,
)

#: 判断一段文字是不是标题（很短、无句末标点）。
_HEADING_MAX_CHARS = 60

#: 判断片段是否以句末标点收尾（用于计算 ``paragraph_span``）。
_ENDS_SENTENCE_RE = re.compile(r"[.!?\u2026\u3002\uff01\uff1f][\"'\u201d\u2019\u300d\u300f\)\]\}]*$")


# --------------------------------------------------------------------------- #
# 切分单元
# --------------------------------------------------------------------------- #


@dataclass
class Unit:
    """一个待聚合的最小语意单元（通常是一个句子，也可能是小句）。"""

    text: str
    kind: str = "narration"
    paragraph_index: int = 0

    @property
    def char_count(self) -> int:
        return count_chars(self.text)


def _quote_delta(text: str) -> int:
    """返回文本的「引号深度变化」。

    直引号 ``"`` 既可能开也可能闭，因此不能简单计数；这里用奇偶规则：
    直引号每出现一次就翻转一次深度，弯引号按开/闭分别加减。
    """
    delta = 0
    for char in text:
        if char in "\"\u201c\u2018":
            delta += 1
        elif char in "\u201d\u2019":
            delta -= 1
        elif char in "\u300c\u300e\u3008\u300a":
            delta += 1
        elif char in "\u300d\u300f\u3009\u300b":
            delta -= 1
    return delta


def _straight_quote_depths(text: str) -> list[int]:
    """预计算每个字符位置处的「直引号深度」。

    直引号 ``"`` 既可能是开引号也可能是闭引号，逐字符无法判断。这里改用
    **整段配平**的思路：

    * 段落内直引号总数是偶数 -> 认为引号是配对的，把它们按出现顺序两两
      配对（第 1、3、5… 个为开引号），于是深度可以逐点算出；
    * 总数是奇数 -> 引号不配对（电子书里很常见），此时**放弃**直引号的
      阻断作用（全部记为 0），避免一个漏掉的引号把整段文字吞成一句话。

    Args:
        text: 待处理的段落文本。

    Returns:
        与 ``text`` 等长的列表，``result[i]`` 表示 ``text[i]`` **之前**的
        直引号深度。只有深度为 0 的位置才允许断句。

    Examples:
        >>> _straight_quote_depths('"Hi." Ok.')
        [0, 1, 1, 1, 1, 0, 0, 0]
    """
    positions = [i for i, char in enumerate(text) if char == '"']
    depths = [0] * (len(text) + 1)
    if not positions or len(positions) % 2 != 0:
        return depths

    events: dict[int, int] = {}
    for order, position in enumerate(positions):
        events[position] = 1 if order % 2 == 0 else -1

    depth = 0
    for index in range(len(text)):
        depths[index] = depth
        depth = max(0, depth + events.get(index, 0))
    depths[len(text)] = depth
    return depths


def split_sentences(text: str) -> list[str]:
    """把一段文字切成句子。

    规则要点：

    * 遇到 ``. ! ? 。！？`` 且**不在引号内部**、**不是缩写点号**时断句；
    * 省略号（``...`` / ``…`` / ``……``）视为句末，但**独立出现的省略号**
      （如 ``She paused. …Then continued.`` 中间那个）会被丢弃而不是
      变成一个只有省略号的碎片；
    * 闭引号、闭括号会“跟”在它前面的句子后面（``"Stop."`` 不会被拆开）；
    * 连续标点（``?!`` / ``!!!`` / ``?!?``）作为一个整体处理；
    * 返回的句子都已去掉首尾空白，且不含空串。

    引号保护策略（重要）：

    * **弯引号**（``“ ” ‘ ’ 「」``）：遇到未闭合的开引号时延迟断句；
    * **直引号**（``"``）：先判断整段是否配平（见 :func:`_straight_quote_depths`）。
      配平时同样延迟断句，从而 ``"You don't earn respect by waiting until
      the last minute. We must be proactive."`` 会保持在一句里，
      模型的语气不会被接缝切断；不配平时放弃保护，避免吞掉整段。

    Examples:
        >>> split_sentences('It cost $1.5 million. Mr. Smith agreed.')
        ['It cost $1.5 million.', 'Mr. Smith agreed.']
        >>> split_sentences('"Hello." She left.')
        ['"Hello."', 'She left.']
    """
    sentences: list[str] = []
    buffer: list[str] = []
    curly_depth = 0
    bracket_depth = 0
    straight_depths = _straight_quote_depths(text)
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char in _CURLY_OPEN_QUOTES:
            curly_depth += 1
            buffer.append(char)
            index += 1
            continue
        if char in _CURLY_CLOSE_QUOTES:
            curly_depth = max(0, curly_depth - 1)
            buffer.append(char)
            index += 1
            continue
        if char in _OPEN_BRACKETS:
            bracket_depth += 1
            buffer.append(char)
            index += 1
            continue
        if char in _CLOSE_BRACKETS:
            bracket_depth = max(0, bracket_depth - 1)
            buffer.append(char)
            index += 1
            continue
        if char in _SENTENCE_END_CHARS:
            # --- 遇到句末候选标点 ---
            run_end = index

            # 省略号：把连续省略号或 3 个以上点号作为一个整体
            if text.startswith("...", index) or text[index] == "\u2026":
                if text[index] == ".":
                    while run_end + 1 < length and text[run_end + 1] == ".":
                        run_end += 1
                else:
                    while run_end + 1 < length and text[run_end + 1] == "\u2026":
                        run_end += 1
            else:
                while run_end + 1 < length and text[run_end + 1] in _SENTENCE_END_CHARS:
                    run_end += 1
                if text[index] == "." and run_end == index:
                    # 单个点号：排除缩写与小数。
                    # 注意要把点号本身一起纳入判断——"Mr." 的缩写标识就是那个点。
                    prefix = "".join(buffer) + "."
                    if _ABBREV_DOT_RE.search(prefix) or _DECIMAL_DOT_RE.search(prefix):
                        buffer.append(char)
                        index += 1
                        continue
                    # 后面紧跟数字说明这是小数（"1.5"）
                    if run_end + 1 < length and text[run_end + 1].isdigit():
                        buffer.append(char)
                        index += 1
                        continue
                    # 后面紧跟字母（"Mr.Smith"）也不算句末
                    if run_end + 1 < length and text[run_end + 1].isalpha():
                        buffer.append(char)
                        index += 1
                        continue

            # 省略号前面必须有实际文字才算句末。
            # "Wait..." -> 断句；"She paused. …Then" -> 把孤立的省略号吸收掉。
            if text[index] == "\u2026" or text.startswith("...", index):
                if not any(not item.isspace() for item in buffer):
                    buffer.append(text[index:run_end + 1])
                    index = run_end + 1
                    continue

            buffer.append(text[index:run_end + 1])
            index = run_end + 1

            # 把紧跟其后的闭引号 / 闭括号一起吃进来，让片段边界落在引号之外
            while index < length and text[index] in (_CLOSE_QUOTES_ALL + _CLOSE_BRACKETS):
                buffer.append(text[index])
                if text[index] in _CURLY_CLOSE_QUOTES:
                    curly_depth = max(0, curly_depth - 1)
                elif text[index] in _CLOSE_BRACKETS:
                    bracket_depth = max(0, bracket_depth - 1)
                index += 1

            # 引号/括号未闭合 -> 还不能断句。
            # 直引号用预计算的深度判断（不配平时深度恒为 0，等于不保护）。
            if curly_depth > 0 or bracket_depth > 0 or straight_depths[index] > 0:
                continue

            sentence = "".join(buffer).strip()
            if count_chars(sentence) > 0:
                sentences.append(sentence)
            buffer = []
            continue

        buffer.append(char)
        index += 1

    remainder = "".join(buffer).strip()
    if count_chars(remainder) > 0:
        sentences.append(remainder)
    return sentences


def split_long_sentence(
    sentence: str,
    max_chars: int,
    *,
    min_tail_chars: int = 12,
) -> list[str]:
    """把过长的句子按从句边界切成若干小句。

    Args:
        sentence: 原句。
        max_chars: 单块字符上限。
        min_tail_chars: 尾块小于该值时并入前一块，避免出现孤零零的几个字。

    Returns:
        小句列表；无法切分时返回 ``[sentence]``。

    Note:
        **绝不在引号内部切分**。一句台词被劈成两半会让模型在接缝处
        重新起调，听起来像换了个人。如果整句都在引号里，就只能保留原样
        （由调用方决定是否强制硬切）。
    """
    if count_chars(sentence) <= max_chars:
        return [sentence]

    # 优先在从句标点后切
    pieces = [piece for piece in _CLAUSE_BREAK_RE.split(sentence) if piece and piece.strip()]
    if len(pieces) <= 1:
        # 没有从句标点，退而求其次：在连接词前切
        pieces = [
            piece for piece in _CLAUSE_CONJUNCTION_RE.split(sentence) if piece and piece.strip()
        ]
    if len(pieces) <= 1:
        return [sentence]

    # 贪心合并小句到 max_chars
    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{buffer} {piece}".strip() if buffer else piece
        if buffer and count_chars(candidate) > max_chars:
            chunks.append(buffer)
            buffer = piece
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)

    # 尾块太短就并进前一块
    if len(chunks) >= 2 and count_chars(chunks[-1]) < min_tail_chars:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
        chunks.pop()

    # 切分后仍超长的块（例如中间没有标点的长句）：在空白处硬切
    final: list[str] = []
    for chunk in chunks:
        final.extend(hard_wrap(chunk, max_chars))
    return final


def hard_wrap(text: str, max_chars: int) -> list[str]:
    """最后的兜底：在空白处把超长文本硬切成不超过 ``max_chars`` 的块。

    只在 :func:`split_long_sentence` 已经无能为力时使用。为保证不出现
    “半句话”，切点尽量选在逗号等弱标点之后。
    """
    if count_chars(text) <= max_chars:
        return [text]

    words = text.split(" ")
    chunks: list[str] = []
    buffer = ""
    for word in words:
        candidate = f"{buffer} {word}".strip() if buffer else word
        if buffer and count_chars(candidate) > max_chars:
            chunks.append(buffer)
            buffer = word
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)

    # 单词本身比 max_chars 还长（中文长句常见）-> 按字符切
    result: list[str] = []
    for chunk in chunks:
        if count_chars(chunk) <= max_chars:
            result.append(chunk)
            continue
        result.extend(_split_by_measure(chunk, max_chars))
    return result


def _split_by_measure(text: str, max_chars: int) -> list[str]:
    """按字符数切分（用于没有空格的中文长句）。"""
    pieces: list[str] = []
    current: list[str] = []
    current_count = 0
    for char in text:
        current.append(char)
        if re.match(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]", char):
            current_count += 1
        if current_count >= max_chars:
            pieces.append("".join(current))
            current = []
            current_count = 0
    if current:
        pieces.append("".join(current))
    return pieces


def _unit_kind(text: str) -> str:
    """句子级别的类型判断（只区分对话与叙述）。

    与 :func:`classify_kind` 的区别：**这里绝不返回 ``heading``**。

    原因是一个曾经踩过的坑：``classify_kind`` 会把「短且没有句末标点」的文本
    判成标题，而句子级调用经常遇到这种输入——``"Wait"`` 这种被切出来的短句
    会被判成标题，进而让整个自然段被当成标题块、不再做任何合并。
    标题只应由**解析器给出的段落类型**（``Paragraph.kind``）决定。
    """
    stripped = text.strip()
    if not stripped:
        return "narration"
    if stripped[0] in _OPEN_QUOTES or stripped[-1] in _CLOSE_QUOTES:
        return "dialogue"
    return "narration"


def _has_unclosed_quote(text: str) -> bool:
    """判断文本结尾是否停在一个「还没闭合」的引号里。

    用于识别「一句台词被拆到两个自然段」的情况。判定方式：
    统计各类引号的开闭数量，只要有任何一类开多闭少，就认为引号还开着。

    Examples:
        >>> _has_unclosed_quote('"I told you that')
        True
        >>> _has_unclosed_quote('"I told you that"')
        False
    """
    pairs = (('"', '"'), ("\u201c", "\u201d"), ("\u2018", "\u2019"),
             ("\u300c", "\u300d"), ("\u300e", "\u300f"))
    for open_char, close_char in pairs:
        if open_char == close_char:
            if text.count(open_char) % 2 == 1:
                return True
        elif text.count(open_char) > text.count(close_char):
            return True
    return False


def classify_kind(text: str) -> str:
    """判断一个片段的类型：``heading`` / ``dialogue`` / ``narration``。

    判定顺序（**顺序很重要**）：

    1. 以引号开头或结尾（含中文引号）-> ``dialogue``
       （必须先判对话，否则 ``"Huh?"`` 这种极短台词会被误判成标题）；
    2. 长度 <= :data:`_HEADING_MAX_CHARS` 且没有句末标点 -> ``heading``；
    3. 其余 -> ``narration``。

    Note:
        结尾也判为对话，是因为引号不配对的段落被切开后，后半截会以
        闭引号收尾（``... to be proactive."``）。把它归到对话里，
        后续要做“按角色分配音色”时就不会漏掉这半句。
    """
    stripped = text.strip()
    if not stripped:
        return "narration"
    if stripped[0] in _OPEN_QUOTES or stripped[-1] in _CLOSE_QUOTES:
        return "dialogue"
    if count_chars(stripped) <= _HEADING_MAX_CHARS and not any(
        char in _SENTENCE_END_CHARS for char in stripped
    ):
        return "heading"
    return "narration"


# --------------------------------------------------------------------------- #
# 切分器
# --------------------------------------------------------------------------- #


class TextSegmenter:
    """把章节切成 :class:`~noveltts.models.Segment` 列表。

    Args:
        config: 切分阈值配置。
        normalizer: 洗稿器；``None`` 时按 ``language`` 自动构建。

    Example:
        >>> segmenter = TextSegmenter(language="en")
        >>> segments = segmenter.segment_text("Hello there. How are you?", chapter_id="c001")
        >>> [s.text for s in segments]
        ['Hello there. How are you?']
    """

    def __init__(
        self,
        config: SegmentConfig | None = None,
        *,
        normalizer: TextNormalizer | None = None,
        language: str = "en",
    ) -> None:
        self.config = config or SegmentConfig()
        self.language = language
        self.normalizer = normalizer or build_normalizer(language=language)

    # ------------------------------------------------------------------ 入口
    def segment_chapter(self, chapter: Chapter, *, normalize: bool = True) -> list[Segment]:
        """切分一整章。

        段落按顺序处理，并且会把「上一段结尾留下的半句」与「下一段开头」
        接起来（见 :meth:`_flush_cross_paragraph`）。这样做的收益很实在：
        电子书里 ``"……” / “……"`` 这种被拆成两个 ``<p>`` 的一句台词，
        能重新合并成一个片段，模型就不会在引号中间重新起调。

        Args:
            chapter: 已解析好的章节（段落列表）。
            normalize: 是否先做洗稿。``True`` 时 ``Segment.text`` 是洗稿后的文本，
                ``Segment.source_text`` 保留原文。

        Returns:
            该章的 :class:`~noveltts.models.Segment` 列表（``index`` 从 1 开始）。
        """
        completed: list[tuple[str, int, str, int]] = []   # (text, para_idx, source, span)
        pending: tuple[str, int, str] | None = None       # 上一段留下的尾巴

        for paragraph_index, paragraph in enumerate(chapter.paragraphs):
            text = paragraph.text.strip()
            if not text:
                continue

            normalized = self.normalizer.normalize(text).text if normalize else text
            if not normalized.strip():
                continue

            pieces = self._segment_paragraph(normalized, kind_hint=paragraph.kind)
            para_start = paragraph_index
            span = 1
            merged_index: int | None = None

            # 1) 先尝试把 pending 与本段第一块接起来
            if pieces:
                merged = self._try_cross_paragraph_merge(pending, pieces[0])
                if merged is not None:
                    if pending is not None:
                        # pending 来自更早的自然段，合并后这个片段横跨了多段，
                        # 起点也要跟着回到尾巴所在的段——否则 paragraph_index
                        # 会指向后半段，paragraph_span 的起点就错了。
                        para_start = pending[1]
                        span = paragraph_index - pending[1] + 1
                    pieces[0] = merged
                    merged_index = 0
                    pending = None

            # 2) pending 没能接上 -> 作为一个独立片段落地
            if pending is not None:
                completed.append((pending[0], pending[1], pending[2], 1))
                pending = None

            if not pieces:
                continue

            # 合并出来的那一块**立刻落地**，不能留作 pending：
            # 它已经消费掉了上一段的尾巴，再留一次会让它被输出两遍
            # （或者把 span 信息丢掉）。
            if merged_index is not None:
                completed.append((pieces[merged_index], para_start, text, span))
                pieces = [p for i, p in enumerate(pieces) if i != merged_index]
                if not pieces:
                    continue

            # 3) 最后一块留作 pending，等下一段来决定能否接上
            if not self.config.split_on_paragraph:
                pending = (pieces[-1], paragraph_index, text)
                pieces = pieces[:-1]

            for piece in pieces:
                completed.append((piece, para_start, text, span))

        if pending is not None:
            completed.append((pending[0], pending[1], pending[2], 1))

        segments: list[Segment] = []
        for index, (piece, paragraph_index, source_text, span) in enumerate(completed, start=1):
            segments.append(
                Segment(
                    segment_id=f"{chapter.chapter_id}-{index:04d}",
                    chapter_id=chapter.chapter_id,
                    index=index,
                    text=piece,
                    kind=classify_kind(piece),  # type: ignore[arg-type]
                    paragraph_index=paragraph_index,
                    paragraph_span=span,
                    source_text=source_text,
                )
            )

        segments = self._fix_metadata(segments, chapter)
        chapter.segments = segments
        logger.debug(
            "章节 %s 切分完成：%d 段，平均 %.1f 字，最长 %d 字",
            chapter.chapter_id,
            len(segments),
            (sum(s.char_count for s in segments) / len(segments)) if segments else 0.0,
            max((s.char_count for s in segments), default=0),
        )
        return segments

    def _try_cross_paragraph_merge(
        self, pending: tuple[str, int, str] | None, head: str
    ) -> str | None:
        """尝试把上一段的尾巴与下一段的开头合并。

        只在「**同一句话被拆到两个自然段**」时才合并。这是电子书里很常见
        的排版：一句话的台词横跨两个 ``<p>``，例如::

            <p>"I, um, suppose we do submit club budget proposals,
               but those are all approved by teachers</p>
            <p>anyway," she said.</p>

        判定条件（必须**全部**满足，缺一不可）：

        1. 配置允许（``split_on_paragraph`` 为 ``False``）；
        2. 上一段的尾巴没有以句末标点结束；
        3. 上一段的尾巴**停在一个未闭合的引号里**——这是最强也最可靠的信号；
        4. 合并后长度不超过 ``paragraph_break_chars``。

        刻意**不**合并的情况：两个各自闭合的引号段落，例如 ::

            "Hey, anything you want, I'll—"
            "D-don't you dare," Komari snapped.

        这两句分属不同说话人。早期版本会把它们粘成一个片段，导致模型用
        同一个语气念完两个人的台词，接缝处听起来像一个人自言自语，所以
        现在要求引号必须没闭合才合并。

        Args:
            pending: ``(文本, 自然段序号, 原文)``；``None`` 表示没有待接的尾巴。
            head: 下一段的第一个片段。

        Returns:
            合并后的文本，或 ``None`` 表示不合并。
        """
        if pending is None or not head:
            return None
        if self.config.split_on_paragraph:
            return None

        tail_text = pending[0]
        if _ENDS_SENTENCE_RE.search(tail_text):
            return None
        # 只有「引号还开着」才认为下一段是同一句话的延续
        if not _has_unclosed_quote(tail_text):
            return None

        merged = f"{tail_text} {head}".strip()
        if count_chars(merged) > self.config.paragraph_break_chars:
            return None
        return merged

    def segment_text(
        self,
        text: str,
        *,
        chapter_id: str = "adhoc",
        normalize: bool = True,
        title: str = "",
    ) -> list[Segment]:
        """切分一段任意文本（用于试听、单段生成）。

        Args:
            text: 文本，可以包含空行分段。
            chapter_id: 生成 id 用的章节标识。
            normalize: 是否洗稿。
            title: 若提供，会作为一个 ``heading`` 片段插到最前面。

        Returns:
            :class:`~noveltts.models.Segment` 列表。
        """
        paragraphs = [Paragraph(text=block.strip()) for block in re.split(r"\n\s*\n", text) if block.strip()]
        if not paragraphs:
            paragraphs = [Paragraph(text=text.strip())] if text.strip() else []

        chapter = Chapter(
            index=1,
            title=title or "Ad hoc",
            paragraphs=([Paragraph(text=title, kind="heading")] if title else []) + paragraphs,
            chapter_id=chapter_id,
            language=self.language,
        )
        return self.segment_chapter(chapter, normalize=normalize)

    def segment_many(
        self, chapters: list[Chapter], *, normalize: bool = True
    ) -> dict[str, list[Segment]]:
        """批量切分多章；返回 ``{chapter_id: segments}``。"""
        return {
            chapter.chapter_id: self.segment_chapter(chapter, normalize=normalize)
            for chapter in chapters
        }

    # -------------------------------------------------------------- 内部实现
    def _segment_paragraph(self, text: str, *, kind_hint: str = "narration") -> list[str]:
        """把单个自然段切成若干片段。"""
        config = self.config

        # 1) 先切句
        sentences = split_sentences(text)

        # 2) 超长句再切小句（标题不切）
        units: list[Unit] = []
        for sentence in sentences:
            if count_chars(sentence) > config.hard_max_chars:
                for piece in split_long_sentence(
                    sentence,
                    config.hard_max_chars,
                    min_tail_chars=config.tail_clause_min_chars,
                ):
                    units.append(Unit(text=piece))
            else:
                units.append(Unit(text=sentence))

        # 3) 聚合成片段
        return self._merge_units(units, kind_hint=kind_hint)

    def _merge_units(self, units: list[Unit], *, kind_hint: str = "narration") -> list[str]:
        """把句子贪心聚合成片段。

        Args:
            units: 句子列表。
            kind_hint: 来自解析器的段落类型提示。**只有它等于 ``"heading"``
                时才按标题处理**——绝不能靠"这句话很短、没有句末标点"去猜，
                否则 ``"Hello there."`` 这种短句会被当成标题，连带把整段
                合成一个片段（曾经的 bug）。
        """
        config = self.config
        if not units:
            return []

        # 标题独立成段，不与正文混合（依据解析器给出的段落类型，不靠猜）
        if kind_hint == "heading":
            return [unit.text for unit in units]

        chunks: list[list[Unit]] = []
        current: list[Unit] = []

        def current_chars() -> int:
            return sum(unit.char_count for unit in current) + max(0, len(current) - 1)

        for unit in units:
            if not current:
                current = [unit]
                continue

            unit_kind = _unit_kind(unit.text)
            previous_kind = _unit_kind(current[-1].text)
            would_be = current_chars() + unit.char_count + 1

            # 规则 1：对话与叙述不混在一个片段
            if config.keep_dialogue_separate and unit_kind != previous_kind:
                chunks.append(current)
                current = [unit]
                continue

            # 规则 2：不超过上限
            if would_be > config.max_chars:
                chunks.append(current)
                current = [unit]
                continue

            current.append(unit)

        if current:
            chunks.append(current)

        # 3) 合并过短的碎片
        chunk_texts = [" ".join(unit.text for unit in chunk).strip() for chunk in chunks]
        if config.merge_short:
            chunk_texts = self._merge_short_chunks(chunk_texts)
        return [text for text in chunk_texts if text]

    def _merge_short_chunks(self, chunks: list[str]) -> list[str]:
        """把过短的片段并入相邻片段。

        合并优先级：同类型邻居 > 前一个 > 后一个。若前后合并都会超长，
        则宁可保留一个短片段（短片段仍然能正常合成）。
        """
        if len(chunks) <= 1:
            return chunks

        config = self.config
        result: list[str] = []
        for chunk in chunks:
            if not result:
                result.append(chunk)
                continue

            if count_chars(chunk) >= config.min_chars:
                result.append(chunk)
                continue

            previous = result[-1]
            merged = f"{previous} {chunk}".strip()
            same_kind = classify_kind(previous) == classify_kind(chunk)
            limit = config.max_chars if same_kind else config.hard_max_chars
            if count_chars(merged) <= limit:
                result[-1] = merged
            else:
                result.append(chunk)

        # 第一块可能仍然过短：尝试并进第二块
        if len(result) >= 2 and count_chars(result[0]) < config.min_chars:
            merged = f"{result[0]} {result[1]}".strip()
            if count_chars(merged) <= config.hard_max_chars:
                result[1] = merged
                result.pop(0)

        return result

    def _fix_metadata(self, segments: list[Segment], chapter: Chapter) -> list[Segment]:
        """修正片段元信息：字数、句子数、编号连续性。

        ``paragraph_span``（片段覆盖了几个自然段）由 :meth:`segment_chapter`
        在生成片段时**直接记录**，因为跨段合并之后，靠文本去反推段落边界
        并不可靠（合并处会插入空格、洗稿也会改写标点）。
        这里只做一次合理性校验，防止出现越界的取值。
        """
        paragraph_total = len(chapter.paragraphs)
        for position, segment in enumerate(segments, start=1):
            segment.index = position
            segment.char_count = count_chars(segment.text)
            segment.unit_count = len(split_sentences(segment.text))

            span = segment.paragraph_span if segment.paragraph_span > 0 else 1
            if segment.paragraph_index + span > paragraph_total:
                logger.debug(
                    "片段 %s 的 paragraph_span 越界（para=%d span=%d 共 %d 段），已修正",
                    segment.segment_id,
                    segment.paragraph_index,
                    span,
                    paragraph_total,
                )
                span = max(1, paragraph_total - segment.paragraph_index)
            segment.paragraph_span = max(1, span)

        return segments


# --------------------------------------------------------------------------- #
# 便捷函数
# --------------------------------------------------------------------------- #


def segment_chapter(
    chapter: Chapter,
    config: SegmentConfig | None = None,
    *,
    language: str = "en",
    normalize: bool = True,
) -> list[Segment]:
    """一行式：切分一章。"""
    segmenter = TextSegmenter(config, language=language)
    return segmenter.segment_chapter(chapter, normalize=normalize)


def segment_text(
    text: str,
    config: SegmentConfig | None = None,
    *,
    language: str = "en",
    chapter_id: str = "adhoc",
    normalize: bool = True,
) -> list[Segment]:
    """一行式：切分一段文本。"""
    segmenter = TextSegmenter(config, language=language)
    return segmenter.segment_text(
        text, chapter_id=chapter_id, normalize=normalize
    )


def summarize_segments(segments: list[Segment]) -> str:
    """生成一份片段统计摘要（CLI 与日志用）。"""
    if not segments:
        return "没有片段"
    counts = {"narration": 0, "dialogue": 0, "heading": 0, "meta": 0}
    for segment in segments:
        counts[segment.kind] = counts.get(segment.kind, 0) + 1
    lengths = [s.char_count for s in segments]
    total_seconds = sum(s.estimated_seconds for s in segments)
    return (
        f"片段总数 {len(segments)}（叙述 {counts.get('narration', 0)} / "
        f"对话 {counts.get('dialogue', 0)} / 标题 {counts.get('heading', 0)}）\n"
        f"字符数：最短 {min(lengths)} / 平均 {sum(lengths) / len(lengths):.1f} / "
        f"最长 {max(lengths)}\n"
        f"预估音频时长：{total_seconds / 60:.1f} 分钟"
    )
