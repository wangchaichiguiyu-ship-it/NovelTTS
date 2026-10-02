"""配置系统：YAML 文件 + 命令行覆盖 -> 强类型 dataclass。

层次结构（与 ``config.yaml`` 一一对应）::

    AppConfig
    ├── ProjectConfig     输入/输出/中间产物路径
    ├── ParseConfig       EPUB/TXT 切章策略
    ├── TextRulesConfig   洗稿规则（缩写、数字、符号、韵律）
    ├── SegmentConfig     语意切分阈值
    ├── VoiceConfig       音色克隆参考音频
    ├── GenerationConfig  采样参数、设备、批量、重试
    ├── AudioConfig       音频格式与响度归一
    └── EncodeConfig      压缩封装

所有路径字段在 :meth:`AppConfig.load` 里统一解析为相对**项目根目录**的绝对路径，
因此无论从哪个工作目录执行 CLI，行为都一致。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

from .utils import deep_merge, get_logger, resolve_path

logger = get_logger("config")

#: 支持的输出压缩格式。
EncodeFormat = Literal["m4a", "mp3", "opus", "flac", "copy"]
#: 支持的设备。
DeviceName = Literal["auto", "cuda", "mps", "cpu"]
#: 支持的浮点精度。
DtypeName = Literal["bfloat16", "float16", "float32"]


# --------------------------------------------------------------------------- #
# 各子配置
# --------------------------------------------------------------------------- #


@dataclass
class ProjectConfig:
    """路径配置。

    Attributes:
        root: 项目根目录（``config.yaml`` 所在目录）。相对路径都相对它解析。
        source_dir: 原始素材目录（epub/txt 放这里）。
        work_dir: 中间产物目录（解析结果、segments JSON、纯文本）。
        output_dir: 最终产物目录（片段 wav、章节 wav、压缩档）。
        cache_dir: 音色 prompt 等缓存。
    """

    root: str = "."
    source_dir: str = "sources"
    work_dir: str = "output/work"
    output_dir: str = "output/audio"
    cache_dir: str = "output/cache"

    # 解析后的绝对路径（由 AppConfig.load 填充）
    root_path: Path = field(default=Path("."), repr=False)
    source_path: Path = field(default=Path("sources"), repr=False)
    work_path: Path = field(default=Path("output/work"), repr=False)
    output_path: Path = field(default=Path("output/audio"), repr=False)
    cache_path: Path = field(default=Path("output/cache"), repr=False)

    def resolve(self, root: Path) -> None:
        """把所有路径解析为绝对路径，并建立目录。"""
        self.root_path = Path(root).resolve()
        self.source_path = resolve_path(self.source_dir, base=self.root_path)
        self.work_path = resolve_path(self.work_dir, base=self.root_path)
        self.output_path = resolve_path(self.output_dir, base=self.root_path)
        self.cache_path = resolve_path(self.cache_dir, base=self.root_path)

    def ensure_dirs(self) -> None:
        for path in (self.work_path, self.output_path, self.cache_path):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ParseConfig:
    """切章策略。

    Attributes:
        min_chapter_chars: 小于该字符数的单元（版权页、图片页等）默认丢弃。
        skip_title_patterns: 命中任一正则的标题会被丢弃（大小写不敏感）。
        merge_short_into_previous: 短单元是“并进上一章”还是“直接丢弃”。
        drop_reference_sections: 是否丢弃参考文献/版权/目录等非正文单元。
        default_language: 未探测出语言时使用的默认值。
        include_nav_titles: 是否把 EPUB 目录标题当作章节标题。
    """

    min_chapter_chars: int = 200
    skip_title_patterns: list[str] = field(
        default_factory=lambda: [
            r"^table of contents",
            r"^contents$",
            r"^color gallery",
            r"^title page$",
            r"^copyright",
            r"^credits",
            r"^newsletter",
            r"^also available",
            r"^about the (author|series)",
            r"^cover",
        ]
    )
    merge_short_into_previous: bool = False
    drop_reference_sections: bool = True
    default_language: str = "en"
    include_nav_titles: bool = True

    #: 需要**逐段**丢弃的样板文字模式（大小写不敏感，做 ``search`` 匹配）。
    #:
    #: 版权页、ISBN 行、"All rights reserved" 这类内容读出来毫无意义，
    #: 而且往往夹杂数字与全大写字母，是 TTS 最容易念错的地方。
    #: 默认值刻意**保守**：只收录几乎不可能出现在正文里的句子。
    #: 命中的段落会被丢弃，并写进 DEBUG 日志，方便核对是否误伤。
    exclude_paragraph_patterns: list[str] = field(
        default_factory=lambda: [
            r"\bAll rights reserved\b",
            r"\bISBN[\s:：]*[\d\-Xx]{10,17}",
            r"\bReproduction in whole or in part\b",
            r"\bwithout (?:the )?(?:prior )?written permission\b",
            r"^Copyright\s*(?:\u00a9|\(c\))?\s*\d{4}",
            r"\bThis (?:book|work) is a work of fiction\b",
            r"\bPrinted in\b[^.]{0,40}\b(?:China|USA|United States|Japan)\b",
        ]
    )


@dataclass
class TextRulesConfig:
    """洗稿（text normalization）规则开关。

    每一项都可以单独关掉，方便对比“洗稿前/后”对听感的影响。

    Attributes:
        enabled: 总开关。
        language: 生效语言；``"auto"`` 表示按章节语言自动选择。
        protect_patterns: 先保护起来、完全不改写的片段（URL、邮箱等）。
        abbreviations: 模型容易念错的缩写 -> 口语化展开。
        acronym_spellout: 命中这些模式的缩写按字母念（``NHK`` -> ``N H K``）。
        large_numbers: 大数字口语化阈值（如 ``1500`` -> ``fifteen hundred``）。
        year_range: 形如 ``1990-1995`` 的年份区间处理为 ``1990 to 1995``。
        ordinal_numbers: ``1st`` / ``2nd`` -> ``first`` / ``second``。
        number_followed_by_noun: 把 ``3 apples`` 这类量词前的数字转成词。
        number_ranges: ``870-120`` 这类数字区间读成 ``A to B``。
        roman_numerals: ``Vol. IV`` -> ``Volume four``。
        dashes: ``A—B`` / ``A–B``（非数字）转成逗号停顿。
        quotes: 把弯引号统一成直引号；``CJK`` 时统一成直角引号。
        ellipsis: ``...`` 统一成单字符省略号。
        unicode_normalize: NFC / NFKC 归一化。
        collapse_whitespace: 折叠多余空白与换行。
        strip_markup: 去掉残留的富文本标记（``<i>`` 等）。
        cjk_numbers: 中文数字规则（``三十`` -> ``30``，电话式数字逐位念）。
        cjk_measure_pairs: 量词补全映射。
        custom_replacements: 你自己的“正则 -> 替换”规则（最后执行）。
    """

    enabled: bool = True
    language: str = "auto"
    protect_patterns: list[str] = field(
        default_factory=lambda: [
            r"https?://\S+",
            r"www\.\S+",
            r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
        ]
    )
    abbreviations: dict[str, str] = field(default_factory=dict)
    acronym_spellout: list[str] = field(
        default_factory=lambda: [r"\b(?:[A-Z]\.){2,}", r"\b[A-Z]{2,5}\b"]
    )
    large_numbers: int = 1100
    year_range: bool = True
    ordinal_numbers: bool = True
    number_followed_by_noun: bool = True
    number_ranges: bool = True
    roman_numerals: bool = True
    dashes: bool = True
    quotes: str = "straight"  # straight | cjk | keep
    ellipsis: bool = True
    unicode_normalize: str = "NFC"  # NFC | NFKC | none
    collapse_whitespace: bool = True
    strip_markup: bool = True
    cjk_numbers: bool = True
    cjk_measure_pairs: dict[str, str] = field(
        default_factory=lambda: {
            "个": "个",
            "只": "只",
            "岁": "岁",
            "年": "年",
            "月": "月",
            "日": "日",
            "点": "点",
            "分": "分",
            "秒": "秒",
            "元": "元",
            "块": "块",
            "米": "米",
            "公里": "公里",
            "层": "层",
            "号": "号",
            "级": "级",
            "班": "班",
            "人": "人",
            "次": "次",
            "遍": "遍",
            "倍": "倍",
            "成": "成",
        }
    )
    custom_replacements: dict[str, str] = field(default_factory=dict)

    #: 需要用「数字上下文」约束的缩写：只有紧跟数字时才展开。
    #:
    #: ``No.`` / ``no.`` 是最典型的坑——在 ``No. 5`` 里它是 Number，
    #: 但在 ``"No. Not everything."`` 里它是口语否定词。展开错了会把对话
    #: 变成 "Number. Not everything."，非常出戏。
    numeric_context_abbreviations: dict[str, str] = field(
        default_factory=lambda: {"No.": "Number", "no.": "number"}
    )


@dataclass
class SegmentConfig:
    """语意切分阈值。

    切分的目的是让模型每次只处理“一小段话”，从而避免长文本常见的
    漏字、串味、语速漂移问题。

    Attributes:
        max_chars: 单个片段的目标上限（可朗读字符数）。
        min_chars: 单个片段的目标下限；过短的碎片会与相邻句合并。
        hard_max_chars: 硬上限，超过则在最接近的空白/标点处强制断开。
        merge_short: 是否合并过短片段。
        keep_dialogue_separate: 对话与叙述是否尽量分开成不同片段。
        split_on_paragraph: 是否禁止跨自然段合并（``True`` 时更保守）。
        paragraph_break_chars: 跨段合并时允许的总长度上限。
        tail_clause_min_chars: 从句切分时，尾部残句小于该值时并入前句。
    """

    max_chars: int = 120
    min_chars: int = 24
    hard_max_chars: int = 200
    merge_short: bool = True
    keep_dialogue_separate: bool = True
    split_on_paragraph: bool = False
    paragraph_break_chars: int = 90
    tail_clause_min_chars: int = 12


@dataclass
class VoiceConfig:
    """音色克隆配置（Base 模型专有）。

    Attributes:
        ref_audio: 参考音频路径。
        ref_text: 参考音频对应的文本。ICL 模式必填；留空则自动退化。
        x_vector_only_mode: ``True`` 只用说话人向量（不需要 ref_text，但相似度略低）。
        mode: ``icl`` 用参考音频+文本（质量最好）；``xvec`` 只用音色向量（最省内存）。
        prompt_cache: 是否把算好的 prompt 缓存到磁盘（跨进程复用，省一次编码）。
    """

    ref_audio: str = ""
    ref_text: str = ""
    x_vector_only_mode: bool = False
    mode: Literal["icl", "xvec"] = "icl"
    prompt_cache: bool = True


@dataclass
class GenerationConfig:
    """模型加载与采样参数。

    Attributes:
        model_path: 本地模型目录或 HuggingFace/ModelScope 仓库 id。
        device: ``auto`` 会按 cuda > mps > cpu 选择。
        dtype: 权重精度。``bfloat16`` 最省内存（约为 float32 的一半）。
        attn_implementation: ``flash_attention_2``（仅 CUDA）或 ``sdpa`` / ``eager``。
        max_new_tokens: 单次生成上限。片段级合成用不到 2048（模型卡评测值），
            默认 1024 已非常充裕；调大只会让显存预留变多。
        batch_size: 一次前向的片段数。16 GB 内存建议 1；云端显存充足可调到 4~8。
        non_streaming_mode: ``True`` 一次性看全文（质量略好）；``False`` 模拟流式输入。
        num_retries: 单段失败重试次数。
        retry_backoff: 重试间隔基数（秒），实际间隔为 ``base * 2**attempt``。
        skip_existing: 已存在的片段 wav 是否跳过（断点续跑的关键）。
        keep_last_segments: 只生成最后 N 段（供试听）；``0`` 表示全部。
        progress_every: 每 N 段打印一次进度。
    """

    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device: DeviceName = "auto"
    dtype: DtypeName = "bfloat16"
    attn_implementation: str = "auto"
    max_new_tokens: int = 1024
    batch_size: int = 1
    non_streaming_mode: bool = True
    num_retries: int = 2
    retry_backoff: float = 1.5
    skip_existing: bool = True
    keep_last_segments: int = 0
    progress_every: int = 10

    # 采样参数；``None`` 表示沿用模型自带 ``generate_config.json``。
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None

    language: str = "English"


@dataclass
class AudioConfig:
    """音频产物格式。

    Attributes:
        sample_rate: 期望采样率；``0`` 表示沿用模型输出（24 kHz）。
        channels: 声道数，默认单声道。
        normalize: 响度归一模式：``peak`` 峰值归一（快、稳定）、
            ``rms`` 感知响度归一（段间更一致）、``none`` 关闭。
        target_dbfs: ``peak`` 模式的目标峰值。
        target_rms_dbfs: ``rms`` 模式的目标 RMS。
        max_gain_db: 增益上限，避免把底噪放大。
        trim_silence: 是否裁掉片段首尾的静音（减少拼接处的怪停顿）。
        silence_threshold_dbfs: 静音判定门限。
        min_gap_seconds: 片段之间的默认停顿。
        tail_quiet_gain: 尾部已有静音时，停顿缩小的比例。
        tail_quiet_threshold_dbfs: 判定“尾部已有静音”的门限。
        join_fade_ms: 拼接处淡入淡出时长，用于消除咔嗒声。
        save_segments: 是否保留片段 wav（压缩为 16-bit 后体积很小，建议保留）。
    """

    sample_rate: int = 24_000
    channels: int = 1
    normalize: Literal["peak", "rms", "none"] = "peak"
    target_dbfs: float = -1.5
    target_rms_dbfs: float = -20.0
    max_gain_db: float = 12.0
    trim_silence: bool = False
    silence_threshold_dbfs: float = -50.0
    min_gap_seconds: float = 0.32
    tail_quiet_gain: float = 0.35
    tail_quiet_threshold_dbfs: float = -38.0
    join_fade_ms: float = 15.0
    save_segments: bool = True


@dataclass
class EncodeConfig:
    """章节成品的压缩封装（依赖 ffmpeg）。"""

    enabled: bool = True
    format: EncodeFormat = "m4a"
    bitrate: str = "64k"
    sample_rate: int = 24_000
    channels: int = 1
    ffmpeg_bin: str = "ffmpeg"
    extra_args: list[str] = field(default_factory=list)
    keep_wav: bool = True


# --------------------------------------------------------------------------- #
# 顶层配置
# --------------------------------------------------------------------------- #


@dataclass
class AppConfig:
    """整条流水线的配置。"""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)
    text_rules: TextRulesConfig = field(default_factory=TextRulesConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)

    #: 实际加载的路径（调试用）。
    config_path: Path | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ 加载
    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        *,
        overrides: dict[str, Any] | None = None,
        root: str | Path | None = None,
    ) -> "AppConfig":
        """加载配置。

        Args:
            config_path: YAML 文件路径；``None`` 时按 ``默认值 + overrides`` 构建。
            overrides: 形如 ``{"generation": {"device": "mps"}}`` 的深层覆盖。
            root: 项目根目录；默认取配置文件所在目录，或当前工作目录。

        Returns:
            :class:`AppConfig`，其中所有路径字段已解析为绝对路径。
        """
        payload: dict[str, Any] = {}
        resolved_root: Path

        if config_path is not None:
            path = Path(config_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"找不到配置文件：{path}")
            payload = _load_yaml(path)
            resolved_root = Path(root).expanduser().resolve() if root is not None else path.parent
        else:
            path = None
            resolved_root = Path(root).expanduser().resolve() if root is not None else Path.cwd()

        if overrides:
            payload = deep_merge(payload, overrides)

        config = cls.from_dict(payload)
        config.config_path = path
        config.project.resolve(resolved_root)
        return config

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        """从字典构建（未知键会被忽略并给出警告）。"""
        known = {f.name: f for f in fields(cls) if f.name != "config_path"}
        hints = _resolved_hints(cls)
        kwargs: dict[str, Any] = {}
        for key, value in (payload or {}).items():
            if key not in known:
                logger.warning("配置项 %r 未知，已忽略", key)
                continue
            if value is None:
                continue
            kwargs[key] = _build_section(hints.get(key), value, key)
        return cls(**kwargs)

    # ------------------------------------------------------------------ 导出
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("config_path", None)
        return data

    # -------------------------------------------------------------- 派生属性
    @property
    def dtype_torch(self) -> Any:
        """把配置里的精度字符串翻译成 ``torch.dtype``（延迟导入 torch）。"""
        import torch  # 局部导入：只有真的要跑模型时才需要 torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.generation.dtype]

    def summary(self) -> str:
        """返回适合打印到日志的配置摘要。"""
        gen, audio, seg = self.generation, self.audio, self.segment
        return (
            f"模型={gen.model_path} 设备={gen.device} 精度={gen.dtype} "
            f"批大小={gen.batch_size}\n"
            f"切分: {seg.min_chars}~{seg.max_chars} 字符/段（硬上限 {seg.hard_max_chars}）\n"
            f"音频: {audio.sample_rate} Hz / {audio.channels}ch / 归一={audio.normalize}\n"
            f"输出: {self.project.output_path}"
        )


# --------------------------------------------------------------------------- #
# 内部辅助
# --------------------------------------------------------------------------- #


def _resolved_hints(section_type: Any) -> dict[str, Any]:
    """取出 dataclass 字段的**真实类型**。

    本模块用了 ``from __future__ import annotations``，因此
    ``dataclasses.fields(cls)[i].type`` 拿到的是**字符串**（``"ProjectConfig"``），
    直接拿它做 ``is_dataclass`` 判断会失败，导致嵌套配置保持成 dict。
    这里用 :func:`typing.get_type_hints` 把注解解析回真实对象。
    """
    if not isinstance(section_type, type) or not is_dataclass(section_type):
        return {}
    try:
        import typing

        return typing.get_type_hints(section_type)
    except Exception as exc:  # noqa: BLE001 - 解析失败时退回空表（字段原样透传）
        logger.debug("解析 %s 的类型注解失败：%s", section_type.__name__, exc)
        return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML。

    优先用 PyYAML；缺失时退回一个只支持本配置文件用法的极简解析器，
    这样即使用户环境里没有 PyYAML，流水线也不会直接挂掉。
    """
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("未安装 PyYAML，使用内置极简解析器")
        return _mini_yaml_load(text)

    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射：{path}")
    return data


def _mini_yaml_load(text: str) -> dict[str, Any]:
    """极简 YAML 解析器：支持嵌套映射、块序列、标量、行内列表/映射。

    只覆盖本项目配置文件用到的语法，不做通用实现。支持的形态::

        key: value                  # 标量
        key:                        # 嵌套映射
          sub: 1
        key:                        # 块序列
          - item1
          - item2
        key: [a, b]                 # 行内序列
        key: {a: 1}                 # 行内映射
        "key:with:colon": value     # 带引号的键

    Note:
        你的 conda 环境里没有 PyYAML，而云端环境也可能装不上，所以这个
        解析器是**实际在用**的路径，不是摆设——``config.yaml`` 必须能被它
        完整读出来（见 tests/test_config_utils.py 的回归用例）。
    """
    root: dict[str, Any] = {}
    # 栈元素：(-1, root) 是虚拟根；(容器缩进, 容器) 是各层容器。
    # 注意第二项是**容器自身所在行的缩进**，不是子元素的缩进——用错会让
    # 同级键无法正确出栈（曾经导致块序列挂到了错误的一层上）。
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    # 记录每个映射最近一次「空值」的键：遇到块序列时把列表挂到它下面
    last_key: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        # --- 块序列项 ---
        if line.startswith("- "):
            item_text = line[2:].strip()
            parent = stack[-1][1]
            if not isinstance(parent, list):
                # 形如下面的结构里，"skip_title_patterns:" 这一行先被当成了
                # 一个空映射压进了栈顶；遇到 "- " 时要用**倒数第二层**（也就是
                # 真正拥有这个键的那一层）把占位空映射替换成列表::
                #
                #     parse:                      <- stack[-2]
                #       skip_title_patterns:      <- stack[-1]（占位空映射）
                #         - "pattern"
                if len(stack) < 2:
                    continue
                owner_indent, owner = stack[-2]
                if not isinstance(owner, dict) or last_key is None or last_key not in owner:
                    continue
                new_list: list[Any] = []
                owner[last_key] = new_list
                # 把占位空映射换成列表，缩进沿用原键所在层级
                stack[-1] = (stack[-1][0], new_list)
                parent = new_list
            parent.append(_mini_scalar(item_text))
            continue

        if ":" not in line:
            continue

        key, _, raw_value = _split_key_value(line)
        raw_value = raw_value.strip()

        # 出栈：只要栈顶容器的缩进 >= 当前行缩进，说明当前行不属于它
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            continue

        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            last_key = key
        else:
            parent[key] = _mini_scalar(raw_value)
            last_key = None

    return root


def _split_key_value(line: str) -> tuple[str, str, str]:
    """把 ``key: value`` 拆开，正确处理带引号的键（键里可能含冒号）。"""
    if line.startswith(("'", '"')):
        quote = line[0]
        end = line.find(quote, 1)
        if end != -1:
            key = line[1:end]
            rest = line[end + 1:]
            if rest.startswith(":"):
                return key.strip(), ":", rest[1:]
    key, sep, value = line.partition(":")
    return key.strip().strip("'\""), sep, value


def _mini_scalar(value: str) -> Any:
    """解析一个标量（含行内列表 / 内联映射）。"""
    value = value.split(" #", 1)[0].strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_mini_scalar(item) for item in _split_inline(inner)]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        for item in _split_inline(inner):
            k, _, v = item.partition(":")
            result[k.strip().strip("'\"")] = _mini_scalar(v.strip())
        return result
    if value in {"true", "True", "yes"}:
        return True
    if value in {"false", "False", "no"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _split_inline(text: str) -> list[str]:
    """按逗号切分，但忽略引号内的逗号。"""
    items: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for char in text:
        if quote:
            buf.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            buf.append(char)
        elif char == ",":
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(char)
    if buf:
        items.append("".join(buf).strip())
    return [item for item in items if item]


def _build_section(section_type: Any, value: Any, key: str) -> Any:
    """把 dict 递归构造成嵌套 dataclass；非 dataclass 类型原样返回。"""
    if not isinstance(section_type, type) or not is_dataclass(section_type):
        return value
    if not isinstance(value, dict):
        logger.warning("配置项 %r 期望映射，收到 %s，已忽略", key, type(value).__name__)
        return section_type()
    hints = _resolved_hints(section_type)
    kwargs: dict[str, Any] = {}
    for f in fields(section_type):
        if f.name not in value:
            continue
        field_value = value[f.name]
        field_type = hints.get(f.name)
        if field_value is not None and isinstance(field_type, type) and is_dataclass(field_type):
            kwargs[f.name] = _build_section(
                field_type, field_value, f"{key}.{f.name}"
            )
        else:
            kwargs[f.name] = field_value
    return section_type(**kwargs)
