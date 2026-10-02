# 接口规范（API Reference）

> **用途**：本项目所有可调用接口的完整规范。为开发可视化界面（GUI / Web）
> 提供准确的数据契约 —— 参数名、类型、默认值、返回值、异常、线程安全说明。
>
> **所有签名均从代码自动导出核对**，不是凭记忆写的。
>
> 相关文档：[overview.md](overview.md)（完整工作流）·
> [user-guide.md](user-guide.md)（命令行用法）

---

## 目录

1. [设计约定](#1-设计约定)
2. [快速上手（三种集成方式）](#2-快速上手三种集成方式)
3. [配置系统](#3-配置系统)
4. [数据模型](#4-数据模型)
5. [阶段一：解析](#5-阶段一解析)
6. [阶段二：洗稿](#6-阶段二洗稿)
7. [阶段三：切分](#7-阶段三切分)
8. [阶段四：合成](#8-阶段四合成)
9. [阶段五：拼接](#9-阶段五拼接)
10. [阶段六：压缩](#10-阶段六压缩)
11. [硬件加速](#11-硬件加速)
12. [音频工具](#12-音频工具)
13. [命令行接口](#13-命令行接口)
14. [GUI 开发建议](#14-gui-开发建议)
15. [异常与错误处理](#15-异常与错误处理)

---

## 1. 设计约定

开发 GUI 前请先理解这几条，它们决定了接口为什么长这样。

### 1.1 惰性导入（懒加载）

`noveltts` 包**只依赖标准库**。`numpy` / `soundfile` / `torch` / `qwen_tts`
都在**用到时才 import**。

**对 GUI 的意义**：

* 启动 GUI 时 import `noveltts` 不会加载 torch（省几秒 + 几百 MB 内存）
* 可以安全地在 GUI 主进程里做解析/切分，**不需要**先装 torch
* 想做"依赖检查"界面，用 `noveltts info` 的 JSON 输出，不要直接 import torch

```python
import noveltts                      # 快，不加载 torch
from noveltts.config import AppConfig # 快
from noveltts.synthesis import SynthesisSession  # 仍不加载 torch
session = SynthesisSession(cfg)      # 这里才可能加载
```

### 1.2 设备与硬件自适应

所有硬件相关决策由 `noveltts.accelerator` 集中处理，**接口层不用关心**。
配置里写 `"auto"` 即可（`device` / `batch_size` / `tf32` / `cudnn_benchmark` /
`attn_implementation` 都支持）。

### 1.3 失败隔离与断点续跑

* 每个片段生成后**立刻落盘**，并写入 `manifest.json`
* 单段失败不影响其它段（批量失败会降级为逐段重试）
* 重跑同一条命令会**跳过已完成**的片段（`skip_existing: true`）

**对 GUI 的意义**：可以放心做"暂停/继续"，无需自己维护状态。
`SynthesisReport.skipped` 告诉你跳过了多少。

### 1.4 显式关闭

`SynthesisSession` 持有引擎（可能占用数 GB 显存），**用完必须 `close()`**，
或用上下文管理器：

```python
with SynthesisSession(config) as session:     # 推荐
    report = session.synthesize_chapter(chapter)
# 自动 close，显存释放
```

> 会话只会关闭**它自己创建**的引擎。若你传入外部引擎（`engine=`），
> 它不会替你关 —— 便于复用同一个引擎跑多章。

### 1.5 返回值都是 dataclass

所有返回都是 `@dataclass`，可直接 `dataclasses.asdict()` 转 JSON 给前端。

### 1.6 单位约定

| 量 | 单位 | 说明 |
|---|---|---|
| 时长 | **秒**（float） | `duration` / `audio_seconds` / `elapsed_seconds` |
| 响度 | **dBFS**（float，负值） | `target_dbfs = -1.5` 表示峰值 -1.5 dBFS |
| 采样率 | **Hz**（int） | 本项目固定 24000 |
| 显存 | **GB**（float） | `total_vram_gb` |
| 音频样本 | **numpy float32 一维数组** | 范围 [-1, 1] |

---

## 2. 快速上手（三种集成方式）

### 方式 A：子进程调用 CLI（**最简单，推荐 GUI 首选**）

优点：完全隔离，崩溃不影响 GUI；能拿到结构化 JSON；不用管线程安全。

```python
import json, subprocess

def run_cli(args: list[str], cwd: str) -> dict:
    """调用 noveltts CLI 并解析 JSON 输出。"""
    proc = subprocess.run(
        ["noveltts", *args, "--json"],     # 支持 --json 的子命令见 §13
        cwd=cwd, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "命令失败")
    return json.loads(proc.stdout)

info = run_cli(["info"], cwd="/path/to/NovelTTS")
chapters = run_cli(["parse", "--list"], cwd="/path/to/NovelTTS")
```

**实时进度**：读 `stdout` 逐行解析日志（格式见 §13.4）。

```python
proc = subprocess.Popen(
    ["noveltts", "synth", "--chapter", "c003", "--limit", "5"],
    cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
for line in proc.stdout:                  # 逐行拿到进度
    if "完成 " in line or "%" in line:
        update_progress_bar(line)
```

### 方式 B：直接调用 Python API（**功能最全**）

```python
from pathlib import Path
from noveltts.config import AppConfig
from noveltts.epub_parser import parse_book
from noveltts.segmenter import TextSegmenter
from noveltts.synthesis import SynthesisSession
from noveltts.concat import concat_from_manifest
from noveltts.encode import encode_audio

config = AppConfig.load("config.yaml")        # 或 from_dict({...})
config.project.ensure_dirs()

book = parse_book("sources/book.epub", config.parse)
chapter = book.chapters[2]

segmenter = TextSegmenter(config.segment, language=book.language)
segments = segmenter.segment_chapter(chapter)

with SynthesisSession(config) as session:
    report = session.synthesize_segments(
        segments, chapter_id=chapter.chapter_id, title=chapter.title
    )
    print(f"成功 {report.generated}，跳过 {report.skipped}，失败 {report.failed}")
    # 进度回调（可选）
    # 见 §8.2 progress_callback 参数

concat_from_manifest(config, chapter.chapter_id)
```

### 方式 C：混合（**大任务推荐**）

* 解析 / 洗稿 / 切分 → **方式 B**（快，纯 CPU，便于实时预览）
* 合成（长时间、占显存）→ **方式 A**（子进程隔离，可随时 kill）

```python
# GUI 里做"预览"用 B，"生成"用 A 起子进程，这样用户能随时取消
```

---

## 3. 配置系统

### 3.1 `AppConfig`

顶层配置容器，聚合 8 个子配置。

```python
from noveltts.config import AppConfig

# ① 从 YAML 加载
config: AppConfig = AppConfig.load(
    config_path="config.yaml",   # str | Path | None，None = 用内置默认值
    overrides={                  # dict | None，深层覆盖（命令行参数走这里）
        "generation": {"device": "cuda", "batch_size": 8},
        "audio": {"speech_rate": 1.15},
    },
    root="/path/to/project",     # str | Path | None，项目根目录（解析相对路径）
)

# ② 从字典构造（GUI 传参最方便）
config = AppConfig.from_dict({
    "project": {"root": "/path/to/NovelTTS", "output_dir": "/data/audio"},
    "generation": {"device": "cuda", "batch_size": "auto"},
    "audio": {"speech_rate": 1.2},
    "voice": {"ref_audio": "assets/ref/doubao.wav", "ref_text": "..."},
})

config.project.ensure_dirs()     # 建好 output/work 等目录
```

**注意**：`from_dict` **不会**自动 resolve 相对路径。要调用
`config.project.resolve(root_path)` 或 `AppConfig.load(..., root=...)`。

### 3.2 配置优先级（**GUI 必须实现这一层**）

```
内置默认值  <  config.yaml  <  config.local.yaml  <  GUI 传入的 overrides
```

对应到 Python：

```python
base = AppConfig.load("config.yaml", root=root)
local = AppConfig.load("config.local.yaml", root=root)   # 若存在
merged = overlay(base, local)                            # 本机覆盖
final = apply_overrides(merged, gui_overrides)           # GUI 参数最高
```

> ⚠️ **顺序不能颠倒**。曾经的 bug：本机配置在命令行之后应用，
> 导致 `--rate` 完全失效（`config.local.yaml` 里的 `speech_rate: 1.0` 把它盖了）。

**`config.local.yaml` 是机器相关设置的归属地**（模型绝对路径、输出目录、
设备），它**不入库**，因此代码更新不会冲掉它。

### 3.3 `ProjectConfig`

```python
@dataclass
class ProjectConfig:
    root: str = ""                   # 项目根目录（绝对路径）
    source_dir: str = "sources"      # 小说放这里
    work_dir: str = "output/work"    # 中间产物
    output_dir: str = "output/audio" # 音频产物
    cache_dir: str = "output/cache"  # 缓存
```

**关键：相对 vs 绝对**

| 写法 | 行为 |
|---|---|
| 相对路径 `output/audio` | 相对 `root` 解析 |
| **绝对路径** `/data/audio` | **原样采用**（云端放数据盘就用这个） |

**派生属性**（只读，由 `resolve()` 计算）：

| 属性 | 类型 | 说明 |
|---|---|---|
| `root_path` | `Path` | 解析后的根目录 |
| `source_path` | `Path` | 解析后的素材目录 |
| `work_path` | `Path` | 解析后的工作目录 |
| `output_path` | `Path` | 解析后的输出目录 |
| `cache_path` | `Path` | 解析后的缓存目录 |

```python
config.project.resolve(Path("/path/to/NovelTTS"))   # 计算派生属性
config.project.ensure_dirs()                        # 建目录
```

> **改配置后必须重新 `resolve()`**。曾经的 bug：只改了 `output_dir`
> 但没重算，`output_path` 仍是旧值，产物照样写回系统盘。

### 3.4 全部子配置字段与默认值

以下为**自动导出**的真实默认值。

#### `ParseConfig`（解析）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `min_chapter_chars` | int | `200` | 小于此长度的"章节"被并入前后章 |
| `skip_title_patterns` | list[str] | 见下 | 匹配到的标题整章丢弃（版权页等） |
| `merge_short_into_previous` | bool | `False` | 短章并入前一章 |
| `drop_reference_sections` | bool | `True` | 丢弃"参考文献/目录"类章节 |
| `default_language` | str | `"en"` | 未识别出语言时的默认值 |
| `include_nav_titles` | bool | `True` | 使用 EPUB 目录标题（更准） |
| `exclude_paragraph_patterns` | list[str] | 见下 | 匹配到的**段落**被丢弃 |

`skip_title_patterns` 默认：

```python
['^table of contents', '^contents$', '^color gallery', '^title page$',
 '^copyright', '^credits', '^newsletter', '^also available',
 '^about the (author|series)', '^cover']
```

`exclude_paragraph_patterns` 默认（7 条，保守设计，只删明确无朗读价值的）：

```python
['\\bAll rights reserved\\b',
 '\\bISBN[\\s:：]*[\\d\\-Xx]{10,17}',
 '\\bReproduction in whole or in part\\b',
 '\\bwithout (?:the )?(?:prior )?written permission\\b',
 '^Copyright\\s*(?:\\u00a9|\\(c\\))?\\s*\\d{4}',
 '\\bThis (?:book|work) is a work of fiction\\b',
 '\\bPrinted in\\b[^.]{0,40}\\b(?:China|USA|United States|Japan)\\b']
```

#### `TextRulesConfig`（洗稿规则）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `True` | 总开关 |
| `language` | str | `"auto"` | `auto`/`en`/`zh` |
| `protect_patterns` | list[str] | URL/邮箱 | 匹配部分**不做任何转换** |
| `abbreviations` | dict | `{}` | 自定义缩写展开 |
| `acronym_spellout` | list[str] | 见下 | 需要逐字母读的模式 |
| `large_numbers` | int | `1100` | ≥ 此值的大数转口语（设很大 = 关闭） |
| `year_range` | bool | `True` | `1990-1995` → `1990 to 1995` |
| `ordinal_numbers` | bool | `True` | `1st` → `first` |
| `number_followed_by_noun` | bool | `True` | `3 apples` → `three apples` |
| `number_ranges` | bool | `True` | `870-120` → `... to ...` |
| `roman_numerals` | bool | `True` | `Chapter VII` → `Chapter seven` |
| `dashes` | bool | `True` | 破折号按上下文处理 |
| `quotes` | str | `"straight"` | 弯引号转直引号（TTS 更稳） |
| `ellipsis` | bool | `True` | `…` → `...` |
| `unicode_normalize` | str | `"NFC"` | Unicode 规范化形式 |
| `collapse_whitespace` | bool | `True` | 合并多余空白 |
| `strip_markup` | bool | `True` | 去 HTML 标签 |
| `cjk_numbers` | bool | `True` | 中文数字处理 |
| `cjk_measure_pairs` | dict | 21 项 | 量词配对（`个`/`只`/`岁`…） |
| `custom_replacements` | dict | `{}` | **最高优先级**的正则替换 |
| `numeric_context_abbreviations` | dict | `{'No.': 'Number', 'no.': 'number'}` | 仅数字上下文生效 |

#### `SegmentConfig`（切分）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_chars` | int | `120` | 目标上限，调小更稳但接缝更多 |
| `min_chars` | int | `24` | 低于此值的碎片会被合并 |
| `hard_max_chars` | int | `200` | 硬上限，**绝不**超过 |
| `merge_short` | bool | `True` | 自动合并过短片段 |
| `keep_dialogue_separate` | bool | `True` | 对话不与叙述混段 |
| `split_on_paragraph` | bool | `False` | 按段落边界强制切 |
| `paragraph_break_chars` | int | `90` | 段落过长时的切分阈值 |
| `tail_clause_min_chars` | int | `12` | 尾部从句最小长度 |

#### `VoiceConfig`（音色）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ref_audio` | str | `""` | 参考音频路径（**必需**，相对项目根或绝对） |
| `ref_text` | str | `""` | 参考音频的**逐字**文本 |
| `x_vector_only_mode` | bool | `False` | `True` = 只用音色向量（不推荐） |
| `mode` | str | `"icl"` | `icl`（有 ref_text）/ `xvec`（无） |
| `prompt_cache` | bool | `True` | 缓存音色 prompt，多章复用省时间 |

> **`mode` 是派生值**，由 `ref_text` 是否为空决定。GUI 应显示当前实际模式：
> `icl` 音色相似度更好；`xvec` 会打折。

#### `GenerationConfig`（生成）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `device` | str | `"auto"` | `auto`/`cuda`/`cuda:1`/`mps`/`cpu` |
| `dtype` | str | `"bfloat16"` | `bfloat16` 最省内存 |
| `attn_implementation` | str | `"auto"` | `auto`/`flash_attention_2`/`sdpa`/`eager` |
| `max_new_tokens` | int | `1024` | 单次生成上限；显存紧就降到 512 |
| `batch_size` | int \| `"auto"` | `1` | `auto` 按显存自推 |
| `tf32` | bool \| `"auto"` | `"auto"` | 算力 ≥ 8.0 时开启 |
| `cudnn_benchmark` | bool \| `"auto"` | `"auto"` | CUDA 上开启 |
| `clear_cache_every` | int | `0` | `0` = auto（CUDA/MPS 每 50 段） |
| `oom_batch_downgrade` | bool | `True` | OOM 时自动把批大小减半 |
| `non_streaming_mode` | bool | `True` | 一次性看全文，质量略好 |
| `num_retries` | int | `2` | 单段失败重试次数 |
| `retry_backoff` | float | `1.5` | 重试退避倍数 |
| `skip_existing` | bool | `True` | **断点续跑的关键** |
| `keep_last_segments` | int | `0` | 只保留最后 N 段（调试用） |
| `progress_every` | int | `10` | 每 N 段打印一次进度 |
| `temperature` | float \| None | `None` | `None` = 用模型默认 |
| `top_k` | int \| None | `None` | |
| `top_p` | float \| None | `None` | |
| `repetition_penalty` | float \| None | `None` | |
| `language` | str | `"English"` | 传给模型的语种名 |

#### `AudioConfig`（音频处理）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `sample_rate` | int | `24000` | Qwen3-TTS 输出固定 24k |
| `channels` | int | `1` | 单声道 |
| `normalize` | str | `"peak"` | `peak`/`rms`/`none` |
| `target_dbfs` | float | `-1.5` | 峰值目标 |
| `target_rms_dbfs` | float | `-20.0` | RMS 目标（`normalize="rms"` 时） |
| `max_gain_db` | float | `12.0` | 增益上限，防止把噪声放大 |
| `trim_silence` | bool | `False` | 首尾静音裁切 |
| `silence_threshold_dbfs` | float | `-50.0` | 静音判定阈值 |
| `min_gap_seconds` | float | `0.32` | 片段间插入的停顿 |
| `tail_quiet_gain` | float | `0.35` | 片段尾部淡出增益 |
| `tail_quiet_threshold_dbfs` | float | `-38.0` | 尾部淡出阈值 |
| `join_fade_ms` | float | `15.0` | 拼接处淡入淡出，消咔嗒声 |
| `save_segments` | bool | `True` | 保留逐段 wav |
| `speech_rate` | float | `1.0` | **语速**：1.2 = 快 20% |

#### `EncodeConfig`（压缩）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `True` | 总开关 |
| `format` | str | `"m4a"` | `m4a`/`mp3`/`opus`/`flac`/`ogg` |
| `bitrate` | str | `"64k"` | 实测 68kbps 足够（语音） |
| `sample_rate` | int | `24000` | |
| `channels` | int | `1` | |
| `ffmpeg_bin` | str | `"ffmpeg"` | 可执行名或绝对路径 |
| `extra_args` | list[str] | `[]` | 追加的 ffmpeg 参数 |
| `keep_wav` | bool | `True` | 压缩后保留无损 wav |

---

## 4. 数据模型

全部是 `@dataclass`，可用 `dataclasses.asdict()` 直接序列化给前端。

### `Book`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `title` | str | **必需** | 书名 |
| `source_path` | Path | **必需** | 源文件 |
| `language` | str | `"en"` | 语言代码 |
| `author` | str | `""` | 作者 |
| `chapters` | list[Chapter] | `[]` | 章节列表 |

### `Chapter`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `index` | int | **必需** | 从 0 开始 |
| `title` | str | **必需** | 章节标题 |
| `paragraphs` | list[Paragraph] | `[]` | 段落 |
| `chapter_id` | str | `""` | **`c001` 形式，UI 主键** |
| `source` | str | `""` | 来源标识 |
| `language` | str | `"en"` | |
| `segments` | list[Segment] \| None | `None` | 惰性填充 |

### `Paragraph`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `text` | str | **必需** | |
| `kind` | str | `"narration"` | `narration` / `dialogue` / `heading` |
| `source` | str | `""` | |

### `Segment`（**GUI 列表的主数据**）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `segment_id` | str | **必需** | `c003-0007` |
| `chapter_id` | str | **必需** | `c003` |
| `index` | int | **必需** | 从 0 开始 |
| `text` | str | **必需** | **送给模型的文本（已洗稿）** |
| `kind` | str | `"narration"` | `narration` / `dialogue` |
| `char_count` | int | `0` | 字符数 |
| `paragraph_index` | int | `0` | 来源段落 |
| `paragraph_span` | int | `1` | 跨几个段落 |
| `unit_count` | int | `1` | 由几个句子合并 |
| `source_text` | str | `""` | 洗稿**前**的原文（做对比用） |

### `SegmentRecord`（单段生成结果）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `segment_id` | str | **必需** | |
| `audio_path` | str | **必需** | 相对项目根，或绝对路径 |
| `duration` | float | `0.0` | 秒 |
| `sample_rate` | int | `0` | |
| `attempts` | int | `1` | 尝试次数 |
| `skipped` | bool | `False` | 是否跳过（已存在） |
| `error` | str \| None | `None` | **非 None 表示该段失败** |

### `ChapterAudioManifest`

每章一个，位于 `output/audio/segments/<chapter_id>/manifest.json`。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `chapter_id` | str | **必需** | |
| `title` | str | **必需** | |
| `text_sha1` | str | **必需** | **文本指纹，用于检测文本变更** |
| `created_at` | str | **必需** | ISO 时间 |
| `updated_at` | str | **必需** | |
| `sample_rate` | int | **必需** | |
| `segment_count` | int | **必需** | |
| `records` | list[dict] | `[]` | 各段记录 |
| `settings` | dict | `{}` | **生成参数快照（含 `speech_rate`）** |

> `text_sha1` 的用途：如果文本变了（改了洗稿规则），旧音频就失效了。
> GUI 可以据此提示"文本已变更，建议重新生成"。

### `SynthesisReport`（**进度与结果**）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `chapter_id` | str | **必需** | |
| `title` | str | `""` | |
| `total` | int | `0` | 总段数 |
| `generated` | int | `0` | 新生成 |
| `skipped` | int | `0` | 跳过（已存在） |
| `failed` | int | `0` | 失败 |
| `audio_seconds` | float | `0.0` | 音频总时长（秒） |
| `elapsed_seconds` | float | `0.0` | 耗时（秒） |
| `errors` | list[tuple[str, str]] | `[]` | `(segment_id, 错误信息)` |

**方法与属性**（已核对）：

| 成员 | 类型 | 说明 |
|---|---|---|
| `ok` | property → `bool` | `failed == 0` |
| `summary()` | `-> str` | 人类可读一行摘要 |
| `as_dict()` | `-> dict` | **含派生字段 `rtf`**，可直接转 JSON 给前端 |

```python
report.ok            # False
report.summary()     # "c003 T | 新生成 7 | 跳过 2 | 失败 1 | 音频 01:40 | 耗时 00:50 | RTF 0.50"
report.as_dict()
# {'chapter_id': 'c003', 'title': 'T', 'total': 10, 'generated': 7, 'skipped': 2,
#  'failed': 1, 'audio_seconds': 100.0, 'elapsed_seconds': 50.0, 'rtf': 0.5,
#  'errors': [{'segment_id': 'c003-0003', 'error': 'boom'}]}
```

> 进度百分比 = `(generated + skipped + failed) / total`，自己算即可
> （没有专门的 `progress()` 方法）。

### `ConcatResult`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `output_path` | Path | **必需** | 成品文件 |
| `chapter_id` | str | **必需** | |
| `segment_count` | int | **必需** | |
| `duration` | float | **必需** | 秒 |
| `size_bytes` | int | **必需** | |
| `elapsed_seconds` | float | `0.0` | |
| `missing` | list[str] | `[]` | **缺失的片段 ID（非空要警告）** |
| `sample_rate` | int | `24000` | |

**方法**：`summary() -> str` → `"c003 | 44 段 | 时长 03:13 | 体积 8.6 MB | 缺失 1 段"`

### `EncodeResult`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `source` | Path | **必需** | 输入 wav |
| `output` | Path | **必需** | 输出压缩文件 |
| `source_bytes` | int | **必需** | |
| `output_bytes` | int | **必需** | |
| `duration` | float | `0.0` | |
| `elapsed_seconds` | float | `0.0` | |
| `encoder` | str | `"ffmpeg"` | 实际使用的编码器 |

**方法与属性**（已核对）：

| 成员 | 类型 | 说明 |
|---|---|---|
| `ratio` | property → `float` | `output_bytes / source_bytes`，约 0.18 |
| `saved_bytes` | property → `int` | 省下多少字节 |
| `summary()` | `-> str` | `"c003.m4a | 8.6 MB -> 1.5 MB（18%，省 7.1 MB）"` |

```python
result.ratio          # 0.1777...
result.saved_bytes    # 7400000
result.summary()      # "c003.m4a | 8.6 MB -> 1.5 MB（18%，省 7.1 MB）"
```

> `saved_bytes` 是 **property 不是方法** —— 写 `result.saved_bytes` 而非 `result.saved_bytes()`。

### `DeviceInfo` / `AccelProfile`

见 §11。

### `SynthesisResult`（引擎返回）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `waveforms` | list[ndarray] | `[]` | **每个输入文本一个波形** |
| `sample_rate` | int | `24000` | |
| `prompt_seconds` | float | `0.0` | 音色 prompt 构建耗时 |
| `generate_seconds` | float | `0.0` | 生成耗时 |
| `texts` | list[str] | `[]` | 回显输入 |

### `AudioInfo`

| 字段 | 类型 | 说明 |
|---|---|---|
| `path` | Path | |
| `sample_rate` | int | |
| `channels` | int | |
| `frames` | int | 采样点数 |
| `duration` | float | 秒 |
| `subtype` | str | 如 `PCM_16` |

---

## 5. 阶段一：解析

### `find_source_files`

```python
def find_source_files(source_dir: str | Path) -> list[Path]
```

列出素材目录里所有可解析文件（`.epub` / `.txt` / `.md`），已排序。

**异常**：目录不存在时抛 `FileNotFoundError`。

### `parse_book`

```python
def parse_book(
    path: str | Path,
    config: ParseConfig | None = None,
    *,
    heading_re: str | None = None,      # 仅 TXT 有意义：自定义标题正则
) -> Book
```

**统一入口**，按扩展名自动分派到 `parse_epub` / `parse_txt`。

```python
book = parse_book("sources/book.epub", config.parse)
for ch in book.chapters:
    print(ch.chapter_id, ch.title, len(ch.paragraphs))
```

**异常**：

| 异常 | 触发条件 |
|---|---|
| `FileNotFoundError` | 文件不存在 |
| `ValueError` | 扩展名不支持 / 解析后无章节 |

### `parse_epub`

```python
def parse_epub(path: str | Path, config: ParseConfig | None = None) -> Book
```

**推荐格式**。用 EPUB 的目录（nav/ncx）确定章节边界与标题 —— 比 TXT 靠正则猜标题可靠得多。

### `parse_txt`

```python
def parse_txt(
    path: str | Path,
    config: ParseConfig | None = None,
    *,
    heading_re: str | None = None,
) -> Book
```

按 `heading_re`（默认识别 `Chapter N` / `第N章` / 全大写短行）切章。

---

## 6. 阶段二：洗稿

> **两个易错点**（都已实测确认）：
> 1. `TextNormalizer.normalize()` 返回 **`NormalizedText`**，不是 `str` —— 要取 `.text`
> 2. `TextNormalizer` **没有** `diff()` / `stats()` 方法（曾误写入文档，已删）

### `build_normalizer`

```python
def build_normalizer(
    rules: TextRulesConfig | None = None,
    *,
    language: str = "en",
    use_default_abbreviations: bool = True,
) -> TextNormalizer
```

构建洗稿器。**建议复用同一个实例**（构建有开销）。

> ⚠️ **`normalize()` 返回 `NormalizedText` 而不是 `str`。**
> 要拿字符串得取 `.text`。这是容易写错的地方。

```python
normalizer = build_normalizer(config.text_rules, language="en")

result = normalizer.normalize("She ran the 1500 in 4:05, Mr. Tanaka said.")
result.text          # 'She ran the fifteen hundred in 4:05, Mister Tanaka said.'
result.report        # NormalizeReport
result.report.changed  # True
result.report.rules    # {'large_numbers': RuleStat(...), ...}

# 批量：一次拿全部结果与汇总报告
texts, report = normalizer.normalize_many(["Vol. 8", "1st place"])
# texts -> ['Volume eight', 'first place']
# report.rules -> 各类改写的统计
```

### `normalize_text`（便捷函数）

```python
def normalize_text(
    text: str,
    rules: TextRulesConfig | None = None,
    *,
    language: str = "en",
) -> str
```

一次性调用。**注意**：内部每次都重建 normalizer，批量处理请用 `build_normalizer`。

```python
from noveltts.normalize import normalize_text
print(normalize_text("Vol. 8, 1st place, $1,500.", language="en"))
# "Volume 8, first place, one thousand five hundred dollars."
#  注意：8 小于 large_numbers 阈值(1100)，所以保持数字形态；
#        1500 超过阈值，转成口语化的 "one thousand five hundred"
```

### `TextNormalizer` 实例方法（**已核对，只有这两个**）

| 方法 | 签名 | 说明 |
|---|---|---|
| `normalize` | `(text: str, *, report: NormalizeReport \| None = None) -> NormalizedText` | 主入口 |
| `normalize_many` | `(texts: Iterable[str]) -> tuple[list[str], NormalizeReport]` | 批量，返回纯文本列表 + 汇总报告 |

**没有 `diff()` / `stats()` 方法** —— 想做"改了什么"的对比界面，
用 `Segment.source_text`（洗稿前）与 `Segment.text`（洗稿后）逐段对比，
或看 `NormalizeReport.rules` 里各条规则的命中次数。

```python
from dataclasses import fields
result = normalizer.normalize("Vol. 8 and 1st place")
print(result.report.changed)          # True
for name, stat in result.report.rules.items():
    print(f"  {name}: {stat.count} 处")   # 各规则命中次数
```

### 数据结构 `NormalizedText` / `NormalizeReport`

| 类型 | 字段 | 说明 |
|---|---|---|
| `NormalizedText` | `text: str` | 洗稿后的文本 |
| | `report: NormalizeReport` | 本次改写的报告 |
| `NormalizeReport` | `rules: dict[str, RuleStat]` | 各规则命中统计 |
| | `changed: bool` | 是否有改动 |

---

## 7. 阶段三：切分

### `TextSegmenter`

```python
class TextSegmenter:
    def __init__(
        self,
        config: SegmentConfig | None = None,
        *,
        normalizer: TextNormalizer | None = None,   # 传入则自动洗稿
        language: str = "en",
    ) -> None
```

**复用实例**（无状态，可跨章复用）：

```python
segmenter = TextSegmenter(config.segment, language=book.language)
```

#### `segment_chapter`

```python
def segment_chapter(
    self,
    chapter: Chapter,
    *,
    normalize: bool = True,
) -> list[Segment]
```

**主接口**。返回的 `Segment.segment_id` 形如 `c003-0007`。

#### `segment_text`

```python
def segment_text(
    self,
    text: str,
    *,
    chapter_id: str = "adhoc",
    normalize: bool = True,
    title: str = "",
) -> list[Segment]
```

给"粘贴一段文本试听"用（GUI 的试用功能）。

### 模块级便捷函数

```python
def segment_chapter(chapter, config=None, *, language="en", normalize=True) -> list[Segment]
def segment_text(text, config=None, *, language="en", chapter_id="adhoc", normalize=True) -> list[Segment]
def summarize_segments(segments: list[Segment]) -> str
```

> 模块级函数每次重建 segmenter。**批量处理请用类**。

### `summarize_segments`

返回人类可读摘要，适合显示在 GUI 状态栏：

```python
"3220 段 | 中位 53 字符 | 最长 199 | 总 192,045 字符 | 预估 3:48:54"
```

---

## 8. 阶段四：合成

### `build_engine`

```python
def build_engine(
    generation: GenerationConfig,
    voice: VoiceConfig,
    *,
    device: str | None = None,
    dry_run: bool = False,          # True = 假引擎，不加载模型
    **fake_kwargs: Any,             # 传给假引擎（测试用）
) -> SynthesisEngine
```

**一般不需要直接调用** —— 交给 `SynthesisSession` 管理。

想要"模型常驻、多章复用"时才有用：

```python
engine = build_engine(config.generation, config.voice)
try:
    for chapter in book.chapters:
        with SynthesisSession(config, engine=engine) as session:
            session.synthesize_chapter(chapter)
finally:
    engine.close()          # 你自己创建的，自己关
```

### `SynthesisSession`

```python
class SynthesisSession:
    def __init__(
        self,
        config: AppConfig,
        *,
        engine: SynthesisEngine | None = None,   # None = 自己建（并负责关）
        dry_run: bool = False,
        device: str | None = None,
    ) -> None
```

**上下文管理器，用完自动 `close()`**：

```python
with SynthesisSession(config) as session:
    report = session.synthesize_chapter(chapter)
```

#### `synthesize_segments`（**底层主接口**）

```python
def synthesize_segments(
    self,
    segments: list[Segment],
    *,
    chapter_id: str = "",
    title: str = "",
    force: bool = False,              # True = 已存在也重做
    language: str | None = None,       # None = 用配置
    batch_size: int | None = None,     # None = 按硬件自推
    write_manifest: bool = True,
) -> SynthesisReport
```

> **`chapter_id` 必须传对**，它决定音频写哪个目录、manifest 写哪里。

#### `synthesize_chapter`（**上层便捷接口**）

```python
def synthesize_chapter(
    self,
    chapter: Chapter,
    *,
    force: bool = False,
    language: str | None = None,
    include_title: bool = True,        # 把章节标题作为第一段一起生成
    limit: int = 0,                    # >0 = 只做前 N 段（试听）
    only_segments: list[int] | None = None,   # 只做指定序号（0 基）
) -> SynthesisReport
```

#### ⚠️ 前置条件：章节**必须已经切分**

`chapter.segments` 为空时抛 `ValueError("章节 cXXX 还没有切分，请先调用 segment_chapter()")`。

`parse_book` 只解析出段落（`chapter.paragraphs`），**不会**自动切分。
所以正确顺序是：

```python
from noveltts.segmenter import TextSegmenter

# ① 先切分
segmenter = TextSegmenter(config.segment, language=book.language)
chapter.segments = segmenter.segment_chapter(chapter)   # ← 关键

# ② 再合成
with SynthesisSession(config) as session:
    report = session.synthesize_chapter(chapter, limit=5)
```

> `synth` / `run` 命令行命令**会**自动完成切分（内部先调 segmenter），
> 但直接调 Python API 时要自己来这一步。

#### 参数语义

| 参数 | 说明 |
|---|---|
| `include_title=True` | 章节标题也读出来（作为一个 `heading` 片段） |
| `limit=N` | 只做前 N 段（试听） |
| `only_segments=[3,5,8]` | 只做指定序号（**1-based**，对应 `--segments`） |

#### `synthesize_text`（试用功能）

```python
def synthesize_text(
    self,
    text: str,
    *,
    chapter_id: str = "adhoc",
    title: str = "",
    language: str | None = None,
) -> list[SegmentRecord]
```

给"粘贴一段文字，立刻听效果"（GUI 的试听框）。

#### `synthesize_segment`

```python
def synthesize_segment(
    self,
    segment: Segment,
    *,
    force: bool = False,
    language: str | None = None,
) -> SegmentRecord
```

单段生成（重做某一段用）。

#### `retry_failed`

```python
def retry_failed(
    self,
    chapter_id: str,
    *,
    language: str | None = None,
) -> SynthesisReport
```

**只重跑 manifest 里标记失败的片段**。跑完长任务后的收尾按钮。

```python
with SynthesisSession(config) as session:
    report = session.retry_failed("c003")
    print(f"补回 {report.generated} 段，仍失败 {report.failed}")
```

#### `close`

```python
def close(self) -> None
```

释放引擎与显存。**只关自己创建的引擎**（`engine=None` 时）。
重复调用安全。

### `dump_segments_index`

```python
def dump_segments_index(
    segments: list[Segment],
    *,
    config: AppConfig | None = None,
    work_path: str | Path | None = None,
) -> Path
```

把片段索引写盘（`concat` 阶段要用）。返回写入路径。

### 自定义引擎（**GUI 想接别的 TTS 时用**）

实现这个协议即可替换后端：

```python
class SynthesisEngine(Protocol):
    sample_rate: int

    def describe(self) -> str:
        """一行描述，写进日志与引擎信息。"""

    def synthesize(
        self,
        texts: list[str],
        *,
        language: str | None = None,
    ) -> SynthesisResult:
        """批量合成。返回的 waveforms 长度**必须**等于 texts 长度。"""

    def close(self) -> None:
        """释放资源。"""
```

> ⚠️ **`waveforms` 长度必须与 `texts` 一致**。
> 框架会核对，不一致就整批判失败 —— 因为少返回一段会导致
> "把第 2 段的音频写进第 1 段的文件名"这种静默错误。

### 环境变量

| 变量 | 作用 |
|---|---|
| `NUMBA_CACHE_DIR` | numba 缓存目录。**若 librosa 所在目录不可写，必须设置**，否则 `import qwen_tts` 直接失败。本项目已在 `noveltts/__init__.py` 自动兜底 |

---

## 9. 阶段五：拼接

### `concat_from_manifest`（**推荐**）

```python
def concat_from_manifest(
    config: AppConfig,
    chapter_id: str,
    *,
    output_path: str | Path | None = None,   # None = output/<chapter_id>.wav
    gap_seconds: float | None = None,        # None = 用配置
) -> ConcatResult
```

从 manifest 拼一章。**流式处理，内存占用与音频长度无关**
（实测 15.5 分钟音频峰值仅 12 MB）。

```python
result = concat_from_manifest(config, "c003")
if result.missing:
    print(f"⚠️ 缺 {len(result.missing)} 段：{result.missing[:5]}")
```

### `concat_chapter`（**底层**）

```python
def concat_chapter(
    config: AppConfig,
    segments: list[Segment],
    *,
    segment_paths: list[Path] | None = None,
    output_path: str | Path | None = None,
    gap_seconds: float | None = None,
    lead_silence_seconds: float = 0.0,
    tail_silence_seconds: float = 0.6,
    chapter_id: str = "",
    title: str = "",
) -> ConcatResult
```

已有 `segments` 与显式路径时用（比如 GUI 里用户手动调整了顺序）。

### `concat_all_chapters`

```python
def concat_all_chapters(
    config: AppConfig,
    chapter_ids: list[str] | None = None,     # None = 自动发现
    *,
    merge_into_single: bool = False,          # True = 再合并成整本
    output_name: str = "full_book",
) -> list[ConcatResult]
```

**整本合并**就是这个函数（`merge_into_single=True`）。

### `discover_chapters`

```python
def discover_chapters(config: AppConfig) -> list[str]
```

扫描输出目录，返回已有音频的章节 ID（已排序）。GUI 启动时填充章节列表用。

---

## 10. 阶段六：压缩

### `encode_audio`

```python
def encode_audio(
    source: str | Path,
    output: str | Path | None = None,        # None = 据 format 自动定扩展名
    *,
    config: AppConfig | None = None,
    encode_config: EncodeConfig | None = None,
) -> EncodeResult
```

```python
from noveltts.encode import encode_audio
from noveltts.config import EncodeConfig

result = encode_audio("output/audio/c003.wav", encode_config=EncodeConfig(format="m4a", bitrate="64k"))
print(f"{result.source_bytes/1e6:.1f}MB -> {result.output_bytes/1e6:.1f}MB")
```

**异常**：编码器不可用抛 `RuntimeError`（先用 `encoder_available()` 检查）。

### `encode_chapters`

```python
def encode_chapters(
    config: AppConfig,
    chapter_ids: list[str] | None = None,     # None = 自动发现
    *,
    keep_wav: bool | None = None,
) -> list[EncodeResult]
```

批量压缩各章。

### `encoder_available`

```python
def encoder_available(config: EncodeConfig | None = None) -> tuple[bool, str]
```

返回 `(是否可用, 说明)`。**GUI 启动时检查并给出提示**。

```python
ok, msg = encoder_available()
if not ok:
    show_warning(f"无法压缩音频：{msg}\n（不影响生成，只是体积大）")
```

macOS 上无 ffmpeg 时会自动退回系统自带 `afconvert`。

---

## 11. 硬件加速

这些接口让你在 GUI 里**展示**加速状态，不必自己探测硬件。

### `preflight`

```python
def preflight(
    generation: GenerationConfig,
    *,
    device: str | None = None,
) -> AccelProfile
```

**探测硬件 + 决定全部加速开关**。`build_engine` 内部会调它。
GUI 想提前显示"将要使用什么配置"，可单独调用。

### `inspect_device`

```python
def inspect_device(requested: str = "auto", *, index: int = 0) -> DeviceInfo
```

```python
info = inspect_device("auto")
print(info.kind, info.name, info.total_vram_gb, info.compute_capability)
```

### `recommend_batch_size`

```python
def recommend_batch_size(
    info: DeviceInfo,
    *,
    max_batch: int = 16,
) -> tuple[int, str]      # (批大小, 判定依据说明)
```

判定依据说明可直接显示给用户（解释"为什么是 12"）。

### `flash_attention_available`

```python
def flash_attention_available() -> tuple[bool, str]   # (是否可用, 说明)
```

### `DeviceInfo`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `kind` | str | `"cpu"` | `cuda` / `mps` / `cpu` |
| `name` | str | `"CPU"` | 设备名 |
| `index` | int | `0` | 序号 |
| `total_vram_gb` | float | `0.0` | 总显存（GB） |
| `free_vram_gb` | float | `0.0` | 可用显存（GB） |
| `compute_capability` | tuple[int,int] \| None | `None` | 如 `(8, 6)` = sm_86 |
| `cuda_version` | str \| None | `None` | |
| `torch_version` | str | `""` | |

### `AccelProfile`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `device` | str | `"cpu"` | 解析后的设备串 |
| `dtype` | str | `"bfloat16"` | |
| `attn_implementation` | str | `"sdpa"` | |
| `batch_size` | int | `1` | |
| `allow_tf32` | bool | `False` | |
| `cudnn_benchmark` | bool | `False` | |
| `clear_cache_every` | int | `0` | |
| `empty_cache` | bool | `False` | |
| `info` | DeviceInfo | | 硬件信息 |
| `reasons` | list[str] | `[]` | **每条决策的理由，可直接展示** |

```python
profile = preflight(config.generation)
for reason in profile.reasons:
    print(" •", reason)
```

### 批大小参考（自动推导结果）

| 显存 | 批大小 |
|---|---|
| 8 G | 2 |
| 12 G | 5 |
| 16 G | 7 |
| 24 G（3090/4090） | 13 |
| 32 G+ | 16（上限） |

依据：固定占用 4.5 G，每样本 1.2 G（实测 0.75 G + 60% 余量）。
**非 CUDA 设备（Mac/CPU）一律返回 1**。

---

## 12. 音频工具

### `read_audio`

```python
def read_audio(
    path: str | Path,
    *,
    start: int = 0,        # 起始采样点
    frames: int = -1,      # 读取点数，-1 = 全部
) -> tuple[ndarray, int]   # (样本, 采样率)
```

**分块读取**，用于流式拼接（避免把整本书读进内存）：

```python
samples, sr = read_audio(path, start=0, frames=24000)   # 读第 1 秒
```

### `write_audio`

```python
def write_audio(
    path: str | Path,
    samples: ndarray,
    sample_rate: int,
    *,
    subtype: str = "PCM_16",
) -> Path
```

**原子写入**（先写临时文件再 rename）。父目录不存在会自动创建。

### `time_stretch`（**语速**）

```python
def time_stretch(
    samples: ndarray,
    *,
    rate: float,                 # 1.0 = 原速，1.2 = 快 20%
    sample_rate: int = 24000,
) -> ndarray
```

**变速不变调**（相位声码器）。音高保持不变。

* 缺 librosa 时**安全返回原音频**并告警（不抛异常，不让整段报废）
* 建议范围 `0.5 ~ 2.0`

### `normalize_audio`

```python
def normalize_audio(samples: ndarray, config: AudioConfig) -> ndarray
```

按 `config.normalize` 做峰值或 RMS 归一。数字静音（全零）会被跳过。

### `probe_audio`

```python
def probe_audio(path: str | Path) -> AudioInfo
```

只读元信息，不解码音频。

### `trim_silence`

```python
def trim_silence(
    samples: ndarray,
    *,
    threshold_dbfs: float,
    sample_rate: int,
) -> ndarray
```

裁掉首尾静音。

---

## 13. 命令行接口

### 13.1 子命令总览

```text
noveltts <命令> [选项]
```

| 命令 | 作用 | 需要 GPU | 支持 `--json` |
|---|---|---|---|
| `info` | 环境与素材自检 | 否 | ✅ |
| `parse` | 解析电子书、切章 | 否 | ✅（`--list`） |
| `normalize` | 洗稿预览 | 否 | — |
| `segment` | 语意切分 | 否 | ✅ |
| `index` | 输出片段索引 | 否 | ✅ |
| `synth` | 合成音频 | ✅ | — |
| `concat` | 拼接 | 否 | ✅ |
| `encode` | 压缩 | 否 | ✅ |
| `run` | 端到端（切分→合成→拼接→压缩） | ✅ | — |

### 13.2 各命令参数

#### 全局

```text
-c, --config <路径>     配置文件（默认 ./config.yaml）
-v, --verbose           详细日志
-q, --quiet             安静模式
--version               版本
```

#### 公共选项（多命令共有）

```text
--source <路径>         指定素材文件（覆盖自动发现）
--chapters-json <路径>  用指定 chapters.json（覆盖自动发现）
```

#### `info`

```text
--json                  输出 JSON
```

示例输出（JSON）：

```json
{
  "version": "1.0.0",
  "root": "/path/to/NovelTTS",
  "config_path": "/path/to/NovelTTS/config.yaml",
  "source_dir": "/path/to/NovelTTS/sources",
  "output_dir": "/path/to/NovelTTS/output/audio",
  "sources": [{"name": "book.epub", "size": 8388608}],
  "generated_chapters": ["c001", "c002"],
  "dependencies": {"audio_io": {"ok": true, "detail": "ok"}, "encoder": {"ok": true, "detail": "ffmpeg"}},
  "model": {"path": "...", "device": "auto", "dtype": "bfloat16",
            "batch_size": "auto", "language": "English"}
}
```

#### `parse`

```text
--out <路径>            输出 chapters.json 路径
--list                  只列出章节（表格）
--dump-text             同时导出每章纯文本
--json                  配合 --list 输出 JSON
```

#### `normalize`

```text
--chapter <ID|序号|标题>
--text <字符串>         直接洗一段文本（不读文件）
--limit <N>             只看前 N 段
--show-unchanged        连未改写的也显示
```

#### `segment`

```text
--chapter <ID|序号|标题>
--out <路径>            输出片段 JSON
--preview <N>           只预览前 N 段
--no-normalize          不洗稿（调试用）
--json
```

#### `index`

```text
--chapter <ID|序号|标题>
--json
```

#### `synth`（**核心**）

```text
--chapter <ID|序号|标题>
--text <字符串>         直接合成一段文本（试听用，不读文件）
--segments <3,5,8-12>   只做指定片段（1 基）
--limit <N>             只做前 N 段（试听）
--force                 已存在也重新生成
--no-title              不生成章节标题
--language <English|...>
--batch-size auto|<N>
--device auto|cuda|mps|cpu
--rate <倍数>           语速（需配 --force 才生效于已存在片段）
--retry-failed          只重跑失败片段
--dry-run               假引擎，不加载模型（验证链路）
```

#### `concat`

```text
--chapter <ID|序号|标题>
--out <路径>
--gap <秒>              片段间停顿
--merge-book            把所有章节合并成整本
--book-name <名称>      整本文件名
--json
```

#### `encode`

```text
--chapter <ID|序号|标题>
--format m4a|mp3|opus|flac|ogg
--bitrate 64k
--keep-wav
--json
```

#### `run`（端到端）

```text
--chapter <ID|序号|标题>    不传 = 全部章节
--limit <N>
--force
--no-encode                 跳过压缩
--no-concat                 跳过拼接
--language / --device / --batch-size / --rate / --dry-run
```

### 13.3 章节选择器（四种写法）

```bash
--chapter c003                          # 完整 ID
--chapter 3                             # 数字序号
--chapter "Loss 1: The Purest..."       # 完整标题（含空格要引号）
--chapter Crimson                       # 标题唯一片段
```

匹配多章时报错并列出候选；匹配不到时报错并提示 `parse --list`。

### 13.4 日志格式（**做进度条要解析这个**）

格式固定为：

```
HH:MM:SS | LEVEL   | logger.name | 消息
```

**值得解析的行**：

| 模式 | 含义 |
|---|---|
| `完成 <章节> \| 新生成 N \| 跳过 N \| 音频 MM:SS \| 耗时 MM:SS \| RTF X.XX` | 单章完成 |
| `[5/44] 100.0%  音频 00:25  预计剩余 00:00` | 进度采样（每 `progress_every` 段） |
| `硬件与加速方案：` … | 加速方案（多行） |
| `批处理失败（N 段），降级为逐段重试：<原因>` | 警告：批处理降级 |
| `有 N 段失败，可用 --only-failed 单独重跑` | 有失败段 |
| `完成：音频 H:MM:SS / 合成耗时 H:MM:SS / RTF X.XX` | 全书完成 |

**正则示例**：

```python
import re
PROGRESS = re.compile(r"\[(\d+)/(\d+)\]\s+([\d.]+)%")
RTF_LINE = re.compile(r"音频\s+([\d:]+)\s*\|.*RTF\s+([\d.]+)")

for line in proc.stdout:
    m = PROGRESS.search(line)
    if m:
        done, total, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
        set_progress(done / total)
```

### 13.5 退出码

| 码 | 含义 |
|---|---|
| `0` | 成功 |
| `1` | 运行时错误（文件不存在、章节找不到、模型加载失败…） |
| `2` | 参数错误（argparse 报的） |

> **注意**：`synth` 有片段失败时**仍返回 0**（因为流程跑完了）。
> 要判断是否有失败，检查日志里的 `有 N 段失败` 或读 manifest 的 `error` 字段。

---

## 14. GUI 开发建议

### 14.1 推荐的界面结构

```text
┌─ 顶部状态栏 ──────────────────────────────────┐
│ 设备: cuda:0 (RTX 3090 24G)  │ 批大小: 12 │ TF32: on │
│ 加速方案: flash_attention_2 · 每 50 段清缓存        │
└───────────────────────────────────────────────┘
┌─ 左：章节列表 ─┬─ 中：片段列表 ─┬─ 右：属性/试听 ─┐
│ c001 Foreword  │ 0001 文本…     │ 原文 / 洗稿后    │
│ c002 Loss 1    │ 0002 文本…     │ [▶ 播放]        │
│ …              │ …              │ 时长 / 字符数    │
└────────────────┴────────────────┴─────────────────┘
┌─ 底部：操作条 ─────────────────────────────────┐
│ [解析] [洗稿预览] [切分] [试听5段] [生成本章]      │
│ [生成全书] [拼接] [压缩] [重跑失败]               │
│ ▓▓▓▓▓▓░░░░ 44%  2:15 已用 / 预计 1:30 剩余       │
└───────────────────────────────────────────────┘
```

### 14.2 界面要暴露的参数（按重要性）

**必须暴露**（用户高频调整）：

| 参数 | 控件 | 说明 |
|---|---|---|
| `--chapter` | 下拉/列表 | |
| `--rate` | 滑块 0.5~2.0 | **旁边提示"改语速需重新生成"** |
| `--limit` | 数字框 | 试听用 |
| `--device` | 下拉 | 显示可选设备 |
| `--batch-size` | 下拉（auto/1/2/4/8/…） | 默认 auto |
| `voice.ref_audio` | 文件选择 | |
| `voice.ref_text` | 多行文本框 | **校验：字数与音频时长是否匹配** |

**进阶（折叠面板）**：

`segment.max_chars` / `text_rules.*` / `generation.max_new_tokens` /
`generation.temperature` / `audio.min_gap_seconds` / `encode.format` / `encode.bitrate`

### 14.3 必须处理的 UX 细节（**都是实际踩过的坑**）

| 场景 | 处理方式 |
|---|---|
| **改语速后点生成** | 必须提醒"已存在的片段会被跳过"，并询问是否 `force=True` |
| 生成中点击关闭 | 提示"任务在后台运行，可断线"（若走 CLI 子进程） |
| `ref_text` 为空 | 警告"音色相似度会打折（xvec 模式）" |
| 生成前 | 检查 `ref_audio` 存在、模型路径有效 |
| 试听按钮 | **默认只做 5 段**，不要让用户一上来就跑全书 |
| 失败片段 | 显著显示 + 提供"重跑失败"按钮（`retry_failed`） |
| 磁盘空间 | 启动检查，提示"整本无损约需 700 MB" |
| 文本被改过 | 比对 manifest 的 `text_sha1`，提示"文本已变更，旧音频可能失效" |
| 长时间无输出 | 显示"正在加载模型…"（首次加载需数秒到数十秒） |

### 14.4 线程模型

`SynthesisSession` **不是线程安全的**，且引擎持有显存。推荐：

```python
# ✅ 推荐：子进程隔离（可随时 kill，崩溃不影响 GUI）
proc = subprocess.Popen(["noveltts", "run", ...])

# ⚠️ 可行但要小心：单后台线程 + 队列
class Worker(threading.Thread):
    def run(self):
        with SynthesisSession(self.config) as s:   # 在线程内创建
            for ch in self.chapters:
                if self.cancelled: break
                report = s.synthesize_chapter(ch)
                self.queue.put(report)

# ❌ 错误：多个线程共用一个 session / engine
```

### 14.5 进度来源

三种，按精度排序：

1. **解析日志的 `[N/M] X%` 行** —— 最细，推荐（格式见 §13.4）
2. **`report.generated / report.total`** —— 每章结束才知道
3. **数 manifest 的 records** —— 最粗

### 14.6 配置持久化

```python
# 保存用户设置 → config.local.yaml（不要写 config.yaml）
import yaml
settings = {"generation": {"device": "cuda", "batch_size": 8},
            "audio": {"speech_rate": 1.15}}
with open(root / "config.local.yaml", "w") as f:
    yaml.safe_dump(settings, f, allow_unicode=True, sort_keys=False)
```

> 用 `config.local.yaml` 而非 `config.yaml`：前者不入库，
> 代码更新不会冲掉用户设置。

---

## 15. 异常与错误处理

### 15.1 异常类型

项目**不定义自定义异常类**，直接用标准异常：

| 异常 | 触发场景 | GUI 应如何提示 |
|---|---|---|
| `FileNotFoundError` | 素材/模型/参考音频不存在 | "找不到文件：X，请检查路径" |
| `ValueError` | 章节匹配失败、参数非法、格式不支持 | 直接显示 `str(exc)`（消息已含解决提示） |
| `RuntimeError` | 模型加载失败、编码器不可用、引擎返回异常 | "生成失败：…"，建议检查设备/依赖 |
| `ImportError` / `ModuleNotFoundError` | 缺 torch / qwen_tts / soundfile | "缺少依赖 X，请先 pip install" |
| `json.JSONDecodeError` | manifest 损坏 | 已被内部捕获，会重建 |

### 15.2 失败隔离保证

| 层级 | 保证 |
|---|---|
| 单段失败 | 记入 `SegmentRecord.error`，**不中断**其它段，按 `num_retries` 重试 |
| 批量失败 | **自动降级为逐段重试**，隔离问题段 |
| 单章失败 | `SynthesisReport.failed` 计数，**不中断**整本书 |
| 全书失败 | 只有致命错误（模型加载不了）才抛异常 |

**GUI 应该**：把 `report.errors` 当作"部分成功"处理，而不是"全失败"。

### 15.3 常见错误消息与对策

| 消息 | 原因 | 解决 |
|---|---|---|
| `找不到章节 'c999'` | ID 错误 | 提示用户先 `parse --list` |
| `章节 'X' 匹配到多个` | 标题不唯一 | 消息里会列出候选 |
| `cannot cache function '__o_fold'` | numba 缓存目录不可写 | **已自动修复**；若仍出现设 `NUMBA_CACHE_DIR` |
| `CUDA out of memory` | 显存不足 | 已自动降批大小；仍失败则降 `max_new_tokens` |
| `flash_attn 不可用` | 没装或版本不匹配 | **不致命**，自动退回 `sdpa`，只是慢 |
| `找不到 ffmpeg` | 未安装 | macOS 会自动用 `afconvert`；仍失败则 `brew install ffmpeg` |
| `参考音频不存在` | 路径错误 | 检查 `voice.ref_audio` |
| `引擎返回段数不匹配` | 引擎实现有 bug | 自定义引擎时检查 `waveforms` 长度 |

### 15.4 幂等性（**GUI 可依赖**）

| 操作 | 重复执行结果 |
|---|---|
| `parse` | 幂等，覆盖 `chapters.json` |
| `segment` | 幂等，覆盖索引 |
| `synth`（默认） | **幂等且可续跑**（跳过已完成） |
| `synth --force` | 重新生成全部 |
| `concat` | 幂等，覆盖成品 |
| `encode` | 幂等，覆盖压缩文件 |

### 15.5 中断安全

* 每个片段生成后立刻落盘（原子写）
* 强杀进程（`kill -9`）最多损失**当前正在生成的那一段**
* manifest 在批次结束后写入，部分重跑时会**合并**保留不在本批的记录

---

## 附录 A：变更记录

| 版本 | 接口变化 |
|---|---|
| v1.0.0 | 首版：`AppConfig` / 解析 / 洗稿 / 切分 / 合成 / 拼接 / 压缩 |
| v1.1.0 | `GenerationConfig` 增 `batch_size`(auto) / `tf32` / `cudnn_benchmark` / `clear_cache_every` / `oom_batch_downgrade`；`AudioConfig` 增 `speech_rate`；新增 `accelerator` 模块；CLI 增 `--rate` |
| v1.2.0 | **`batch_size` 真正生效**（此前是死配置）；`SynthesisEngine` 契约要求 `waveforms` 长度 == `texts` 长度 |
| v1.2.1 | `config.local.yaml` 本机覆盖；`--rate` 等命令行参数优先级修正为最高 |
| v1.3.x | 输出目录支持项目外绝对路径；`scripts/deploy_remote.sh`；`scripts/remote_status.sh` |

## 附录 B：最小可用示例（完整可跑）

```python
"""从零到有声书，全部用 Python API。"""
from pathlib import Path

from noveltts.config import AppConfig, EncodeConfig
from noveltts.epub_parser import parse_book
from noveltts.segmenter import TextSegmenter
from noveltts.synthesis import SynthesisSession
from noveltts.concat import concat_from_manifest
from noveltts.encode import encode_audio, encoder_available

ROOT = Path("/path/to/NovelTTS")

# 1) 配置
config = AppConfig.from_dict({
    "project": {"root": str(ROOT), "output_dir": str(ROOT / "output/audio")},
    "generation": {"device": "auto", "batch_size": "auto"},
    "voice": {"ref_audio": "assets/ref/doubao.wav", "ref_text": "(逐字填入)"},
    "audio": {"speech_rate": 1.15},
})
config.project.resolve(ROOT)
config.project.ensure_dirs()

# 2) 解析
book = parse_book(ROOT / "sources/book.epub", config.parse)
print(f"《{book.title}》 {len(book.chapters)} 章")

# 3) 选一章
chapter = book.chapters[2]
print(f"处理 {chapter.chapter_id} {chapter.title}")

# 4) 切分（**必须先做**：parse_book 只给段落，不切分）
from noveltts.segmenter import TextSegmenter
segmenter = TextSegmenter(config.segment, language=book.language)
chapter.segments = segmenter.segment_chapter(chapter)
print(f"切分：{len(chapter.segments)} 段")

# 5) 先生成 5 段试听
with SynthesisSession(config) as session:
    preview = session.synthesize_chapter(chapter, limit=5)
    print(f"试听：生成 {preview.generated} 段，{preview.audio_seconds:.1f} 秒音频")

# 6) 满意后做整章
with SynthesisSession(config) as session:
    report = session.synthesize_chapter(chapter)
    print(f"完成：生成 {report.generated}，跳过 {report.skipped}，失败 {report.failed}")
    if report.errors:
        print("失败片段：", report.errors[:3])

# 7) 拼接
result = concat_from_manifest(config, chapter.chapter_id)
print(f"拼接：{result.output_path.name} {result.duration/60:.1f} 分钟")
if result.missing:
    print(f"⚠️ 缺失 {len(result.missing)} 段")

# 8) 压缩
ok, why = encoder_available()
if ok:
    enc = encode_audio(result.output_path, config=config,
                       encode_config=EncodeConfig(format="m4a", bitrate="64k"))
    print(f"压缩：{enc.source_bytes/1e6:.1f}MB -> {enc.output_bytes/1e6:.1f}MB")
else:
    print(f"跳过压缩（{why}）")
```
