# 产品化设计文档：从命令行工具到可用软件

> **这份文档是关于"未来要做什么"，不是"已经做了什么"。**
> 当前项目 `v2.0.0` 是一个命令行流水线；本文描述如何把它包装成一个
> 普通人能用的软件。
>
> 阅读顺序建议：[TECH-DEBT.md](TECH-DEBT.md)（先补地基）→ 本文（再谈产品）。
>
> **一句话结论（先说重要的）**：你要求的"分角色 + 情绪 instruct"和
> 当前的"音色克隆"**在模型层面是互斥的**。必须做一个选择。
> 详见 [§2.2](#22-关键技术结论情绪控制与音色克隆互斥)。

---

## 第一部分：需求描述

### 1.1 用户场景

一个不懂技术的用户，想把自己手上的一本电子书变成有声书。

```text
┌─────────────────────────────────────────────────────────────┐
│  第 1 步：拖入素材                                            │
│  ┌───────────────┐                                          │
│  │  拖入电子书     │  book.epub                              │
│  └───────────────┘                                          │
│  ┌───────────────┐                                          │
│  │  拖入参考音频   │  narrator.wav（旁白音色，可跳过）           │
│  └───────────────┘                                          │
├─────────────────────────────────────────────────────────────┤
│  第 2 步：连接云端 GPU                                        │
│   SSH 主机: connect.nmb2.seetacloud.com  端口: 16402         │
│   用户名 : root          密码/密钥: ••••••••                  │
│   [测试连接]  ✅ 已连接  RTX 4090 24G                         │
├─────────────────────────────────────────────────────────────┤
│  第 3 步：点「开始生成」，然后等                                 │
│   ▓▓▓▓▓▓▓▓░░░░░░░░  42%                                      │
│   正在合成 第 4 章 / 共 10 章 · 预计剩余 38 分钟                  │
├─────────────────────────────────────────────────────────────┤
│  第 4 步：拿到文件                                            │
│   📁 我的有声书/                                              │
│     ├── 我的有声书.m4b          带章节标记，可跳转              │
│     ├── 我的有声书.srt          字幕                          │
│     ├── 我的有声书.vtt          网页字幕                       │
│     ├── c001.m4a … c010.m4a     分章版本                      │
│     └── 角色表.json             识别出的角色与配音分配           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 功能需求清单

| # | 需求 | 优先级 | 难度 | 说明 |
|---|---|---|---|---|
| F1 | 拖入电子书 / 参考音频 | P0 | 低 | 文件选择 + 格式校验 |
| F2 | 输入 SSH 连接信息并自动部署环境 | P0 | **高** | 密码登录、装依赖、下模型、传项目 |
| F3 | 一键跑完整个流水线 | P0 | 低 | 已有 `noveltts run` |
| F4 | 音频内含**章节时间戳** | P0 | 中 | m4b 章节标记 + cue 表 |
| F5 | 生成**字幕文件** | P0 | 中 | SRT / VTT / ASS |
| F6 | AI **识别对话角色** | P0 | **高** | LLM 做说话人归属 |
| F7 | **旁白与角色分开朗读** | P0 | 中 | 多音色分配 |
| F8 | 每句对话打**情绪 instruct** | P1 | 中高 | 需支持 instruct 的模型 |
| F9 | 进度可视化 + 可中断续跑 | P1 | 低 | 已有断点续跑，需暴露到 UI |
| F10 | 失败段的查看与重做 | P1 | 中 | 依赖 TECH-DEBT §1 的质检器 |
| F11 | 角色表可人工修正 | P1 | 中 | 用户能改"这句话是张三说的" |

### 1.3 非功能需求

| 维度 | 要求 |
|---|---|
| **用户门槛** | 不装 Python、不用命令行、不需要懂 GPU |
| **成本透明度** | 显示预估耗时与云 GPU 费用 |
| **容错** | SSH 断开、实例重启、生成中断都要能恢复 |
| **隐私** | 电子书内容尽量不上传第三方（LLM 除外，见 §6） |
| **可解释** | 用户能看到"为什么这句分配给了这个角色/情绪" |

---

## 2. 关键技术结论（**动手前必须知道**）

### 2.1 当前项目的能力边界

现在的 `v2.0.0` 用 **Base 模型 + 音色克隆**：

- ✅ 单音色朗读全文（音色可以是你给的任意参考音频）
- ❌ 没有情绪控制
- ❌ 没有角色区分（全部一个声音）

### 2.2 关键技术结论：情绪控制与音色克隆**互斥**

我查证了 Qwen3-TTS 官方文档的模型对比表（[Available Models](https://mintlify.wiki/QwenLM/Qwen3-TTS/concepts/models)）：

| 模型 | 参数 | 特性 | **Instruction 控制**（情绪） |
|---|---|---|---|
| `1.7B-VoiceDesign` | 1.7B | 用自然语言描述创建音色 | ✅ |
| `1.7B-CustomVoice` | 1.7B | **9 个预置优质音色 + 风格控制** | ✅ |
| **`1.7B-Base`** | 1.7B | **3 秒音色克隆** | **❌** |
| `0.6B-CustomVoice` | 0.6B | 9 个预置音色 | ❌ |
| `0.6B-Base` | 0.6B | 3 秒音色克隆 | ❌ |

**看第 3 行和第 2 行的对比**：

```python
# Base（克隆）—— 有一个 instruct 参数就报错，因为它不支持
generate_voice_clone(text=..., ref_audio=..., ref_text=...)   # ← 当前用的
# 参数里没有 instruct

# CustomVoice（预置音色）—— 支持情绪
generate_custom_voice(
    text=[...], language=[...], speaker=["Ryan", ...],
    instruct=["愤怒地说", "温柔地说"],       # ← 每个角色/每句话的情绪
)
```

**结论：切换模型的能力，等于做一次取舍。**

### 2.3 三个方案对比

| | 方案 A：坚持克隆 | **方案 B：CustomVoice（推荐）** | 方案 C：混合 |
|---|---|---|---|
| 模型 | `1.7B-Base` | `1.7B-CustomVoice` | 旁白 Base + 对话 CustomVoice |
| 音色来源 | **你给的任意参考音频** | 9 个预置（英中日韩各有人声） | 旁白用你的音频，角色用预置 |
| 角色区分 | ❌ 无 | ✅ 9 个音色可分配 | ✅ |
| 情绪控制 | ❌ 无 | ✅ 逐句 instruct | 仅对话有情绪 |
| 用户拖参考音频 | ✅ 有用 | ⚠️ **无用**（仅作旁白备选） | ✅ 旁白用 |
| 实现复杂度 | 已实现 | 中（改引擎 + 角色分配） | 高（两个模型/两趟合成） |
| 借鉴"有声书"效果 | 单人朗读 | **更接近真人多角色广播剧** | 最接近出版级 |

**推荐方案 B**，理由：

1. 你的原话是「**以实现分角色朗读为首要，克隆可以抛弃**」—— 方案 B 正好符合
2. 方案 B 一条路同时解决 **F6 角色区分 + F7 分角色 + F8 情绪**
3. 方案 A 无论怎么优化都做不出"广播剧"效果

> ⚠️ **对用户界面的影响**：方案 B 下，用户拖进来的参考音频**不再用于克隆**。
> 界面上必须诚实说明，否则用户会以为"用了我的声音"而实际没有。
> 建议改成：「旁白音色（可选）」→ 体现为"你可以从预置音色里挑，或上传一段音频作为旁白"。

### 2.4 预置音色清单（来自官方 CustomVoice 文档）

| 音色 | 特征 | 母语 |
|---|---|---|
| Vivian | 明亮、略微锋利的年轻女声 | 中文 |
| Serena | 温暖柔和的年轻女声 | 中文 |
| Uncle_Fu | 低沉醇厚的成熟男声 | 中文 |
| Dylan | 清亮的北京腔年轻男声 | 中文（北京话） |
| Eric | 略带沙哑的成都方言男声 | 中文（四川话） |
| **Ryan** | 节奏感强的男声 | **英文** |
| **Aiden** | 阳光的美式男声，中频清晰 | **英文** |
| Ono_Anna | 活泼的日语女声 | 日语 |
| Sohee | 情感丰富的韩语女声 | 韩语 |

> ⚠️ **英文只有 2 个音色（Ryan / Aiden）**。这是方案 B 的**最大限制**：
> 一本英文小说如果有 5 个角色 + 1 个旁白，**音色不够分**。
>
> 缓解办法（按推荐顺序）：
> 1. 用 `VoiceDesign` 生成更多音色（它也能 instruct，且能凭空创造音色）
> 2. 音高微调（`librosa.effects.pitch_shift`）把同一音色派生出变体
> 3. 重要角色用预置，次要角色合并（"路人甲/乙"共用一个音色）
>
> **这条限制会影响产品设计**，见 [§4.4 角色配音分配](#44-角色配音分配核心难题)。

---

## 3. 系统架构

### 3.1 分层视图

```text
┌──────────────────────────────────────────────────────────────────┐
│  桌面界面（Electron / Tauri）                                      │
│  拖入文件 · SSH 表单 · 进度条 · 角色表编辑 · 试听                   │
└───────────────────────────┬──────────────────────────────────────┘
                            │ 本地 HTTP / IPC（JSON）
┌───────────────────────────▼──────────────────────────────────────┐
│  本地服务（Python FastAPI）—— 跑在用户机器上                        │
│  ┌──────────────┬──────────────┬──────────────┬────────────────┐ │
│  │ 任务编排      │ 文本分析      │ SSH 代理      │ 产物组装        │ │
│  │ (本地 CPU)    │ (调 LLM)     │ (paramiko)   │ (字幕/章节)     │ │
│  └──────────────┴──────────────┴──────────────┴────────────────┘ │
└───────────────────────────┬──────────────────────────────────────┘
                            │ SSH（用户提供的凭据）
┌───────────────────────────▼──────────────────────────────────────┐
│  云端 GPU 实例                                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  NovelTTS worker（本项目的演进版）                            │  │
│  │  解析 → 洗稿 → 切分 → 说话人归属 → 情绪标注 → 多音色合成        │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 关键设计决策：什么在本地跑，什么在云端跑

| 任务 | 放在哪 | 理由 |
|---|---|---|
| 电子书解析、洗稿、切分 | **本地** | 纯 CPU，快，且**不上传原文** |
| **说话人归属（LLM）** | 云端（或本地小模型） | 见 §6 隐私取舍 |
| 情绪标注 | 与上同一趟 LLM 调用 | 合并成一次 prompt，省一半调用 |
| TTS 合成 | **云端** | 需要 GPU |
| 字幕生成、章节标记 | **本地** | 只需 cue 表，秒级完成 |
| 音频拼接压缩 | 云端（已生成）或本地 | 云端省下载的往返，本地省云时长 |

> **重要经济性考虑**：GPU 有 9 个音色 → 若按音色分组批处理，
> 需要**按音色重排 + 分组**，否则同一批里混着不同 speaker 会怎样？
> 官方 batch 示例显示 `speaker` 可以是**逐项的列表**，所以
> **一批里可以混不同音色**（待实测确认质量）。若不能，则需按音色分趟，
> 会导致模型多次重载。**这是要在 PoC 阶段验证的关键点。**

### 3.3 端到端数据流

```text
book.epub
   │ ① 解析（本地）
   ▼
chapters[] ── ② 洗稿（本地） ──▶ normalized text
   │ ③ 切分（本地）
   ▼
segments[]  {segment_id, text, char_count, kind: narration|dialogue}
   │ ④ 说话人归属 + 情绪标注（LLM，一次调用处理一章）
   ▼
segments[]  + {speaker, emotion, confidence, evidence}
   │ ⑤ 角色配音分配
   ▼
segments[]  + {voice_id, instruct}
   │ ⑥ 上传任务描述到云端，多音色批量合成
   ▼
segments/*.wav + cue 表 {segment_id, start, end, text, speaker, voice_id, emotion}
   │ ⑦ 本地组装
   ▼
├── 我的有声书.m4b    （音频 + 章节标记）
├── 我的有声书.srt / .vtt / .ass
├── c001.m4a … c010.m4a
└── 角色表.json       （可人工修正后重跑）
```

**关键洞察**：**cue 表是整个产品的枢纽**。
它一旦存在，字幕、章节标记、按角色导出、人工校对全都变成简单操作。
当前项目**没有** cue 表（只有 manifest 的 duration），**这是要新增的核心数据结构**。

---

## 4. 分阶段实现计划

### Phase 0：PoC 验证（1~2 天）—— **必须先做，不要跳过**

这一步的目的：**在写任何界面之前，验证技术路线可行**。

| 验证项 | 方法 | 失败的后果 |
|---|---|---|
| **V1** CustomVoice 能否批量混不同 speaker | 一批 3 条不同 speaker，听效果 | 若不能，需按音色分趟，性能大降 |
| **V2** instruct 对情绪的控制力有多强 | 同一句文本，5 种 instruct，对比听感 | 若很弱，F8 情绪需求要降级 |
| **V3** 9 个音色是否够用 / 音高微调能否派生 | 用 librosa 移调 ±2 半音，听是否自然 | 影响角色分配策略 |
| **V4** 说话人归属准确率 | 拿现有那本书的 1 章，人工标 20 句做基准 | 若很差，F6 需人工介入为主 |
| **V5** VoiceDesign 补音色是否可行 | 设计 3 个新音色，听是否稳定 | 影响音色不够时的备选方案 |

**产出**：一份 PoC 报告 + 是否继续的判断。

> **强烈建议先做 V4**，因为它是**唯一可能推翻整个方案**的验证项。
> 如果 LLM 无法可靠区分"这句话是谁说的"（尤其是省略主语的对话），
> 那么 F6/F7 就要重新设计成"用户手动标注为主、AI 辅助"。

### Phase 1：核心引擎改造（1~2 周）

在现有代码基础上，**不涉及界面**。

#### 1.1 新增 `cue` 表（核心数据结构）

```python
# noveltts/models.py
@dataclass
class Cue:
    """一段音频的时间位置与元信息。字幕和章节标记都由它生成。"""
    segment_id: str
    chapter_id: str
    index: int
    start: float          # 秒，相对本章音频起点
    end: float
    text: str             # 用于字幕的文本（洗稿后的）
    display_text: str     # 用于字幕的展示文本（可含说话人前缀）
    speaker: str          # "narrator" 或角色名
    voice_id: str         # 实际使用的音色
    emotion: str          # 情绪标签
    instruct: str         # 实际发给模型的指令
    kind: str             # narration | dialogue | heading
```

**为什么必须有 `display_text` 和 `text` 两个字段**：
`text` 是送给模型的（洗稿后，"Volume eight"），
`display_text` 是给读者看的（原书原文，"Vol. 8"）。
**字幕必须用原文，不能用手洗稿后的文本** —— 否则字幕里会出现一堆拼写出来的数字。

> ⚠️ 这是当前项目**没有**做的：`Segment` 同时承担了"给模型读"和"给用户看"两个职责。
> 虽然 `Segment` 有 `source_text`（洗稿前），但它没有保持段落级对齐。

#### 1.2 引擎抽象：支持多模型

```python
# noveltts/engine.py 的演进
class SynthesisEngine(Protocol):
    """现有协议不变，但增加多音色能力。"""

    def synthesize_multi(
        self,
        items: list[SynthesisItem],       # 每条含 text + voice + instruct
        *,
        language: str | None = None,
    ) -> SynthesisResult:
        """多音色、带情绪指令的批量合成。"""
```

```python
@dataclass
class SynthesisItem:
    text: str
    voice_id: str            # "Ryan" / "Aiden" / 自定义
    instruct: str = ""       # "愤怒地质问" / ""（无情绪）
    speaker: str = ""        # 角色名（仅用于 cue 表，不进模型）
```

**兼容策略**：保留现有的 `synthesize(texts)` 单音色路径，
`Base` 模型引擎实现它，`CustomVoice` 引擎实现 `synthesize_multi`。
上层根据模型能力选择调用哪个。

#### 1.3 说话人归属（LLM）

```python
# noveltts/casting.py
@dataclass
class SpeakerAttribution:
    segment_id: str
    speaker: str              # "narrator" | "花泽" | "未知角色1"
    emotion: str              # "neutral" | "angry" | "happy" | "sad" | ...
    confidence: float         # 0~1，低置信度要提示用户复核
    evidence: str             # 判断依据（"上一段是张三说，这里没有引出新说话人"）

def attribute_speakers(
    chapter: Chapter,
    segments: list[Segment],
    *,
    llm: LLMClient,
    known_characters: list[str] | None = None,
) -> list[SpeakerAttribution]:
    """用 LLM 判断每段是谁说的、什么情绪。

    关键：**一次调用处理一整章**，而不是逐段调用。
    因为对话归属强依赖上下文（"他说" 指谁，要看前文）。
    """
```

**Prompt 设计要点**（这是准确率的关键）：

1. **给整章上下文**，不要逐段问
2. **先让 LLM 抽出角色列表**，再分配（两阶段，减少"发明新角色"）
3. **要求输出 evidence**（判断依据）—— 便于人工复核，也提升 LLM 自身准确率
4. **要求 confidence** —— 低置信度的段落要提示用户
5. **明确指示"保持已有角色名，不要发明同义词"**（避免"张三"/"张先生"/"他"被当成 3 个角色）
6. **结构化输出**（JSON schema / function calling）

**输出示例**：

```json
{
  "characters": [
    {"name": "温水和彦", "gender": "male", "role": "protagonist"},
    {"name": "八奈见杏菜", "gender": "female", "role": "main"}
  ],
  "attributions": [
    {"segment_id": "c001-0003", "speaker": "narrator", "emotion": "neutral",
     "confidence": 0.95, "evidence": "无引号的叙述段"},
    {"segment_id": "c001-0007", "speaker": "八奈见杏菜", "emotion": "angry",
     "confidence": 0.88, "evidence": "引号内对话，前句为「八奈见瞪着他说」"}
  ]
}
```

#### 1.4 情绪 → instruct 模板

```python
# noveltts/emotion.py
EMOTION_INSTRUCT = {
    "neutral":  "",
    "angry":    "用愤怒、提高音量的语气说",
    "happy":    "用愉快、轻快的语气说",
    "sad":      "用低落、缓慢的语气说",
    "surprised":"用惊讶、语调上扬的语气说",
    "nervous":  "用紧张、略带颤抖的语气说",
    "shy":      "用害羞、声音渐弱的语气说",
    "cold":     "用冷淡、平淡的语气说",
    # 旁白专用
    "narration": "用平稳、清晰的叙述语气，语速适中",
}
```

> ⚠️ **instruct 用中文还是英文？** 官方英文示例里 instruct 是英文
> （`"Male, 25 years old, friendly and professional tone"`），
> 中文示例里是中文。**需要实测哪个对英文文本效果更好**。列入 Phase 0 V2。

#### 1.5 角色配音分配

见 [§4.4](#44-角色配音分配核心难题)。

#### 1.6 cue 表生成（拼接时顺手产出）

```python
# noveltts/concat.py 的扩展
def concat_with_cues(
    config: AppConfig,
    chapter_id: str,
    *,
    cues_path: str | Path | None = None,
) -> tuple[ConcatResult, list[Cue]]:
    """边拼接边记录每段的时间位置。"""
```

**为什么拼接时生成最准**：因为拼接过程**本来就知道**每段的精确时长
和插入的间隔（`min_gap_seconds`）、淡入淡出（`join_fade_ms`）、
首尾静音。事后用时长累加会有累积误差。

#### 1.7 字幕与章节标记导出

```python
# noveltts/subtitles.py（新模块）
def write_srt(cues: list[Cue], path: Path, *, with_speaker: bool = True) -> Path:
    """生成 SRT 字幕。"""

def write_vtt(cues: list[Cue], path: Path) -> Path:
    """WebVTT，网页播放器用。"""

def write_ass(cues: list[Cue], path: Path) -> Path:
    """ASS，可给不同角色不同颜色。"""

def write_ffmetadata(chapters: list[Chapter], path: Path) -> Path:
    """生成 ffmpeg 章节元数据文件。

    格式：
        ;FFMETADATA1
        [CHAPTER]
        TIMEBASE=1/1000
        START=0
        END=294000
        title=Foreword
    """
```

**章节标记的实现**（已确认 ffmpeg 支持，见本文档开头验证）：

```bash
# 用 ipod muxer 产出带章节的 m4b（有声书标准格式）
ffmpeg -i book.m4a -i chapters.ffmetadata -map_metadata 1 \
       -codec copy -f ipod book.m4b

# 验证
ffprobe -show_chapters book.m4b
```

**产物矩阵**：

| 文件 | 内容 | 用途 |
|---|---|---|
| `我的有声书.m4b` | 音频 + 章节标记 | Apple Books / 播客App，可跳章 |
| `我的有声书.m4a` | 音频（兼容格式） | 通用播放器 |
| `我的有声书.srt` | 字幕 | 视频软件/VLC |
| `我的有声书.vtt` | 网页字幕 | 浏览器 `<track>` |
| `我的有声书.ass` | 样式化字幕 | 可区分角色颜色 |
| `c001.m4a` … | 分章 | 单章收听 |
| `角色表.json` | 角色与配音分配 | 人工修正后可重跑 |
| `cues.json` | 时间轴元数据 | 二次开发 |

### Phase 2：云端自动化（1~2 周）

#### 2.1 SSH 连接与凭据管理

```python
# noveltts/remote/connection.py
@dataclass
class RemoteTarget:
    host: str
    port: int = 22
    user: str = "root"
    auth: PasswordAuth | KeyAuth

class RemoteConnection:
    """封装 paramiko，提供幂等的远程操作。"""

    def test(self) -> RemoteInfo:
        """测试连接，返回 GPU / 磁盘 / 已装依赖情况。"""

    def ensure_environment(self, on_progress: Callable) -> None:
        """幂等地准备环境（可重复调用）。

        1. 检查 python / torch / CUDA
        2. 检查 qwen-tts，缺则 pip install
        3. 检查 ffmpeg，缺则 apt install
        4. 检查模型，缺则下载（或提示用户手动准备）
        """
```

**凭据安全**（重要）：

| 做法 | 评价 |
|---|---|
| ❌ 存明文到配置文件 | 绝对不要 |
| ✅ 用系统钥匙串（`keyring` 库）：macOS Keychain / Windows Credential Manager | **推荐** |
| ✅ 支持密钥登录（优先于密码） | 推荐，且更安全 |
| ✅ 只在内存中保留密码，不落盘 | 至少要做到这个 |
| ✅ 首次连接让用户确认主机指纹 | 防中间人 |

#### 2.2 一键部署（复用并改进现有脚本）

现有 `scripts/deploy_remote.sh` 可复用，但产品化要改：

1. **打包时写入版本信息**（见 [TECH-DEBT §9](TECH-DEBT.md)）
2. **支持 git clone 方式**（版本明确、可回滚）
3. **依赖检查要能跳过**（用户可能已装好，别每次都 `pip install`）
4. **模型检查**：不在云端自动下 14G（太慢），改为**提示用户**
   或使用预装镜像

#### 2.3 远端任务执行与进度回传

```python
# 远端跑：把任务写成 JSON 传过去，输出 JSONL 进度流
ssh.run("noveltts worker --task /tmp/task.json --progress-jsonl")
```

**进度协议**（JSONL，一行一个事件）：

```jsonl
{"event":"start","chapter":"c001","total_segments":62}
{"event":"batch_done","chapter":"c001","done":12,"total":62,"eta_seconds":95}
{"event":"chapter_done","chapter":"c001","audio_seconds":294.0,"rtf":0.49}
{"event":"qa_issues","chapter":"c001","count":2,"issues":[...]}
{"event":"all_done","total_audio_seconds":16131.0}
```

**这个协议是本地服务与界面之间的契约**，也是远端与本地之间的契约。
现有项目的日志是给人看的，**需要加一个机器可读的 JSONL 输出模式**。

#### 2.4 断线与恢复

| 场景 | 处理 |
|---|---|
| 本地网络断 | 远端任务继续（`setsid`），本地重连后继续读日志 |
| SSH 完全断开 | 后台任务不受影响（已实现），重连后 tail 日志 |
| 云端实例被释放 | **检测并提示**，用户需重新开始（已生成的段无法取回） |
| 本地程序崩溃 | 任务状态落盘（`task-state.json`），重启后恢复进度显示 |

### Phase 3：界面（2~3 周）

#### 3.1 技术选型

| 方案 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| **Electron + React** | 生态成熟，跨平台，好调试 | 体积大（~150MB） | ✅ **推荐**，除非很在意体积 |
| Tauri + React | 体积小（~10MB），Rust 后端 | 生态较小，需 Rust | 体积敏感时选 |
| PySide6 / Qt | 与 Python 同语言 | 界面开发效率低，样式难做 | 不推荐 |
| Web（本地起服务 + 浏览器） | 最简单，跨平台 | 需要用户开浏览器，体验略差 | 快速原型可用 |

**推荐 Electron**，因为：
1. 前端生态现成（拖拽上传、进度动画、表单校验都有成熟库）
2. 调试方便
3. 与本地 Python 服务用 HTTP/IPC 通信，职责清晰

#### 3.2 界面组件清单

| 组件 | 职责 | 复用/自建 |
|---|---|---|
| 文件拖拽区 | 校验 epub 格式、显示元信息 | 自建（`react-dropzone`） |
| SSH 表单 | 连接测试、进度提示、错误诊断 | 自建 |
| **角色表编辑器** | 显示 AI 识别的角色 + 音色分配，允许改 | **自建，核心难点** |
| 台词校对视图 | 逐句显示"谁说的 / 什么情绪"，可修正 | 自建 |
| 试听播放器 | 生成前先听 5 句，波形图 + 字幕同步 | 自建（`wavesurfer.js`） |
| 进度面板 | 分章进度、ETA、失败段列表 | 自建 |
| 产物面板 | 下载/打开文件、试听成品 | 自建 |

#### 3.3 关键技术难点：角色表编辑

这是**整个界面最难也最有价值**的部分。因为：

1. AI 的角色识别**一定会有错**（尤其省略主语的对话）
2. 用户必须能**低成本地修正**
3. 修正后要能**只重跑受影响的段**（不能全书重来）

设计建议：

```text
┌── 角色表 ──────────────────────────────────────────┐
│ 角色         出现次数   分配音色        试听         │
│ 旁白          1240      [Ryan ▼]       [▶]         │
│ 温水和彦       412       [Aiden ▼]      [▶]         │
│ 八奈见杏菜     388       [Vivian ▼]     [▶]         │
│ 烧盐柠檬       156       [Ono_Anna ▼]   [▶]         │
│ 未知角色1       12       [Serena ▼]     [▶] ⚠️低置信 │
│                                      [+ 合并到…]    │
└────────────────────────────────────────────────────┘
│ ⚠️ 有 12 句置信度低，建议复核        [立即复核]        │
└────────────────────────────────────────────────────────┘
```

**"只重跑受影响的段"的实现**：cue 表 + manifest 让这件事变得简单 ——
改完角色表后，比对哪些 segment 的 `voice_id`/`instruct` 变了，
只重新合成那些段，然后**重新拼接**（拼接很快，几秒）。

### Phase 4：打磨（1~2 周）

| 事项 | 说明 |
|---|---|
| 错误诊断 | SSH 失败、显存不足、磁盘满 → 给出可操作的建议 |
| 成本预估 | 显示"预计 1.5 小时 GPU 时间，约 3 元" |
| 质检报告 | 集成 [TECH-DEBT §1](TECH-DEBT.md) 的质检器，列出可疑段 |
| 多语言 | 洗稿规则、prompt、instruct 都要支持中/日/英 |
| 安装包 | 打包为 `.dmg` / `.exe` / `.AppImage`，内置 Python 运行时 |
| 自动更新 | 或至少提示新版本 |

---

## 5. 技术栈

### 5.1 总览

```text
┌─ 界面层 ─────────────────────────────────────────────────────┐
│ Electron 30+ · React 18 · TypeScript · Vite                  │
│ TailwindCSS（样式）· zustand（状态）· react-dropzone（拖拽）    │
│ wavesurfer.js（波形试听）· Radix UI / shadcn（组件）            │
├─ 本地服务层（Python 3.12）─────────────────────────────────────┤
│ FastAPI + uvicorn（HTTP/SSE 给界面）                           │
│ pydantic v2（数据校验，与 dataclass 互转）                      │
│ paramiko（SSH）· keyring（凭据）· cryptography（密钥）           │
│ httpx（调 LLM / 下载）· tenacity（重试）                        │
├─ 文本分析层 ──────────────────────────────────────────────────┤
│ LLM API（见 §5.3）· instructor / outlines（结构化输出约束）       │
│ jieba 或 sudachipy（中/日分词，辅助台词识别）                    │
├─ 音频层 ─────────────────────────────────────────────────────┤
│ soundfile · librosa · numpy                                  │
│ ffmpeg（编码 / 章节标记 / 字幕混流）                            │
├─ 云端合成层 ──────────────────────────────────────────────────┤
│ PyTorch 2.8 + CUDA 12.x                                      │
│ qwen-tts（Base / CustomVoice / VoiceDesign）                  │
│ flash-attn（可选）· vLLM（可选，高吞吐）                        │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 界面层选型理由

| 技术 | 为什么 | 替代方案 |
|---|---|---|
| **Electron** | 生态最成熟；拖拽/播放器/动画库齐全；调试方便 | Tauri（体积小但要 Rust） |
| **React + TS** | 组件生态最大；类型安全对复杂表单很重要 | Vue / Svelte |
| **Vite** | 构建快，HMR 好 | webpack（慢） |
| **TailwindCSS** | 快速做出一致界面，无需写 CSS 文件 | CSS Modules |
| **zustand** | 轻量状态管理，适合这种"任务 + 表单"模型 | Redux（太重） |
| **wavesurfer.js** | 现成的波形播放器，支持选区与时间轴 | 自己写 canvas（不值） |

### 5.3 LLM 选型（**关键决策**）

用途：说话人归属 + 情绪标注。输入是**一章原文**（约 2500~50000 字符）。

| 方案 | 优点 | 缺点 | 适用 |
|---|---|---|---|
| **云 API**（Claude / GPT / Gemini / Qwen） | 准确率最高，无需硬件；长上下文好 | **要上传原文**（隐私）；按 token 计费 | **推荐起步** |
| **本地小模型**（Qwen2.5-14B / 32B via Ollama） | 原文不出本机，隐私最好 | 用户要下载 10~20G 模型；慢；准确率略低 | 隐私敏感时 |
| 云端 GPU 上跑 LLM | 不占用用户机器 | 占显存，与 TTS 争资源 | 不推荐 |
| 规则 + 正则 | 零成本、零隐私风险 | 准确率低（只适合引号显式标注的书） | 兜底降级 |

**推荐策略：可插拔 + 分级降级**

```python
class SpeakerAttributor(Protocol):
    def attribute(self, chapter, segments) -> list[SpeakerAttribution]: ...

class LLMAttributor:      ...    # 云 API
class LocalLLMAttributor: ...    # Ollama
class RuleBasedAttributor:        # 正则兜底
    """只处理 `"..." said X.` 这类显式句式，其余标为 narrator + 低置信度。"""
```

**成本估算**：一本 19 万字 ≈ 13 万 token 输入。
用中等价位模型（约 $3/M input）≈ **$0.4 / 本**。可接受。
但**要注意**：如果按段调用而非按章调用，成本会翻几十倍。**必须按章批处理。**

**结构化输出**：用 function calling / JSON schema 强约束，避免解析自由文本。

### 5.4 音频与容器层

| 需求 | 技术 | 说明 |
|---|---|---|
| 读写 wav | `soundfile` | 已有，流式 |
| 变速不变调 | `librosa.effects.time_stretch` | 已有 |
| **音高微调**（派生音色） | `librosa.effects.pitch_shift` | **新增** |
| 编码 m4a/mp3 | `ffmpeg` | 已有 |
| **章节标记 m4b** | `ffmpeg -f ipod` + ffmetadata | **新增**（已验证可行） |
| **字幕 SRT/VTT/ASS** | 自己生成文本格式 | **新增**，格式简单无需库 |
| 波形图（界面） | `wavesurfer.js` 前端生成 | 不必后端出图 |
| 时长探测 | `soundfile.info` / `ffprobe` | 已有 |

> **字幕不需要第三方库**。SRT/VTT 格式极简单，自己拼接字符串即可
> （避免又一个"重复造轮子"的反面：这里用库和手写成本差不多，
> 而手写能精确控制"是否显示说话人"这类需求）。

### 5.5 云端与运维层

| 需求 | 技术 | 说明 |
|---|---|---|
| SSH | `paramiko` | 纯 Python，跨平台 |
| 凭据存储 | `keyring` | 系统钥匙串 |
| 远端长任务 | `setsid` + `nohup`（已有方案） | 已验证抗断线 |
| 进度回传 | JSONL over SSH stdout | 需新增机器可读输出 |
| 部署 | git clone + 版本标记 | 见 TECH-DEBT §9 |
| 实例探测 | `nvidia-smi --query-gpu --format=csv` | 已有 |

### 5.6 打包与分发

| 平台 | 方案 | 注意 |
|---|---|---|
| macOS | Electron Builder → `.dmg`（arm64 + x64） | 需签名与公证，否则用户看到"未验证开发者" |
| Windows | NSIS 安装包 | 需签名 |
| Linux | AppImage / .deb | 最简单 |

**Python 运行时怎么办**（这是个真问题）：

| 方案 | 评价 |
|---|---|
| 用户自己装 Python + `pip install` | ❌ 违背"零门槛"目标 |
| **PyInstaller 打包成单文件二进制** | ✅ 推荐，界面只调用这个二进制 |
| 用 `python-build-standalone` 内嵌 | ✅ 体积可控，但集成麻烦 |
| 云端做所有 Python 工作 | ⚠️ 那原文就要上传（隐私问题） |

**推荐**：本地 Python 部分（解析/洗稿/切分/字幕）用 **PyInstaller**
打成单文件，由 Electron 以子进程方式调用，通过 JSONL 通信。
这样界面层完全没有 Python 依赖问题。

---

## 6. 风险与取舍

### 6.1 技术风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| **LLM 说话人归属准确率不够** | **中高** | 高（核心功能） | Phase 0 V4 先验证；设计中保留人工修正路径；规则兜底 |
| **英文只有 2 个预置音色** | 高 | 中 | VoiceDesign 补音色；音高微调派生；合并次要角色 |
| **instruct 情绪控制力弱** | 中 | 中 | Phase 0 V2 验证；若弱则降级为"仅区分音色，不做情绪" |
| **CustomVoice 不能批量混 speaker** | 中 | 中 | 按音色分趟（性能降但可行） |
| 用户 SSH 环境千奇百怪 | 高 | 中 | 提供"环境自检"报告 + 明确错误提示；推荐固定镜像 |
| 显存不足导致批处理失败 | 中 | 低 | 已有 OOM 自动降级 |
| 模型许可 / 商用限制 | ? | **高** | **见 §6.3，必须核实** |

### 6.2 产品取舍

| 取舍 | 建议 |
|---|---|
| **云端 GPU vs 本地 GPU** | 只支持云端（本地 GPU 用户的驱动/CUDA 问题会淹没支持成本） |
| **自动租 GPU vs 用户自备** | 先做**用户自备**（如你现在的 AutoDL）。自动租需要对接云厂商 API + 计费，复杂且涉及支付合规 |
| **支持哪些输入格式** | 先只支持 EPUB 与 TXT。PDF 解析质量参差，会带来大量支持成本 |
| **一次能处理多长的书** | 先限定"单本 ≤ 50 万字"，超长书的成本与失败率都要重新评估 |

### 6.3 ⚠️ 必须先核实的合规问题

**这几条如不澄清，产品化可能直接不成立：**

| 问题 | 为什么要紧 |
|---|---|
| **Qwen3-TTS 的商用许可** | 模型权重有许可条款（Apache 2.0？还是有限制？）。**是否需要商用授权要查清** |
| **音色克隆的合法性** | 克隆他人声音可能侵犯声音权。方案 B 用预置音色规避了大部分风险，但"上传参考音频"功能仍有风险 |
| **电子书版权** | 把有版权的书做成有声书并分发，是侵权。**产品定位必须是"个人使用工具"**，不能内置任何书源 |
| **LLM 上传原文的隐私** | 用户可能不愿把未出版手稿上传第三方。**必须显著说明并给本地 LLM 选项** |

> 建议：产品定位为**个人自用工具**（类似 ffmpeg / HandBrake 的定位），
> 明确声明"使用者需自行确保对素材拥有合法使用权"。
> **不要**做成一键生成并分享的平台，那会立刻踩版权。

### 6.4 成本模型（给用户看）

以一本 19 万字 / 4.5 小时音频为例：

| 项目 | 用量 | 单价 | 成本 |
|---|---|---|---|
| GPU 合成 | 约 1.6 小时 @ 3090 | ~1.5 元/小时 | **约 2.4 元** |
| GPU 合成（若用 4090，更快） | 约 1 小时 | ~2 元/小时 | 约 2 元 |
| LLM 角色+情绪标注 | 约 13 万 input token | ~$3/M | **约 3 元** |
| 云端存储 | 1.6 GB | 少量 | 忽略 |
| **合计** | | | **约 5~6 元/本** |

> 注意：**LLM 成本可能超过 GPU 成本**。如果按段调用 LLM 而不是按章，
> 成本会涨到几十元。这是一个必须在架构上守住的点。

---

## 7. 建议的推进顺序

```text
Phase 0  PoC 验证（1~2 天）★ 不要跳过
  ├─ V4 说话人归属准确率  ← 先做这个，它可能推翻方案
  ├─ V1 批量混 speaker
  ├─ V2 instruct 情绪控制力
  ├─ V3 音色数量与派生
  └─ V5 VoiceDesign 补音色
       ↓ 若 V4/V2 不达标，回到 §2.3 重新选方案

Phase 0.5  先补 TECH-DEBT 的 P0/P1
  ├─ 音频质检器
  ├─ 真实模型测试
  ├─ 迁移 pytest
  └─ 删自写 YAML
       ↓ 地基不稳，产品化会把问题放大

Phase 1  核心引擎改造（1~2 周）
  ├─ cue 表（核心数据结构）
  ├─ SynthesisItem + synthesize_multi
  ├─ casting.py（说话人归属）
  ├─ emotion.py（情绪 → instruct）
  ├─ 角色配音分配
  └─ subtitles.py + 章节标记

Phase 2  云端自动化（1~2 周）
  ├─ SSH 连接 + 凭据安全存储
  ├─ 环境自检与幂等部署
  ├─ JSONL 进度协议
  └─ 断线与恢复

Phase 3  界面（2~3 周）
  ├─ 文件拖拽 + SSH 表单
  ├─ 角色表编辑器  ← 最难也最有价值
  ├─ 试听播放器
  └─ 进度与产物面板

Phase 4  打磨与分发（1~2 周）
  ├─ 错误诊断与成本预估
  ├─ 质检报告集成
  ├─ 多语言
  └─ 打包安装
```

**总计约 8~12 周**（单人全职）。若只做 Phase 0~1 + 最简单的命令行界面，
**2~3 周**就能验证"分角色 + 情绪"的核心价值。

---

## 8. 最小可行产品（MVP）建议

如果不想等 8~12 周，**建议先做这个**：

| 做 | 不做 |
|---|---|
| ✅ Phase 0 PoC 全部验证项 | ❌ 图形界面 |
| ✅ cue 表 + SRT 字幕 + m4b 章节标记 | ❌ SSH 自动化（用户手动贴命令） |
| ✅ CustomVoice 多音色 + 分角色朗读 | ❌ 情绪 instruct（等 PoC 结论） |
| ✅ 说话人归属（LLM），输出角色表 JSON | ❌ 角色表图形编辑器 |
| ✅ 保留现有命令行体验 | ❌ 打包安装 |

**MVP 交付形态**：在现有 `noveltts` 上新增子命令：

```bash
noveltts cast --chapter c001            # 识别角色 + 情绪，输出 cast.json
noveltts cast --chapter c001 --review   # 人工核对（可选）
noveltts run --multivoice --cast cast.json   # 多音色合成
noveltts subtitle --book "书名"          # 生成 srt/vtt/ass
noveltts chapterize --book "书名"        # 生成带章节的 m4b
```

**MVP 的价值**：验证"分角色 + 字幕 + 章节"是否真的比单音色好得多。
如果好，再投入界面；如果不够好，省下 8 周。

---

## 附：本文档的依据

| 结论 | 依据 |
|---|---|
| Base 模型不支持 instruct | Qwen3-TTS 官方模型对比表（Instruction Control 列 ❌） |
| CustomVoice 支持逐项 instruct | [官方 CustomVoice 文档](https://mintlify.wiki/QwenLM/Qwen3-TTS/guides/custom-voice) 的 `instruct=["", "Very happy.", ""]` 批量示例 |
| 9 个预置音色及其特征 | 同上，含母语与描述 |
| VoiceDesign 支持 instruct | [官方 Voice Design 文档](https://mintlify.wiki/QwenLM/Qwen3-TTS/guides/voice-design) 的 `instruct=` 参数 |
| 支持 10 种语言 | 官方模型页语言列表 |
| ffmpeg 可写章节标记 | 本机 `ffmpeg -muxers` 显示 `ipod`；`ffprobe -show_chapters` 存在 |
| 当前项目无字幕能力 | `grep -c "srt\|vtt\|subtitle\|timestamp" noveltts/*.py` 全为 0 |
| GPU 利用率偏低 | 云端整本生成期间采样的 `nvidia-smi`（27~39%） |
