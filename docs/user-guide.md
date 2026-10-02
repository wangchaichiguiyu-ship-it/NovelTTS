# NovelTTS 使用说明书

> 这份文档面向**实际操作**：从零配环境、导入小说、逐条敲命令，到拿到有声书。
>
> 配套文档：
> - [cloud-deployment.md](cloud-deployment.md) —— 云端（AutoDL / NVIDIA GPU）部署与验证
> - [git-workflow.md](git-workflow.md) —— 本仓库的 git 使用规范与常用命令
> - [../README.md](../README.md) —— 设计与原理
>
> 设计细节与原理请见 [README.md](../README.md)，这里只讲"怎么做、会看到什么"。

---

## 0. 本次已经为你完成的事情

| 项目 | 状态 |
|---|---|
| 依赖检查 | numpy 2.5.3 / soundfile 0.14.0 / torch 2.14.0 / qwen-tts 0.1.1 / transformers 4.57.3 / accelerate 1.12.0 **全部已就绪**，无需再装 |
| ffmpeg | `/opt/homebrew/bin/ffmpeg` 已就绪（压缩 m4a 用） |
| `noveltts` 命令 | **已安装**到 conda 环境 `qwen3-tts`（原来只在项目目录里能 import，没有命令） |
| numba 缓存故障 | **已修复**（见 §6.1，这是个会让模型完全跑不起来的坑） |
| 试运行 | **已完成**，用真实模型生成了 5 段试听音频 |

### 试听文件在这里

```
output/audio/preview_c003.wav          ← 5 段拼在一起，26.5 秒，直接播放这个
output/audio/segments/c003/c003-0001.wav ~ c003-0005.wav    ← 5 个独立片段
```

播放：

```bash
conda activate qwen3-tts
cd /Users/wanghaoxin/Documents/AustinProject/NovelTTS
afplay output/audio/preview_c003.wav          # macOS 自带播放器
open output/audio/preview_c003.wav            # 或用默认应用打开
```

这 5 段对应的是**第一章里最开头的 5 个片段**（含章节标题）：

| 片段 | 文本 | 时长 |
|---|---|---|
| c003-0001 | `Intermission: Appeal to Emotion` | 2.3s |
| c003-0002 | `AFTER SCHOOL, IN THE STUDENT COUNCIL ROOM, President Houkobaru Hibari was wiping her desk off with a damp rag.` | 6.6s |
| c003-0003 | `Treasurer Sakurai Hiroto stopped organizing shelves long enough to turn to her and smile.` | 5.4s |
| c003-0004 | `"Jumping the gun a little, no? You still have a month left," he said.` | 4.2s |
| c003-0005 | `"You don't earn respect by waiting until the last minute. We must be proactive about leaving places better than we found them."` | 6.7s |

**这次实际跑出来的参数**（说明你的配置是对的）：

```
设备 mps          精度 bfloat16      注意力 sdpa
音色模式 icl      参考音频 doubao.wav  输出采样率 24000 Hz
模型加载 5.3s     音色 prompt 构建 3.9s     生成速度 RTF ≈ 2.2~2.5
```

> **RTF（Real-Time Factor）= 生成耗时 ÷ 音频时长。**
> RTF 2.2 表示"生成 1 秒音频要花 2.2 秒"。这台 Mac 上整本书
> （3 小时 49 分音频）大约需要 **8~10 小时**；云端 CUDA + FlashAttention 通常能到
> RTF 0.1~0.3，也就是 **20~60 分钟**。

---

## 1. `c003` 这种参数是什么？命名规范

### 1.1 它是"章节 ID"，由程序自动生成

你不需要自己编。解析小说时，程序按**阅读顺序**给每一章编号：

```
c003
│└─┴─ 三位数字，从 001 开始，左补零
└──── 固定字母 c（chapter）
```

所以 ID 依次是 `c001`、`c002`、`c003` …… `c010`、`c011` ……
**注意 `c010` 不是 `c10`**（三位补零，这样排序才对）。

### 1.2 怎么知道一本书有哪些 ID

```bash
conda activate qwen3-tts
cd /Users/wanghaoxin/Documents/AustinProject/NovelTTS
noveltts parse --list
```

输出（你的这本书的真实结果）：

```
《Too Many Losing Heroines! Vol. 8》 共 10 章

ID             字数        预估  标题
------------------------------------------------------------------------
c001        3,600     04:18  Foreword
c002       48,499     57:46  Loss 1: The Purest of Intentions
c003        2,490     03:00  Intermission: Appeal to Emotion
c004       35,590     42:24  Loss 2: Burdened by the Weight
c005        3,816     04:36  Intermission: I'll Have You Know I'm Kind of a Big Deal
c006       53,795   1:04:04  Loss 3: Crimson Rhapsody
c007        2,527     03:03  Intermission: Everybody, Do Your Share
c008       25,179     30:00  Loss 4: Enemy of Women Who?
c009       15,282     18:13  Epilogue: Onward to Adulthood
c010        1,267     01:31  Afterword
------------------------------------------------------------------------
合计        192,045   3:48:54
```

左起第一列就是 ID。**`--chapter` 参数就填这一列。**

### 1.3 好消息：`--chapter` 支持四种写法

不必死记 ID，下面四种都能选中同一章（已实测）：

| 写法 | 例子 | 说明 |
|---|---|---|
| 完整 ID | `--chapter c003` | 最明确，脚本里推荐用这种 |
| 数字序号 | `--chapter 3` | 等价于 `c003`，偷懒可用 |
| 完整标题 | `--chapter "Intermission: Appeal to Emotion"` | 标题含空格/冒号要加引号 |
| 标题里的唯一片段 | `--chapter Crimson` | 只要**唯一匹配**就行；匹配到多章会报错并列出候选 |

```bash
noveltts synth --chapter 3              # 等价于 c003
noveltts synth --chapter Crimson        # 自动选中 c006 Loss 3: Crimson Rhapsody
noveltts synth --chapter c999           # ✗ 报错：找不到章节 'c999'
```

### 1.4 另外两个容易混的参数

| 参数 | 含义 | 例子 |
|---|---|---|
| `--chapter c003` | 选**哪一章** | 章节 ID / 序号 / 标题 |
| `--segments 3,5,8-12` | 选**这一章里的哪几段** | 片段序号，支持逗号与区间 |
| `--limit 5` | 只做**前 5 段**（试听用） | 数字，0 表示全部 |

片段序号看这个命令的输出：

```bash
noveltts segment --chapter c003 --preview 20
```

```
=== c003 Intermission: Appeal to Emotion
片段总数 44（叙述 20 / 对话 23 / 标题 1）
字符数：最短 7 / 平均 56.6 / 最长 111
  [   1] heading     27字 | Intermission: Appeal to Emotion
  [   2] narration   89字 | AFTER SCHOOL, IN THE STUDENT COUNCIL ROOM, ...
  [   3] narration   75字 | Treasurer Sakurai Hiroto stopped organizing shelves ...
```

方括号里的 `1`、`2`、`3` 就是 `--segments` 要填的序号。

---

## 2. 本地操作（这台 Mac）：完整流程

### 2.0 每次都先做这两步

```bash
conda activate qwen3-tts
cd /Users/wanghaoxin/Documents/AustinProject/NovelTTS
```

> ⚠️ **`noveltts` 命令只在 `qwen3-tts` 环境里存在。**
> 你的终端默认在 `base` 环境，所以直接敲 `noveltts` 会提示
> `command not found`——这是你遇到的问题的根因，`conda activate` 之后就正常了。
>
> 验证一下：
> ```bash
> conda activate qwen3-tts && which noveltts
> # 应输出 /Users/wanghaoxin/yes/envs/qwen3-tts/bin/noveltts
> ```

### 2.1 第 1 步：放小说 / 检查环境

把 `.epub` 或 `.txt` 放进 **`sources/`** 目录（这个目录已存在）：

```bash
cp ~/Downloads/某本小说.epub sources/
noveltts info
```

`noveltts info` 会列出素材、检查依赖、显示模型路径：

```
素材文件：
  - 某本小说.epub (8.0 MB)

依赖检查：
  音频读写 : OK - ok
  音频编码 : OK - ffmpeg

模型设置（synth / run 才会真正加载）：
  路径     : Qwen3-TTS-12Hz-1.7B-Base
  设备     : auto  精度: bfloat16
```

### 2.2 第 2 步：解析并切章

```bash
noveltts parse --list
```

**产物**：`output/work/chapters.json`

里面是按章切好的结构化数据。**看输出的 ID 列**，记下你要做的章节。

如果切章结果不对（比如把版权页也当成一章），可以：
```bash
noveltts parse --list -v        # 看详细的丢弃日志
```
然后在 `config.yaml` 的 `parse` 段里调整 `min_chapter_chars` 与
`skip_title_patterns`。

### 2.3 第 3 步（**强烈建议**）：先看洗稿效果

```bash
noveltts normalize --chapter c003
```

**这一步不写任何文件，只是让你预览模型将收到什么文本。** 输出形如：

```
[c003 第2段]
  原: “Jumping the gun a little, no? You still have a month left,” he said.
  新: "Jumping the gun a little, no? You still have a month left," he said.

[c003 第4段]
  原: She'd already qualified for the 1500 that morning.
  新: She'd already qualified for the fifteen hundred that morning.

洗稿统计：共 20 处改写
  quote                          20 处
      '“' -> '"'
```

**重点看有没有"改错"的地方。** 发现某条规则改多了，就在 `config.yaml`
的 `text_rules` 里关掉它：

```yaml
text_rules:
  large_numbers: 1100      # 把 1500 读成 fifteen hundred；不想这样改就设成 999999
  roman_numerals: false    # 关掉罗马数字转换
```

### 2.4 第 4 步：切分成片段

```bash
noveltts segment --chapter c003 --preview 20
```

**产物**：更新 `output/work/chapters.json`，并写入 `output/work/segments/c003.json`
（片段 ID → 文本的索引，后面"失败重跑"要用）。

看输出里的"片段总数 / 字符数 / 最长"，确认切分粒度合理：

```
片段总数 44（叙述 20 / 对话 23 / 标题 1）
字符数：最短 7 / 平均 56.6 / 最长 111
```

如果片段太长/太短，调 `config.yaml`：

```yaml
segment:
  max_chars: 120      # 调小 → 片段更短、更稳但接缝更多
  min_chars: 24       # 调大 → 减少碎片
```

### 2.5 第 5 步：**先试听 5 段**（不要一上来跑整本）

```bash
noveltts synth --chapter c003 --limit 5
```

这一步会**真正加载模型**（约 5 秒）并调用 GPU/CPU 推理。首次运行正常输出：

```
加载模型 Qwen3-TTS-12Hz-1.7B-Base（device=mps, dtype=bfloat16, attn=sdpa）…
模型加载完成，用时 5.3s
构建音色 prompt（参考音频=doubao.wav, 模式=icl）…
音色 prompt 构建完成，用时 3.9s
[5/5] 100.0%  音频 00:26  预计剩余 00:00
完成 c003 Intermission: Appeal to Emotion | 新生成 5 | 跳过 0 | 音频 00:26 | 耗时 01:03 | RTF 2.52
```

**产物**：

```
output/audio/segments/c003/
├── c003-0001.wav ~ c003-0005.wav     ← 逐段音频（24kHz 单声道）
└── manifest.json                      ← 清单：时长、参数、错误记录
```

听一下：
```bash
afplay output/audio/segments/c003/c003-0002.wav
```

**确认音色、语速、发音都满意再往下走。** 不满意就调 `config.yaml`：

| 症状 | 调整 |
|---|---|
| 不像参考音色 | 检查 `voice.ref_text` 是否与 `doubao.wav` 内容**逐字一致** |
| 语速太快/太慢 | **用 `--rate`**（见下方「调节语速」）；或换一个语速合适的参考音频 |
| 语气太平 | 试 `generation.temperature: 0.8`（默认跟随模型配置） |
| 某些词念错 | 用 `text_rules.custom_replacements` 改写那个词 |

### 2.6 调节语速（变速不变调）

最常用的手段是 `--rate`，**不需要改配置文件**：

```bash
noveltts synth --chapter c003 --limit 5 --rate 1.2    # 快 20%
noveltts synth --chapter c003 --limit 5 --rate 0.85   # 慢 15%
noveltts run --rate 1.15                              # 全书都快 15%
```

| `--rate` | 效果 |
|---|---|
| `1.0` | 原速（默认） |
| `1.15` | 略快，最自然 |
| `1.25` | 明显快，通勤听刚好 |
| `0.85` | 放慢，适合难懂的内容 |
| `1.5` | 很快，开始有轻微金属感 |

实现方式是**相位声码器**（librosa），**音高保持不变**——不是简单加快采样率，
所以不会变成"快进的唐老鸭"。建议范围 `0.5~2.0`，微调（0.9~1.15）最自然。

想固定下来就写进 `config.yaml`：

```yaml
audio:
  speech_rate: 1.15
```

语速会记录在 `manifest.json` 的 `settings.speech_rate` 里，半年后也知道这版是怎么生成的。

> **改了语速要重新生成吗？** 要。语速是在生成后对音频做的处理，
> 已存在的片段会被跳过。所以调语速时请配合 `--force`：
> ```bash
> noveltts synth --chapter c003 --limit 5 --rate 1.2 --force
> ```

### 2.7 第 6 步：生成整章

```bash
noveltts synth --chapter c003
```

**这一段可以随时 Ctrl-C。** 已经生成的片段都在磁盘上，重新运行会自动跳过
（输出里会显示 `跳过 N`），只补没做完的部分。

跑完看汇总：
```
完成 c003 Intermission: Appeal to Emotion | 新生成 39 | 跳过 5 | 音频 03:00 | 耗时 06:40 | RTF 2.22
```

### 2.8 第 7 步：拼接 + 压缩

```bash
noveltts concat --chapter c003      # 片段 → 章节 wav
noveltts encode --chapter c003      # wav → m4a
```

**产物**：

```
output/audio/c003.wav      ← 无损母版（约 48 KB/秒，3 分钟的章节约 8 MB）
output/audio/c003.m4a      ← 压缩成品（约 1/8 体积，手机直接能放）
```

### 2.9 或者：一条命令搞定第 4~7 步

```bash
noveltts run --chapter c003
```

等价于 `segment → synth → concat → encode` 全流程，输出里会分段打印：

```
[合成] c003 Intermission: Appeal to Emotion | 新生成 44 | 跳过 0 | 音频 03:00 | 耗时 06:40
[拼接] c003 | 44 段 | 时长 03:02 | 体积 8.5 MB | 耗时 1.2s
[压缩] c003.m4a | 8.5 MB -> 1.1 MB（13%，省 7.4 MB） | 耗时 3.1s
```

### 2.10 做完整本书

```bash
noveltts run                                  # 全书所有章节
noveltts concat --merge-book --book-name "Too Many Losing Heroines Vol 8"
                                              # 再把各章合并成一个整本 wav
```

---

## 3. 云端部署（AutoDL + NVIDIA GPU）

完整步骤在 **[cloud-deployment.md](cloud-deployment.md)**，包含：

* AutoDL 实例的选择建议（GPU 型号 / 显存 / 镜像 / 数据盘）
* 用 VSCode Remote-SSH 连上去之后，逐条敲哪些命令
* NVIDIA 专属优化开关（程序按 `config.yaml` 自动判断，一般不用手改）
* **三层验证清单**：环境对不对、项目传全没传全、能不能真的出声
* 长跑建议（nohup / tmux、断点续跑、成本估算）

一句话预告：**云端不需要改任何代码**，只改 `config.yaml` 的 `generation` 段：

```yaml
generation:
  device: cuda              # auto 也会自动选到 cuda，写死更明确
  batch_size: auto          # 按显存自动决定（24G 建议 4~8）
  attn_implementation: auto # CUDA 上自动用 flash_attention_2
```

## 4. 命令速查表

```bash
conda activate qwen3-tts
cd /Users/wanghaoxin/Documents/AustinProject/NovelTTS
```

| 命令 | 作用 | 加载模型 | 主要产物 |
|---|---|---|---|
| `noveltts info` | 环境/素材/模型总览 | 否 | 无 |
| `noveltts parse --list` | 解析并切章，列出章节 ID | 否 | `output/work/chapters.json` |
| `noveltts normalize --chapter c003` | **洗稿预览** | 否 | 无（只打印） |
| `noveltts segment --chapter c003 --preview 20` | 语意切分 | 否 | `output/work/segments/c003.json` |
| `noveltts index --chapter c003` | 生成片段文本索引 | 否 | 同上 |
| `noveltts synth --chapter c003 --limit 5` | 试听 5 段 | **是** | `output/audio/segments/c003/*.wav` |
| `noveltts synth --text "Hello there."` | 一键生成一小段 | **是** | `output/audio/segments/adhoc/` |
| `noveltts synth --chapter c003 --segments 3,5,8-12` | 只做指定片段 | **是** | 同上 |
| `noveltts synth --chapter c003` | 整章 | **是** | 同上 |
| `noveltts synth --chapter c003 --retry-failed` | 只补失败的片段 | **是** | 同上 |
| `noveltts concat --chapter c003` | 拼接成章 | 否 | `output/audio/c003.wav` |
| `noveltts concat --merge-book` | 各章合并成整本 | 否 | `output/audio/full_book.wav` |
| `noveltts encode --chapter c003` | 压缩 | 否 | `output/audio/c003.m4a` |
| `noveltts run --chapter c003` | 一键全流程 | **是** | 以上全部 |
| `noveltts run` | 全书全流程 | **是** | 以上全部 |

常用附加参数：

```bash
--force              # 已存在的片段也重新生成（续跑时不要加）
--dry-run            # 用假引擎跑通链路，不加载模型，产出正弦波
--language Chinese   # 覆盖语言（默认 English）
--device cuda        # 覆盖设备：auto/cuda/mps/cpu
--rate 1.2           # 语速倍率（变速不变调），1.0 原速
--batch-size auto    # 批大小：auto 按显存自推，或写数字
-v / -q              # 更详细 / 更安静的日志
```

---

## 5. 产物目录说明

跑完之后 `output/` 长这样：

```
output/
├── work/                              中间产物（删了可重建，不影响音频）
│   ├── chapters.json                  解析结果：全部章节 + 全部片段
│   ├── text/c001.txt … c010.txt       每章纯文本（肉眼校对用）
│   └── segments/c001.json …           片段 ID → 文本索引
│
└── audio/                             成品
    ├── segments/c001/
    │   ├── c001-0001.wav …            逐段音频（24kHz / 单声道 / 16bit）
    │   └── manifest.json              清单：每段时长、尝试次数、错误、生成参数
    ├── c001.wav                       章节母版（无损）
    ├── c001.m4a                       章节成品（压缩，手机可放）
    └── full_book.wav                  整本合并（如果用了 --merge-book）
```

**`manifest.json` 值得看一眼**，它记录了这份音频是怎么生成的：

```json
{
  "chapter_id": "c003",
  "segment_count": 44,
  "done_count": 44,
  "failed_count": 0,
  "total_duration": 180.5,
  "settings": {
    "model_path": "Qwen3-TTS-12Hz-1.7B-Base",
    "device": "mps", "dtype": "bfloat16",
    "ref_audio": "doubao.wav", "voice_mode": "icl",
    "language": "English", "sample_rate": 24000
  },
  "records": [ { "segment_id": "c003-0001", "duration": 2.32, "error": null }, … ]
}
```

半年后回头听，也知道这一版是用什么参数生成的。

---

## 6. 故障排查

### 6.1 `command not found: noveltts`

**原因**：命令安装在 conda 环境 `qwen3-tts` 里，而你的终端默认在 `base`。

```bash
conda activate qwen3-tts
which noveltts        # /Users/wanghaoxin/yes/envs/qwen3-tts/bin/noveltts
```

想一劳永逸，可以在 `~/.zshrc` 末尾加一个别名：

```bash
echo 'alias noveltts="conda run -n qwen3-tts noveltts"' >> ~/.zshrc
```

### 6.2 `RuntimeError: cannot cache function '__o_fold': no locator available`

这是 **librosa + numba 的缓存路径问题**，会导致 `import qwen_tts` 直接失败，
模型根本加载不起来。

**已经在本项目里修好了**（`noveltts/__init__.py` 里的 `ensure_numba_cache_dir()`）：
程序会自动检测 librosa 的 `__pycache__` 是否可写，不可写时把 `NUMBA_CACHE_DIR`
指向临时目录。你**不需要做任何事**，正常敲命令就行。

如果哪天在别的项目里又遇到，手工绕过：

```bash
export NUMBA_CACHE_DIR=/tmp/numba_cache
```

### 6.3 想确认硬件加速有没有生效

每次 `synth` / `run` 在**加载模型之前**都会打印一段「硬件与加速方案」：

```
硬件与加速方案：
设备：cuda:NVIDIA GeForce RTX 3090 | 显存 24.0G（可用 23.8G） | 算力 sm_86 | torch 2.4.0 | cuda 12.4
配置：device=cuda:0 dtype=bfloat16 attn=flash_attention_2 batch_size=8
开关：tf32=True cudnn_benchmark=True empty_cache=True clear_every=50
判定依据：
  - 批大小 auto -> 8：...
  - 注意力 auto -> flash_attention_2（可用（flash-attn 2.7.0））
  ...
```

要点：

* `attn=sdpa` 而不是 `flash_attention_2` -> flash-attn 没装或版本不匹配（不影响使用，只是慢）
* `batch_size=1` 但显存很大 -> 配置里把 `batch_size` 写死成 1 了，改成 `auto`
* `tf32=False` 但卡是 30 系以上 -> 检查 `compute_capability` 是否 >= 8.0

所有开关都在 `config.yaml` 的 `generation` 段，写 `auto` 让程序判断：

```yaml
generation:
  batch_size: auto          # 按显存自推（8G→1, 24G→6, 32G→9）
  attn_implementation: auto # CUDA 且装了 flash-attn 就用
  tf32: auto                # 算力 >= 8.0 时开启
  cudnn_benchmark: auto
  clear_cache_every: 0      # 0 = auto（CUDA/MPS 每 50 段清理一次）
```

### 6.4 `RuntimeError: 引擎已关闭` 或显存不足（OOM）

* 把 `config.yaml` 的 `generation.batch_size` 改回 `1`
* 把 `max_new_tokens` 从 1024 降到 512
* 云端确认 `flash-attn` 装上了（能省不少显存）

### 6.5 生成的音频有的段特别长 / 有怪声

单段失败或退化会记在 `manifest.json` 的 `error` 字段。定位并重做：

```bash
conda run -n qwen3-tts python -c "
import json
m = json.load(open('output/audio/segments/c003/manifest.json'))
for r in m['records']:
    if r['error'] or r['duration'] > 30: print(r)
"
noveltts synth --chapter c003 --segments 42 --force    # 只重做那一段
```

### 6.6 断点了 / SSH 断了

**直接重跑同一条命令**，已生成的片段自动跳过。看到 `跳过 N` 就是正常续跑。

### 6.7 想从头再来

```bash
rm -rf output              # 删掉全部中间产物与音频
noveltts run               # 重新跑
```

`.wav` 文件都很大，注意磁盘空间：整本 3 小时 49 分的无损音频约 **1.6 GB**，
压缩成 m4a 后约 **200 MB**。

---

## 7. 建议的执行顺序（一句话版）

```bash
conda activate qwen3-tts
cd /Users/wanghaoxin/Documents/AustinProject/NovelTTS

noveltts info                                  # ① 确认环境
noveltts parse --list                          # ② 看章节 ID
noveltts normalize --chapter c003              # ③ 检查洗稿（不写文件）
noveltts synth --chapter c003 --limit 5        # ④ 试听 5 段
afplay output/audio/segments/c003/c003-0002.wav

noveltts run                                   # ⑤ 满意后跑全书（可随时中断续跑）
noveltts concat --merge-book --book-name "TM LH Vol 8"   # ⑥ 合并整本
```

第 ④ 步是**最关键的一步**——花 1 分钟，避免白跑 9 小时。
