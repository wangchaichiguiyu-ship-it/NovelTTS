# NovelTTS —— 小说转有声书流水线

把 EPUB / TXT 小说转成有声书：**洗稿**（修正模型念不对的缩写与数字）→ **语意切分** →
**Qwen3-TTS 声音克隆**逐段合成 → **流式拼接**成章 → **压缩**成 m4a。

> ### 📘 文档导航
>
> | 我想… | 看这里 |
> |---|---|
> | **完整走一遍**：从导入电子书到拿到成品 | **[docs/overview.md](docs/overview.md)** ← 交付总纲，含复现检查清单 |
> | **开发可视化界面** / 查接口与参数 | **[docs/API-REFERENCE.md](docs/API-REFERENCE.md)** ← 全部接口规范 |
> | 在自己电脑上操作 | [docs/user-guide.md](docs/user-guide.md) |
> | 上云端 GPU 跑 | [docs/cloud-deployment.md](docs/cloud-deployment.md) |
> | 接手项目 / 看已知弱点 | [docs/TECH-DEBT.md](docs/TECH-DEBT.md) |
> | 想做成拖拽式软件 | [docs/PRODUCTIZATION.md](docs/PRODUCTIZATION.md) |
> | 参与开发 / 版本管理 | [docs/git-workflow.md](docs/git-workflow.md) |
>
> 本文档偏重**设计与原理**。

**已实测跑通的成果**：一本 19 万字的轻小说 → **4 小时 29 分有声书**，
3220 个片段，**零失败**，RTX 3090 上耗时 **1 小时 36 分**（RTF 0.37）。

针对 **16 GB 统一内存的 Mac** 设计，也适用于云端 NVIDIA GPU（30 系及以上）。

支持**语速调节**（变速不变调）、**CUDA 自适应加速**（批大小/FlashAttention/TF32 按实测显存自动决定）、
**断点续跑**（任何时刻中断，损失只有当前那一段）。

```text
EPUB/TXT ──parse──▶ 章节 ──normalize──▶ 洗稿文本 ──segment──▶ 片段
                                                              │
                              output/audio/segments/*.wav ◀──synth──┘
                                        │
                            章节 wav ◀──concat──┘
                                        │
                            章节 m4a ◀──encode──┘
```

---

## 1. 快速开始

### 1.1 环境

```bash
conda activate qwen3-tts

# 必需依赖（文本处理 + 音频读写）
pip install numpy soundfile

# 真正合成语音所需的依赖（本机已装好则可跳过）
pip install torch qwen-tts transformers accelerate

# 可选：YAML 配置解析（不装也能跑，内置了极简解析器）
pip install PyYAML
```

### 1.2 安装本项目

```bash
cd /path/to/NovelTTS
pip install -e .          # 之后可以直接用 noveltts 命令
# 或者不安装，用 python -m noveltts.cli 也能跑
```

### 1.3 三步跑通

```bash
noveltts info                              # 看看素材、依赖、模型路径是否就绪
noveltts synth --chapter c003 --limit 5    # 只生成 5 段试听（会加载模型）
noveltts concat --chapter c003             # 拼成一个 wav
```

**确认听感没问题后**，再跑整章/整本：

```bash
noveltts run --chapter c003      # 一章：合成 + 拼接 + 压缩
noveltts run                     # 全书
```

> **不想烧算力先验证链路？** 所有合成命令都支持 `--dry-run`：
> 用假引擎产出正弦波，不加载任何模型，但**磁盘布局、manifest、断点续跑
> 的行为与真实跑批完全一致**。
> ```bash
> noveltts run --chapter c003 --dry-run
> ```

---

## 2. 目录结构

```text
NovelTTS/
├── config.yaml                   # 唯一的配置文件（中文注释逐项说明）
├── pyproject.toml
├── run_tests.py                  # 零依赖测试运行器（不需要 pytest）
├── noveltts/
│   ├── cli.py                    # 命令行入口（noveltts <子命令>）
│   ├── config.py                 # 配置系统（YAML -> 强类型 dataclass）
│   ├── models.py                 # Chapter / Segment / Manifest 数据结构
│   ├── epub_parser.py            # EPUB / TXT 解析与切章
│   ├── normalize.py              # 洗稿（缩写、数字、符号）
│   ├── segmenter.py              # 语意切分
│   ├── engine.py                 # Qwen3-TTS 封装（唯一 import torch 的地方）
│   ├── accelerator.py            # 硬件探测与加速方案（CUDA/MPS/CPU 自适应）
│   ├── synthesis.py              # 合成编排：逐段落盘、断点续跑、失败隔离
│   ├── audio_io.py               # 音频读写、响度归一、静音裁切
│   ├── concat.py                 # 流式拼接
│   ├── encode.py                 # ffmpeg / afconvert 压缩
│   └── utils.py                  # 日志、原子写、统计
├── tests/                        # 测试（332 个用例，全部不加载模型）
├── sources/                      # 放你的 epub / txt
└── output/                       # 所有产物（可整目录删除重跑）
    ├── work/                     # 中间产物：章节 JSON、片段 JSON、纯文本
    └── audio/
        ├── segments/<章节>/*.wav # 逐段音频（断点续跑的基石）
        ├── segments/<章节>/manifest.json
        ├── <章节>.wav            # 拼接成品
        └── <章节>.m4a            # 压缩成品
```

---

## 3. 命令一览

| 命令 | 作用 | 会加载模型？ |
|---|---|---|
| `noveltts info` | 素材、章节规模、依赖、模型配置概览 | 否 |
| `noveltts parse --list` | 解析 EPUB/TXT 并切章 | 否 |
| `noveltts normalize --chapter c002` | **洗稿预览**：看模型实际会收到什么文本 | 否 |
| `noveltts segment --chapter c002 --preview 60` | 语意切分并查看每个片段 | 否 |
| `noveltts index --chapter c002` | 生成片段文本索引（失败重跑依赖它） | 否 |
| `noveltts synth --chapter c002` | 合成整章 | **是** |
| `noveltts synth --text "..."` | **一键生成一小段** | **是** |
| `noveltts synth --chapter c002 --segments 3,5,8-12` | 只合成指定片段 | **是** |
| `noveltts synth --chapter c002 --limit 5` | 只合成前 5 段（试听） | **是** |
| `noveltts synth --chapter c002 --retry-failed` | 只重跑失败的片段 | **是** |
| `noveltts concat --chapter c002` | 拼接成章节 wav | 否 |
| `noveltts concat --merge-book` | 所有章再合并成整本 | 否 |
| `noveltts encode --chapter c002` | 压缩成 m4a | 否 |
| `noveltts run --chapter c002` | 一键：切分 + 合成 + 拼接 + 压缩 | **是** |

常用参数：

```bash
--device auto|cuda|mps|cpu     # 覆盖设备
--batch-size auto              # auto 按显存自推，或写数字（Mac 16G 请写 1）
--rate 1.2                     # 语速倍率（变速不变调），1.0 原速
--language English|Chinese|... # 覆盖语言
--force                        # 已生成的片段也重新生成
--dry-run                      # 用假引擎跑通链路，不加载模型
-v / -q                        # 更详细 / 更安静的日志
```

---

## 4. 三种典型用法

### 4.1 一键生成一章的有声书

```bash
noveltts run --chapter c002
```

输出：

```text
output/audio/segments/c002/c002-0001.wav   ... 逐段音频
output/audio/segments/c002/manifest.json       清单（时长/错误/参数）
output/audio/c002.wav                          无损章节母版
output/audio/c002.m4a                          压缩成品（约 1/8 体积）
```

### 4.2 专门生成一小段

```bash
# 命令行直接给文本
noveltts synth --text "The 1500 meter race began at 3 p.m." 

# 输出到 output/audio/segments/adhoc/，可直接播放
```

或者在 Python 里：

```python
from noveltts.config import AppConfig
from noveltts.synthesis import SynthesisSession

config = AppConfig.load("config.yaml")
with SynthesisSession(config) as session:            # 这里才加载模型
    records = session.synthesize_text(
        "She said it was over. He did not believe her.",
        chapter_id="try1",
    )
    for r in records:
        print(r.audio_path, f"{r.duration:.2f}s")
```

### 4.3 只重跑失败的片段

长跑批里偶尔会有一两段失败（显存抖动、极端文本）。**不需要整章重来**：

```bash
noveltts synth --chapter c002 --retry-failed
```

失败信息记录在 `manifest.json` 的 `error` 字段里，重跑时只处理这些片段。

---

## 5. 洗稿：模型念不对的地方怎么改

`noveltts normalize --chapter c002` 可以逐条看到改动。规则都在 `config.yaml`
的 `text_rules` 段里，可以逐项关掉对比效果。

以下是**实测有效**的几类改动（都是这本轻小说里真实出现的）：

| 原文 | 洗稿后 | 说明 |
|---|---|---|
| `Vol. 8` | `Volume 8` | 缩写展开 |
| `Mr. Tanaka` | `Mister Tanaka` | 头衔 |
| `approx.` | `approximately` | 句末时自动补回句号 |
| `1st place` | `first place` | 序数词 |
| `870-120` | `eight hundred seventy to one hundred twenty` | 比分读成 "A to B" |
| `1990-1995` | `1990 to 1995` | 年份区间 |
| `$1,500.00` | `one thousand five hundred dollars` | 金额（`$` 不能留） |
| `1500` | `fifteen hundred` | 口语化大数 |
| `100 percent` | `one hundred percent` | 量词数字 |
| `Chapter VII` | `Chapter seven` | 罗马数字（`I`/`V` 永不转换，避免误伤代词） |
| `…—` | 逗号 / 句号 / 删除 | 破折号按上下文分四种情形处理 |
| `20 percent` | `twenty percent` | 量词白名单 |
| `"Huh?"` | 保持 | **刻意不动**：模型念得很好，改了反而出戏 |
| `Class 2-C` | 保持 | 班级名：读 "two C" 是自然的 |
| `H2O` / `B29` | 保持 | 字母数字混合标识符 |
| `3:30 p.m.` | 保持 | 时刻 |
| `400-meter` | 保持 | 连字符复合词 |
| `"No. Not everything."` | 保持 | `No.` 后面不接数字时是**否定词**，不能展开成 Number |

### 设计原则

1. **保守**：只在高度确定的场景改写。拿不准的一律不动。
2. **可审计**：每次运行都打印规则命中统计（规则名 / 次数 / 样例），
   你可以据此判断某条规则是不是改多了。
3. **可回滚**：每个片段都保留 `source_text`（洗稿前原文），排查"这句话怎么念错了"时直接对照。

### 样板文字过滤

版权页、ISBN 行、"All rights reserved" 这类内容读出来毫无意义，而且夹杂数字与
全大写字母，最容易念错。`config.yaml` 的 `parse.exclude_paragraph_patterns`
会**逐段**丢掉它们（不是整章丢弃，所以后记末尾夹着的版权声明不会带走正文）：

```yaml
parse:
  exclude_paragraph_patterns:
    - '\bAll rights reserved\b'
    - '^Copyright\s*(?:\u00a9|\(c\))?\s*\d{4}'
```

默认值刻意保守——**实测这本书 19 万字一条都没命中**，不会误伤正文。
想看丢了什么：`noveltts parse -v`。

保护机制：如果某个单元的所有段落都命中规则，程序会**保留原文并给出警告**，
宁可多读几句版权声明，也不会让一章凭空消失。

### 词表怎么加

```yaml
text_rules:
  abbreviations:
    "dept.": "department"      # 大小写敏感，想同时匹配就写两条
  numeric_context_abbreviations:
    "No.": "Number"            # 只在后面跟数字时展开
  custom_replacements:
    "\\bNukumizu\\b": "Nukumizu"   # 最高优先级，最后执行
```

---

## 6. 语意切分：为什么要切、怎么切

### 为什么

TTS 模型每次生成有 `max_new_tokens` 上限，而且**文本越长越容易出问题**：
漏字、后半段音色漂移、语速越读越快。把一次合成限制在"一到两句话"
可以显著提升稳定性，代价只是拼接时多了几个接缝。

### 怎么切

分四层，由粗到细：

1. **句级切分** —— 遇到 `. ! ? 。！？` 断句，但会跳过：
   - 缩写点号（`Mr.` / `5 p.m.` / `U.S.`）
   - 小数（`1.5`）与省略号（`...`）
   - **引号内部**的标点（`"Stop. Now."` 不会被劈成两半）
2. **超长句再切分** —— 超过硬上限时按从句边界（逗号、分号、连接词）切开
3. **聚合成片段** —— 相邻句子贪心合并到 `max_chars` 为止；
   叙述与对话不混在一个片段里，避免模型用旁白语气念台词
4. **收尾修正** —— 过短的碎片与邻居合并

实测这本 19 万字的小说：**3220 个片段**，平均 60 字符，
中位数 53 字符，最长 199 字符，**没有一个超过硬上限**。

```bash
noveltts segment --chapter c002 --preview 20
```

---

## 7. 硬件加速（CUDA / MPS / CPU 自适应）

同一份代码要跑在三种差别很大的环境上，所以把「硬件能力 → 优化开关」的判断
集中在 `noveltts/accelerator.py`，并**全部实测**，不做任何假设。

```yaml
generation:
  device: auto              # cuda > mps > cpu
  batch_size: auto          # 按显存推导
  attn_implementation: auto # CUDA 且装了 flash-attn 就用
  tf32: auto                # 算力 >= 8.0 时开启
  cudnn_benchmark: auto
  clear_cache_every: 0      # 0 = auto（CUDA/MPS 每 50 段）
  oom_batch_downgrade: true # OOM 时自动减半批大小
```

每次开跑前会打印判定依据，一眼看清有没有生效：

```
硬件与加速方案：
设备：cuda:NVIDIA GeForce RTX 3090 | 显存 24.0G（可用 23.8G） | 算力 sm_86 | torch 2.4.0 | cuda 12.4
配置：device=cuda:0 dtype=bfloat16 attn=flash_attention_2 batch_size=8
开关：tf32=True cudnn_benchmark=True empty_cache=True clear_every=50
判定依据：
  - 批大小 auto -> 8：总显存 24.0G × 0.8 与可用 23.8G 取小 = 19.2G；...
  - 注意力 auto -> flash_attention_2（可用（flash-attn 2.7.0））
  - TF32 auto -> 开启（CUDA，sm_86）
```

### 批处理是最大的性能来源（实测 3.3 倍）

同一次前向处理多段文本，而不是一段一次。RTX 3090 实测：

| 方式 | 音频 | 耗时 | RTF | 峰值显存 |
|---|---|---|---|---|
| 逐段 | 25.4 s | 47 s | 1.74 | 4.7 G |
| 批量 5 段 | 25.4 s | 13.4 s | **0.53** | 7.7 G |
| 整章 44 段（批 12） | 3 分 13 秒 | 1 分 43 秒 | **0.53** | — |

整本书（3.82 小时音频）由约 6.6 小时降到 **约 2.0 小时**。

整批失败时会**降级为逐段重试**，把某一段的问题与同批其它段隔离——
批处理快的代价是一荣俱荣一损俱损，所以必须有这条兜底路径。

### 批大小怎么推出来的

固定占用 4.5 GB（bf16 权重 + 说话人编码器 + 运行时），每多一个批内样本约 2.5 GB，
再乘 0.8 的安全系数，并且取「总显存」与「**可用**显存」的较小值——
后者很重要：实例上可能已有别的进程占着卡，只看总显存会乐观到 OOM。

| 显存 | 推荐批大小 |
|---|---|
| 8 G | 2 |
| 12 G | 5 |
| 16 G | 7 |
| **24 G（3090/4090）** | **13** |
| 32 G+ | 16（上限） |

（固定占用按 4.5 G，每样本按 1.2 G 估算；后者来自 3090 实测的 0.75 G/样本，
再留约 60% 余量。）

**Mac 用户注意**：统一内存架构下显存与内存是同一块物理内存，批处理只会更危险，
所以 `auto` 在非 CUDA 设备上一律返回 1。

### 三个容易踩的坑（都已处理）

1. **配置写了 `flash_attention_2` 但没装** —— 会在**加载 3.4 GB 权重之前**探测并
   降级到 `sdpa`，而不是等加载完再崩。
2. **老卡上开 TF32** —— TF32 需要算力 ≥ 8.0（Ampere/30 系起），更老的卡上显式关闭。
3. **OOM 后整章反复失败** —— 自动把批大小减半并继续，不会一个过大的批次拖垮整章。

## 8. 语速调节（变速不变调）

```bash
noveltts synth --chapter c003 --limit 5 --rate 1.2   # 快 20%
noveltts run --rate 1.15                             # 全书都快 15%
```

```yaml
audio:
  speech_rate: 1.15    # 1.0 原速，1.2 快 20%，0.85 慢 15%
```

用**相位声码器**（librosa）实现，**音高保持不变**——不是简单加快采样率，
所以不会变成"快进的唐老鸭"。测试里直接断言了 220 Hz 主频在 0.8/1.25/1.5
三档速率下都不动。

* 建议范围 `0.5 ~ 2.0`，微调（`0.9 ~ 1.15`）最自然
* librosa 已是 `qwen_tts` 的依赖，**不引入新依赖**
* 缺 librosa 时安全退回原速并警告，不会让整段音频报废
* 语速写进 `manifest.settings.speech_rate`，保证音频可复现

## 9. 16 GB 内存友好的设计

这是本项目最花心思的部分。**内存峰值的决定因素不是音频总长，而是同时持有多少东西。**

| 环节 | 朴素做法的问题 | 本项目的做法 | 实测 |
|---|---|---|---|
| 模型权重 | 每段重新 load（3.4 GB × N） | 全程只加载一次，复用同一个实例 | — |
| 音色 prompt | 每段重算参考音频编码 | `create_voice_clone_prompt` 算一次，逐段复用 | — |
| 逐段合成 | 攒够一批再落盘，内存持续膨胀 | **每生成一段立刻写盘并断引用**；每批后 `gc.collect()` + 清设备缓存 | — |
| 章节拼接 | `np.concatenate(所有片段)`：500 段 ≈ 480 MB，全书 > 3 GB | 按 1 秒的块流式读→写，峰值 = 一个块 | **15.5 分钟音频 → 峰值 12 MB** |
| 解析/切分 | 把每段音频留在内存里 | 只处理当前章节，整本书的文本对象共 **2.1 MB** | 3220 段 |
| 导入开销 | `import` 就拉起 torch | 全部延迟导入；`parse`/`normalize`/`segment` 零依赖 | torch 未加载 |

### 具体建议

* **`batch_size` 保持 1**。Mac 的显存和内存是同一块物理内存，批处理会把多段
  音频同时留在显存里，得不偿失。云端显存充裕（≥24 GB）再考虑调到 4~8。
* **`max_new_tokens` 不要盲目调大**。默认已从模型卡的 2048 降到 **1024**：
  片段平均只有 60 字符，1024 个 token 对应约 85 秒音频，余量充足。
  调大只会让显存预留变多。
* **跑批可以随时 Ctrl-C**。已生成的片段都在磁盘上，重跑会自动跳过。
* **音频用 `PCM_16` 落盘**，比 float32 省一半磁盘和 IO，听感无损。

---

## 10. 云端部署

> 完整步骤（AutoDL 选实例、VSCode Remote SSH、三层验证清单、成本估算）
> 见 **[docs/cloud-deployment.md](docs/cloud-deployment.md)**。以下是要点。

```bash
# 1) 拉代码 + 装依赖
pip install -e ".[tts,yaml]"

# 2) 把模型下到本地（推荐，避免运行时联网）
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base

# 3) 装 FlashAttention 2（仅 CUDA，能显著降显存）
pip install -U flash-attn --no-build-isolation
```

`config.yaml` 里改成：

```yaml
generation:
  device: cuda              # 或保持 auto，程序会自己选到 cuda
  batch_size: auto          # 按显存自推：24G→6, 32G→9（也可写死数字）
  attn_implementation: auto # CUDA 且装了 flash-attn 就用它
  tf32: auto                # 30 系（sm_86）自动开启
```

其余命令完全一致。**建议先在云端跑 `--dry-run` 验证链路**，再正式开跑。

---

## 11. 参考音频与音色克隆

模型是 `Qwen3-TTS-12Hz-1.7B-Base`（**声音克隆**），必须提供参考音频：

```yaml
voice:
  ref_audio: doubao.wav     # 参考音频
  ref_text: ""              # ← 请填上这段音频里实际说的话
  mode: icl                 # icl = 参考音频+文本（质量最好）
```

> **关于 `ref_text`**：当前是空的。留空会自动退回 `x_vector_only` 模式
> （只用说话人向量，**相似度会略低**，且日志会给出警告）。
> 想让克隆效果最好，请听一遍参考音频，把它说的内容**逐字**填进 `ref_text`。
> 文本越准，克隆越像。

想换音色：把任意 3~15 秒的干净人声 wav 放进项目目录，改 `ref_audio` 与
`ref_text` 即可。参考音频越"干净"（无背景音乐、无混响），效果越好。

---

## 12. 测试

### 零依赖运行（不需要 pytest）

```bash
python run_tests.py              # 全部
python run_tests.py normalize    # 只跑文件名含 normalize 的
python run_tests.py -v           # 打印每个用例名
```

### 用 pytest

```bash
pip install pytest
pytest tests/ -q
```

**457 个用例，全部通过，且全部不加载模型。** 涉及推理的用例统一用
`FakeSynthesisEngine` 注入——它产出正弦波，但会让整条流水线（含磁盘布局、
manifest、断点续跑、失败隔离）真实地跑一遍。

覆盖范围：

| 文件 | 用例 | 覆盖内容 |
|---|---|---|
| `test_normalize.py` | 80 | 洗稿规则逐条 |
| `test_accelerator.py` | 53 | **CUDA 分支**：批大小/注意力/TF32/缓存/OOM |
| `test_segmenter.py` | 50 | 句级切分边界、聚合、跨段合并 |
| `test_audio.py` | 55 | 响度归一、静音裁切、流式拼接、压缩命令、**变速不变调** |
| `test_parser.py` | 42 | EPUB/TXT 切章、标题去重、样板文字过滤 |
| `test_cli.py` | 50 | 全部子命令 + dry-run 全链路 + **本机配置覆盖** |
| `test_config_utils.py` | 45 | 配置嵌套构造、内置 YAML 解析器、numba 缓存兜底 |
| `test_engine_contract.py` | 29 | **对 qwen_tts 的调用契约**（见下） |
| `test_pipeline.py` | 53 | 合成会话、断点续跑、失败隔离、**批处理**、语速 |

### 引擎契约测试：把最贵的失败提前到本机

`test_engine_contract.py` 用 `sys.modules` 注入一个**假的 `qwen_tts`**，
把真实调用签名逐条断言下来（`from_pretrained`、`create_voice_clone_prompt`、
`generate_voice_clone` 的参数名与形状）。

这解决了一个很实际的痛点：参数名写错（`dtype` 写成 `torch_dtype` 之类）
本机因为**不能加载模型**而不会暴露，到了云端才会在加载完 3.4 GB 权重之后炸掉
——那是最贵的失败方式。有了这层契约测试，这类错误在本机秒级发现，
且有 pytest 时就自动变成普通单测，不需要额外维护。

测试里包含多个**回归用例**，对应开发中实际踩过的坑，例如：

* `B29` 里的 `29` 被误改写（正则 `|` 把断言切开了）
* `"No. Not everything."` 被改成 `Number.`
* `870-120` 被当成一个数字读成 "eight hundred seventy thousand one hundred twenty"
* 短句 `"Hello there."` 被误判成标题，导致整段不再合并
* 纯静音被"归一化"时增益冲向无穷大

---

## 13. 常见问题

**Q：`info` 报"音频读写不可用 - 缺少依赖"**
A：`pip install numpy soundfile`。

**Q：报 `RuntimeError: cannot cache function '__o_fold': no locator available`**
A：**本项目已自动修复**，你不需要做任何事。

这条报错来自 librosa 的 `@numba.jit(cache=True)`：numba 在**导入期**要为它选一个
缓存位置，首选是 `<librosa>/core/__pycache__`。该目录不可写时（conda 环境在只读
位置、或缺写权限）numba 找不到任何可用位置，直接抛异常——而这发生在
`import librosa` 期间，于是 **整个 `qwen_tts` 都导入不了，模型根本加载不起来**。

`noveltts/__init__.py` 里的 `ensure_numba_cache_dir()` 会在导入包时自动检测这个
目录是否可写，不可写就把 `NUMBA_CACHE_DIR` 指到一个可写的临时目录。
如果你在**别的项目**里遇到同样问题，手工绕过：`export NUMBA_CACHE_DIR=/tmp/numba_cache`。

**Q：没有 ffmpeg，能压缩吗？**
A：macOS 上没有 ffmpeg 时会自动改用系统自带的 `afconvert`（能出 m4a）。
想装 ffmpeg：`brew install ffmpeg`。

**Q：中途断了怎么办？**
A：**直接重跑同一条命令**。已生成的片段会被跳过（`skip_existing: true`），
只补没做完的部分。想强制重生成加 `--force`。

**Q：一章里有几段失败了**
A：`noveltts synth --chapter <id> --retry-failed`。

**Q：拼接出来的停顿忽长忽短**
A：程序已经做了两件事：片段尾部自带静音时把间隔缩短为 0.35 倍；句子被切断时
再乘 0.5。如果还想微调，改 `audio.min_gap_seconds`，或用
`noveltts concat --chapter c002 --gap 0.25` 临时覆盖。

**Q：想换语言（比如念中文原文）**
A：`noveltts synth --chapter c001 --language Chinese`，
并把 `text_rules.language` 设为 `zh` 或 `auto`。

**Q：我不想让它读版权页 / 广告页**
A：`config.yaml` 里有 `parse.exclude_paragraph_patterns`（逐段过滤），
默认已包含版权与 ISBN 规则。想确认丢了什么用 `noveltts parse -v`。
另外 `parse.min_chapter_chars` 会整体丢弃过短的单元（版权页、图片页）。

**Q：想听某一段单独的效果**
A：`noveltts synth --chapter c002 --segments 42 --force`，
然后在 `output/audio/segments/c002/c002-0042.wav` 里找。

---

## 14. 产物清单

```text
output/
├── work/
│   ├── chapters.json               # 解析结果（含全部分片）
│   ├── text/c001.txt ...           # 每章纯文本（方便肉眼校对）
│   └── segments/c001.json ...      # 片段 id -> 文本索引
└── audio/
    ├── segments/c001/
    │   ├── c001-0001.wav ...       # 逐段音频（24 kHz / 单声道 / PCM_16）
    │   └── manifest.json           # 清单：路径、时长、尝试次数、错误、生成参数
    ├── c001.wav                    # 章节母版（无损）
    └── c001.m4a                    # 压缩成品
```

`manifest.json` 里记录了生成时的关键参数（模型路径、精度、语言、采样参数、
参考音频、归一化设置），所以**半年后回头听，也知道这版音频是怎么来的**。

---

## 15. 许可与致谢

* TTS 模型：[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（Apache-2.0）
* 本项目代码：MIT

请自行确认所处理文本的版权与授权。
