# 完整工作流：从电子书到有声书

> **这份文档是交付总纲。** 目标：任何人（或 AI agent）照着走一遍，
> 就能从零复现整个项目 —— 从拿到一本电子书，到租云端 GPU 生成，
> 到拿到最终成品。
>
> 配套文档：
> [user-guide.md](user-guide.md)（本地操作细节）·
> [cloud-deployment.md](cloud-deployment.md)（云端部署细节）·
> [API-REFERENCE.md](API-REFERENCE.md)（接口规范，供开发 GUI）·
> [git-workflow.md](git-workflow.md)（版本管理）
>
> 项目实际成果（已跑通）：一本 19 万字的轻小说 → **4 小时 29 分有声书**，
> 3220 个片段，零失败，云端 3090 耗时 1 小时 36 分。

---

## 0. 一页速览

```text
① 准备        本地或云端装依赖，把 epub/txt 放进 sources/
                    ↓
② 解析        noveltts parse --list          → 得到章节 ID（c001, c002 …）
                    ↓
③ 洗稿预览    noveltts normalize --chapter cXXX   → 确认数字/缩写改写合理
                    ↓
④ 切分        noveltts segment --chapter cXXX     → 每段 40~120 字符
                    ↓
⑤ 试听        noveltts synth --chapter cXXX --limit 5   → 先听 5 段！
                    ↓
⑥ 全书生成    setsid nohup noveltts run &    → 可断线，自动续跑
                    ↓
⑦ 拼接+压缩   自动完成；整本合并用 concat --merge-book
                    ↓
⑧ 交付        下载 m4a 到本地，关机/释放实例
```

**唯一必读的注意事项：第 ⑤ 步不要跳过。** 花 2 分钟试听，避免白跑 2 小时。

---

## 1. 阶段一：环境准备

### 1.1 本地（macOS / Apple Silicon）

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

pip install torch numpy soundfile PyYAML    # macOS 上 torch 自动带 MPS 支持
pip install -U qwen-tts

cd /path/to/NovelTTS
pip install -e . --no-deps                  # 安装 noveltts 命令

# 可选：压缩音频需要 ffmpeg（macOS 没有时会自动退回 afconvert）
brew install ffmpeg
```

**验证**：

```bash
which noveltts                    # 应指向 conda env 的 bin
noveltts info                     # 依赖检查应显示 OK
python run_tests.py | tail -2     # 应显示全部通过
```

> ⚠️ **`noveltts` 命令只在该 conda 环境里存在。**
> 终端默认在 `base` 环境时会报 `command not found` —— 先 `conda activate qwen3-tts`。

### 1.2 云端（AutoDL + NVIDIA GPU）

推荐配置：**RTX 3090 / 4090，显存 ≥ 24G**，镜像选 **PyTorch 2.x + CUDA 12.x**。

```bash
# 连上实例后
cd /root/autodl-tmp

pip install -U qwen-tts numpy soundfile PyYAML
apt-get update && apt-get install -y ffmpeg

# 模型：官方 qwen3-tts 镜像通常已自带，否则下载
python -c "import os; print(os.path.isdir('/root/models/Qwen3-TTS-12Hz-1.7B-Base'))"
# 若为 False：
# modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir /root/models/Qwen3-TTS-12Hz-1.7B-Base
```

> **关键**：AutoDL 的 conda 不在**非交互式 SSH** 的 PATH 里。
> 脚本里请用绝对路径 `/root/miniconda3/bin/python`，
> 或先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`。

详细步骤与验证清单见 **[cloud-deployment.md](cloud-deployment.md)**。

---

## 2. 阶段二：导入电子书

### 2.1 放文件

```bash
cp ~/Downloads/某本小说.epub sources/
noveltts info                   # 确认它被识别到
```

支持 `.epub`（**推荐**）、`.txt`、`.md`。

> **为什么推荐 EPUB**：EPUB 的目录（nav/ncx）能给出完整章节标题，
> 切章质量明显更好。TXT 只能靠正则识别标题行，遇到排版怪的书会切错。

### 2.2 解析并查看章节

```bash
noveltts parse --list
```

输出（真实例子）：

```
《Too Many Losing Heroes! Vol. 8》 共 10 章

ID             字数        预估  标题
------------------------------------------------------------------------
c001        3,600     04:18  Foreword
c002       48,499     57:46  Loss 1: The Purest of Intentions
c003        2,490     03:00  Intermission: Appeal to Emotion
...
------------------------------------------------------------------------
合计        192,045   3:48:54
```

**第一列就是章节 ID**，后面所有 `--chapter` 参数都用它。

### 2.3 章节 ID 命名规范（重要）

```
c003
│└─┴─ 三位数字，从 001 开始，左补零
└──── 固定字母 c (chapter)
```

* 按**阅读顺序**自动编号：`c001`、`c002` …… `c010`、`c011`
* **是 `c010` 不是 `c10`**（补零是为了排序正确）

`--chapter` 支持**四种写法**（都已实测）：

| 写法 | 例子 | 说明 |
|---|---|---|
| 完整 ID | `--chapter c003` | 最明确，脚本里推荐 |
| 数字序号 | `--chapter 3` | 等价于 c003 |
| 完整标题 | `--chapter "Loss 1: The Purest of Intentions"` | 含空格要加引号 |
| 标题唯一片段 | `--chapter Crimson` | 唯一匹配即可；匹配多章会报错并列出候选 |

### 2.4 产物

```
output/work/chapters.json      解析结果（含全部章节与片段）
output/work/text/c001.txt …    每章纯文本，便于肉眼校对
```

---

## 3. 阶段三：洗稿（文本规范化）

### 3.1 先预览，不要盲改

```bash
noveltts normalize --chapter c003
```

这一步**不写任何文件**，只是让你看模型将收到什么文本：

```
[c003 第4段]
  原: She'd already qualified for the 1500 that morning.
  新: She'd already qualified for the fifteen hundred that morning.

洗稿统计：共 20 处改写
  quote                          20 处
      '“' -> '"'
```

### 3.2 改写规则一览（实测有效的部分）

| 原文 | 洗稿后 | 说明 |
|---|---|---|
| `Vol. 8` | `Volume 8` | 缩写展开 |
| `Mr. Tanaka` | `Mister Tanaka` | 头衔 |
| `1st place` | `first place` | 序数词 |
| `870-120` | `eight hundred seventy to one hundred twenty` | 数字区间读成 "A to B" |
| `1990-1995` | `1990 to 1995` | 年份区间 |
| `$1,500.00` | `one thousand five hundred dollars` | 金额（`$` 不能留） |
| `1500` | `fifteen hundred` | 口语化大数 |
| `Chapter VII` | `Chapter seven` | 罗马数字（`I`/`V` 永不转换，避免误伤代词） |
| `word—word` | `word, word` | 破折号按上下文分四种情形 |

**刻意不改的**（改了反而更糟）：

| 原文 | 为什么不动 |
|---|---|
| `"Huh?"` | 模型念得很好 |
| `Class 2-C` | 班级名，读 "two C" 是自然的 |
| `H2O` / `B29` | 字母数字混合标识符 |
| `3:30 p.m.` | 时刻 |
| `400-meter` | 连字符复合词 |
| `"No. Not everything."` | `No.` 后不接数字时是**否定词**，不能展开成 Number |

### 3.3 调整规则

```yaml
# config.yaml 的 text_rules 段
text_rules:
  large_numbers: 1100        # 调大发阈值可减少改动；设 999999 等于关闭
  roman_numerals: false      # 整条关掉
  abbreviations:
    "dept.": "department"    # 大小写敏感
  custom_replacements:
    "\\bNukumizu\\b": "Nukumizu"    # 最高优先级
```

---

## 4. 阶段四：语意切分

```bash
noveltts segment --chapter c003 --preview 20
```

**为什么必须切分**：TTS 模型单次生成有 token 上限，且文本越长越容易
漏字、音色漂移、语速失控。切成"一到两句话"能显著提升稳定性。

**切分策略**（四层，由粗到细）：

1. **句级切分** —— 遇 `. ! ? 。！？` 断句，但跳过缩写点号（`Mr.` / `5 p.m.`）、
   小数（`1.5`）、以及**引号内部**的标点
2. **超长句再切** —— 超过硬上限时按从句边界（逗号、分号、连接词）切
3. **聚合成片段** —— 相邻句子贪心合并到 `max_chars`；叙述与对话不混段
4. **收尾修正** —— 过短碎片与邻居合并

**实测**：这本书 19 万字 → **3220 个片段**，中位数 53 字符，
最长 199，**无一超硬上限**。

调整阈值：

```yaml
segment:
  max_chars: 120      # 调小 → 更稳但接缝更多
  min_chars: 24       # 调大 → 减少碎片
  hard_max_chars: 200
```

---

## 5. 阶段五：试听（**最关键的一步**）

```bash
noveltts synth --chapter c003 --limit 5
```

> 命令行会自动完成"切分"，你不用手动做。
> 但**如果你用 Python API**，`parse_book` 只给段落、不给片段，
> 必须先 `segmenter.segment_chapter(chapter)`，否则会报
> `章节 cXXX 还没有切分`。详见 [API-REFERENCE.md](API-REFERENCE.md)。

**这一步才会加载模型。** 首次运行会看到：

```
硬件与加速方案：
设备：cuda:NVIDIA GeForce RTX 3090 | 显存 23.6G（可用 23.3G） | 算力 sm_86 | torch 2.8.0 | cuda 12.8
配置：device=cuda:0 dtype=bfloat16 attn=flash_attention_2 batch_size=12
开关：tf32=True cudnn_benchmark=True empty_cache=True clear_every=50
判定依据：
  - 批大小 auto -> 12：总显存 23.6G × 0.8 与可用 23.3G 取小 = 18.8G；...
  - 注意力 auto -> flash_attention_2（可用（flash-attn 2.8.3））
  - TF32 auto -> 开启（CUDA，sm_86）
模型加载完成，用时 2.0s
音色 prompt 构建完成，用时 1.9s
完成 c003 | 新生成 5 | 跳过 0 | 音频 00:26 | 耗时 00:23 | RTF 0.87
```

**验收要点**：

| 检查项 | 期望 |
|---|---|
| `attn=` | 有 flash-attn 时应为 `flash_attention_2` |
| `batch_size=` | 24G 卡约 12~13；Mac 应为 1 |
| `tf32=` | 30 系及以上应为 `True` |
| `voice_mode=` | 应为 `icl`（若显示 `xvec` 说明 `ref_text` 没填） |
| 音色 | 听一下像不像参考音频 |

听：

```bash
afplay output/audio/segments/c003/c003-0002.wav      # macOS
# 或 VSCode 远程窗口里右键 → Download
```

**不满意就现在调**：

| 症状 | 调整 |
|---|---|
| 语速不合适 | `--rate 1.15`（见 §6.2） |
| 语气太平 | `generation.temperature: 0.8` |
| 不像参考音色 | 检查 `voice.ref_text` 是否与参考音频**逐字一致** |
| 某些词念错 | `text_rules.custom_replacements` |

---

## 6. 阶段六：全书生成

### 6.1 启动（可断线）

```bash
cd /root/autodl-tmp/NovelTTS       # 云端；本地则是项目目录

setsid nohup noveltts run > run_full.log 2>&1 < /dev/null &
```

| 部分 | 作用 | 少了会怎样 |
|---|---|---|
| `setsid` | 新建**独立会话** | SSH 断开时进程收 SIGHUP 被杀 |
| `nohup` | 忽略 SIGHUP | 同上（双保险） |
| `> run_full.log 2>&1` | 日志落文件 | 日志进已关闭的终端，看不到 |
| `< /dev/null` | 切断 stdin | 可能读到 EOF 异常退出 |
| `&` | 后台 | 占着终端 |

**验证真脱离了**：

```bash
ps -eo pid,sid,etime,cmd | grep "[n]oveltts run"
# PID 与 SID 相同 → 已独立会话
```

实测：整个 1.5 小时里反复断连重连，进程始终存活。

### 6.2 调语速（变速不变调）

```bash
noveltts synth --chapter c003 --rate 1.2 --force      # 快 20%
noveltts run --rate 1.15                              # 全书都用这个语速
```

> ⚠️ **必须加 `--force`**：语速是**生成后**对音频做的处理，
> 已存在的片段会被跳过，语速**不会**变。这是最容易踩的坑。

| `--rate` | 听感 |
|---|---|
| `1.0` | 原速 |
| `1.1` ~ `1.15` | 略快，**最自然**（推荐起点） |
| `1.25` | 明显快，通勤听刚好 |
| `1.5` | 很快，开始有轻微金属感 |
| `0.85` | 放慢 |

实现用相位声码器，**音高保持不变**（不会变"快进的唐老鸭"）。
建议范围 `0.5 ~ 2.0`。

### 6.3 监控进度

```bash
# 本地 Mac 上（推荐）：一键面板
./scripts/remote_status.sh
./scripts/remote_status.sh --watch 60

# 或直接看远程日志
ssh autodl 'tail -f /root/autodl-tmp/NovelTTS/run_full.log' \
  | grep --line-buffered -E "完成 |失败|降级"
```

**正常与异常判断**：

| 指标 | 正常 | 异常 → 怎么办 |
|---|---|---|
| 片段数 | 持续增长 | 长时间不涨 → 看日志是否卡住 |
| 失败 / 降级 | `0 / 0` | 出现数字 → 跑完用 `synth --retry-failed` |
| RTF | `0.3 ~ 0.6`（3090） | 明显变大 → GPU 被抢或显存不足 |
| GPU 利用率 | 30~60% | 长期 0% → 进程可能已死 |

### 6.4 中途断了怎么办

**直接重跑同一条命令。** 已生成的片段会自动跳过（日志显示 `跳过 N`）。

每段生成完立刻落盘，所以**任何时刻中断，损失只有当时那一段**。

### 6.5 性能参考（RTX 3090 实测）

| 方式 | 音频 | 耗时 | RTF |
|---|---|---|---|
| 逐段生成 | 25.4 s | 47 s | 1.74 |
| **批量 5 段** | 25.4 s | 13.4 s | **0.53** |
| 整章 44 段 | 3:13 | 1:43 | 0.53 |
| **整本 3220 段** | **4:21:14** | **1:35:44** | **0.37** |

批处理是最大性能来源（3.3 倍）。整本约 2 小时可完成。

---

## 7. 阶段七：拼接与压缩

`noveltts run` **已经自动**给每章做了拼接 + 压缩。
若还要**整本合并成一个文件**：

```bash
noveltts concat --merge-book --book-name "我的有声书"
noveltts encode                     # 压缩整本
```

### 产物清单

```
output/audio/
├── segments/c001/…c010/                    逐段音频（断点续跑的基石）
│   └── manifest.json                       每段时长/错误/生成参数
├── c001.wav … c010.wav                     各章拼接版（无损母版）
├── c001.m4a … c010.m4a                     各章压缩版
├── 我的有声书.wav                           整本拼接版（无损）
└── 我的有声书.m4a                           整本压缩版  ← 交付主文件
```

### 体积参考（4.48 小时音频）

| 类型 | 体积 | 说明 |
|---|---|---|
| 整本 wav | 738 MB | 无损母版，约 **48 KB/秒** |
| 整本 m4a | 131 MB | AAC 68kbps，约 1/5.6 |
| 各章 wav ×10 | ~740 MB | 合计 |
| 逐段 wav ×3220 | 726 MB | 确认成品后可删 |

### 下载到本地

```bash
# 在本地 Mac 上执行
scp "autodl:/root/autodl-tmp/NovelTTS-output/audio/我的有声书.m4a" ~/Downloads/
```

或用 VSCode 远程窗口右键 → Download。

### 数据库清理

```bash
# 确认成品听着没问题后
rm -rf output/audio/segments          # 占最大头，可按需重新生成
```

---

## 8. 阶段八：交付与关机

### 8.1 关机前必须确认

```bash
# 本地：确认成品已落地且完整
ls -lh ~/Downloads/*.m4a
shasum -a 256 ~/Downloads/我的有声书.m4a
ssh autodl 'sha256sum /root/autodl-tmp/NovelTTS-output/audio/我的有声书.m4a'
# 两边哈希一致 = 文件完整
```

| 云端内容 | 本地有吗 | 关机影响 |
|---|---|---|
| **整本 m4a**（成品） | ✅ 必须有 | — |
| 各章 m4a | 可选 | 保留实例可取回 |
| 各章 wav（无损） | 可选 | 只能重新生成 |
| 逐段 wav | 不需要 | 可重新生成 |
| 模型 14 G | 不需要 | 官方镜像自带 |
| 代码 + 文档 | ✅ 在 git 里 | — |

### 8.2 关机 vs 释放

| 操作 | 数据盘 | 计费 | 建议 |
|---|---|---|---|
| **关机**（Stop） | 保留 | 停止 GPU 计费 | ✅ 推荐，下次开机还能取回 |
| **释放**（Release） | 清空 | 完全停止 | 只在确定不再需要时用 |

**结论：成品下载并校验通过后，可以放心关机。**

---

## 9. 复现检查清单

给接手的 agent / 协作者：

```bash
# ── 代码完整性 ──────────────────────────────
git clone <repo> && cd NovelTTS
git log --oneline --decorate | head       # 应看到 v1.0.0 ~ v1.3.x 的标签
python run_tests.py | tail -2             # 应全绿（457 个用例，不需要 GPU）

# ── 环境完整性 ──────────────────────────────
noveltts info                             # 依赖检查 OK
python -c "import qwen_tts; print('ok')"  # TTS 库可用

# ── 数据完整性 ──────────────────────────────
ls sources/                               # 有 epub/txt
ls assets/ref/                            # 有参考音频（音色复现所必需）
python -c "
from noveltts.config import AppConfig
c = AppConfig.load('config.yaml')
print('模型:', c.generation.model_path)
print('参考音频:', c.voice.ref_audio, c.project.root_path.joinpath(c.voice.ref_audio).is_file())
"

# ── 端到端（不烧算力）──────────────────────
noveltts parse --list
noveltts segment --chapter c001 --preview 5
noveltts run --chapter c001 --limit 3 --dry-run    # 假引擎，产出正弦波
```

三层全过，就说明项目完整、环境正确、可以正式跑。

---

## 10. 常见陷阱汇总

按"踩坑概率 × 代价"排序：

| 陷阱 | 症状 | 正确做法 |
|---|---|---|
| **不试听就跑全书** | 2 小时后才发现音色/语速不对 | 先 `--limit 5` 听一下 |
| **改语速不加 `--force`** | 语速毫无变化 | `--rate X --force` |
| **直接前台跑全书** | SSH 一断白跑 2 小时 | `setsid nohup ... &` |
| **改 `config.yaml` 放本机路径** | 更新代码后被冲掉 | 写 `config.local.yaml`（不入库） |
| **产出放系统盘** | AutoDL 系统盘只有 30G，很快满 | `output_dir` 指向数据盘 |
| **忘填 `voice.ref_text`** | 音色相似度打折 | 逐字填入参考音频内容 |
| **`noveltts: command not found`** | 终端在 base 环境 | `conda activate qwen3-tts` |
| **`import qwen_tts` 报 numba 缓存错** | 模型完全加载不起来 | 本项目已自动修复（`ensure_numba_cache_dir`） |
| **云端 conda 不在 PATH** | `python: command not found` | 用 `/root/miniconda3/bin/python` 绝对路径 |

---

## 11. 项目结构

```
NovelTTS/
├── config.yaml                 通用默认配置（入库）
├── config.local.yaml           本机覆盖（不入库）★ 机器相关设置写这里
├── config.local.example.yaml   本机配置模板
├── noveltts/                   源码（8237 行）
│   ├── cli.py                  命令行入口
│   ├── config.py               配置系统
│   ├── models.py               数据结构
│   ├── epub_parser.py          EPUB/TXT 解析
│   ├── normalize.py            洗稿
│   ├── segmenter.py            语意切分
│   ├── engine.py               TTS 引擎封装（唯一 import torch）
│   ├── accelerator.py          硬件探测与加速方案
│   ├── synthesis.py            合成编排（批处理/续跑/失败隔离）
│   ├── audio_io.py             音频读写、变速、归一
│   ├── concat.py               流式拼接
│   ├── encode.py               压缩
│   └── utils.py                工具
├── tests/                      测试（457 个用例）
├── scripts/
│   ├── deploy_remote.sh        一键部署到远程
│   └── remote_status.sh        进度面板
├── assets/ref/                 参考音频（入库，音色复现所必需）
├── sources/                    小说放这里（不入库，版权）
├── docs/                       文档
│   ├── overview.md             ← 本文档
│   ├── user-guide.md
│   ├── cloud-deployment.md
│   ├── API-REFERENCE.md
│   └── git-workflow.md
└── run_tests.py                零依赖测试运行器
```

---

## 12. 关键参数速查

### 命令行

```bash
noveltts <命令> [选项]

# 全局
-c/--config <路径>         配置文件（默认 ./config.yaml）
-v / -q                    日志详细 / 安静

# 常用
--chapter <ID|序号|标题>    选章节
--source <路径>            指定素材文件
--rate <倍数>              语速（1.0 原速），需配 --force
--limit <N>                只做前 N 段（试听）
--segments <3,5,8-12>      只做指定片段
--force                    已存在也重新生成
--dry-run                  假引擎，不加载模型
--device auto|cuda|mps|cpu 覆盖设备
--batch-size auto|<N>      批大小
--language <English|...>   覆盖语言
```

### 配置文件（优先级：默认 < config.yaml < config.local.yaml < 命令行）

```yaml
project:      source_dir / work_dir / output_dir / cache_dir
parse:        min_chapter_chars / skip_title_patterns / exclude_paragraph_patterns
text_rules:   enabled / language / abbreviations / large_numbers / …
segment:      max_chars / min_chars / hard_max_chars / …
voice:        ref_audio / ref_text / mode(icl|xvec)
generation:   device / dtype / batch_size / tf32 / max_new_tokens / …
audio:        speech_rate / normalize / min_gap_seconds / …
encode:       format(m4a|mp3|opus|flac) / bitrate / keep_wav
```

完整字段与默认值见 **[API-REFERENCE.md](API-REFERENCE.md)**。

---

## 13. 交付物清单

一次完整生成的交付物：

| 交付物 | 位置 | 说明 |
|---|---|---|
| **整本有声书** | `output/audio/<书名>.m4a` | 主交付物，手机可直接播放 |
| 无损母版 | `output/audio/<书名>.wav` | 需要重新压缩时用 |
| 分章版本 | `output/audio/cXXX.m4a` | 按章收听 |
| 生成清单 | `output/audio/segments/*/manifest.json` | 每段时长/错误/参数，可追溯 |
| 项目代码 | git 仓库 | 含全部文档与测试 |
| 本工作流文档 | `docs/overview.md` | ← 你正在看的 |
| 接口规范 | `docs/API-REFERENCE.md` | 供二次开发 |
