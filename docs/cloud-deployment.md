# 云端部署指南（AutoDL + VSCode Remote SSH）

> 配套文档：[user-guide.md](user-guide.md)（命令规范与本地流程）、
> [git-workflow.md](git-workflow.md)（版本管理）。
>
> **目标**：从一台全新的 GPU 实例开始，到能稳定跑完整本书。
> 每一步都给出**验证方式**——环境对不对、项目传全没传全、能不能真的出声。

---

## 0. 先明确一件事：云端不需要改代码

本项目的加速逻辑全在 `noveltts/accelerator.py`，它按**实测硬件**决定方案。
到了云端你只需要改 `config.yaml` 里的几个值（或者什么都不改，`auto` 会自己判断）：

```yaml
generation:
  device: auto              # 会自动选中 cuda
  batch_size: auto          # 按显存推导（30G 卡约 8~9）
  attn_implementation: auto # CUDA 且装了 flash-attn 就用它
  tf32: auto                # 30 系（sm_86）自动开启
```

程序在**加载模型之前**会打印一份判定依据，形如：

```
硬件与加速方案：
设备：cuda:NVIDIA GeForce RTX 3090 | 显存 24.0G（可用 23.8G） | 算力 sm_86 | torch 2.4.0 | cuda 12.4
配置：device=cuda:0 dtype=bfloat16 attn=flash_attention_2 batch_size=8
开关：tf32=True cudnn_benchmark=True empty_cache=True clear_every=50
判定依据：
  - 批大小 auto -> 8：总显存 24.0G × 0.8 与可用 23.8G 取小 = 19.2G；扣除固定占用 4.5G 后约 14.7G，按每样本 2.5G 估算得 8
  - 注意力 auto -> flash_attention_2（可用（flash-attn 2.7.0））
  - TF32 auto -> 开启（CUDA，sm_86）
  - cudnn.benchmark auto -> 开启（仅 CUDA）
  - 缓存清理间隔 auto -> 每 50 段清理
```

**看到这段就说明加速已生效。** 如果 `attn=sdpa` 而你装了 flash-attn，看第 6 节。

---

## 1. 选实例（AutoDL）

| 项目 | 建议 | 说明 |
|---|---|---|
| GPU | **3090 / 4090 / A5000** 以上 | 显存 ≥ 24G。本项目模型只占约 4.5G，显存越大越能开大批大小 |
| 显存 | **≥ 24G** | 30G 左右很充裕；24G 也能把 `batch_size` 开到 8 |
| 镜像 | **PyTorch 2.x + CUDA 12.x** | 选带 PyTorch 的基础镜像，省掉装 torch 的功夫 |
| 数据盘 | **建议开** | 模型权重约 4G，音频输出整本约 200MB~2G，系统盘经常不够 |
| 计费 | 按量计费 | 先跑通再考虑包时 |

> **重要**：实例的**系统盘**在关机后可能被重置，**数据盘**才持久。
> 把项目与模型都放数据盘（AutoDL 通常是 `/root/autodl-tmp`）。

### 验证

```bash
nvidia-smi                    # 应看到你的卡型号与显存
df -h /root/autodl-tmp        # 数据盘应有几十~几百 G 可用
```

---

## 2. 用 VSCode Remote-SSH 连上去

1. VSCode 安装扩展 **Remote - SSH**（`ms-vscode-remote.remote-ssh`）
2. `F1` → `Remote-SSH: Add New SSH Host`
3. 按 AutoDL 控制台的指引填，通常是：

```sshconfig
Host autodl
    HostName connect.xxx.seetacloud.com
    Port 12345
    User root
```

4. `F1` → `Remote-SSH: Connect to Host` → `autodl`
5. 连上后 `F1` → `Open Folder` → `/root/autodl-tmp/NovelTTS`

连接成功后，VSCode 左下角会显示 `SSH: autodl`。
**之后所有终端命令都在这个远程终端里跑**（`` Ctrl+` `` 打开）。

> 端口与密码在 AutoDL 控制台的「自定义服务 / SSH 登录」里，每次开机可能变。

---

## 3. 配环境（逐条敲，每步都有验证）

### 3.1 基础依赖

AutoDL 的 PyTorch 镜像通常已带 torch、numpy、conda。先确认：

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

期望输出类似：

```
torch 2.4.0 | CUDA True | NVIDIA GeForce RTX 3090
```

⚠️ `CUDA False` 说明镜像选错了，回控制台换一个带 CUDA 的镜像。

### 3.2 建一个干净环境（可选但推荐）

镜像自带的 base 环境可能和 `qwen-tts` 的依赖打架，建议新建：

```bash
conda create -n noveltts python=3.12 -y
conda activate noveltts

# 重新装 torch（按你的 CUDA 版本，CUDA 12.4 为例）
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> 如果图省事，也可以直接用镜像的 base 环境，跳过这一步。
> 但一旦出现依赖冲突，新建环境是最省时间的解法。

### 3.3 装项目依赖

```bash
pip install -U qwen-tts
pip install numpy soundfile PyYAML

# ffmpeg（压缩音频用）
apt-get update && apt-get install -y ffmpeg
```

### 验证（第 1 层：环境）

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import qwen_tts; print('qwen_tts ok')"
python -c "import soundfile; print('soundfile ok')"
ffmpeg -version | head -1
```

四条都通过才继续。

### 3.4 装 FlashAttention 2（强烈建议）

它能显著降低显存占用并提速。**没装也能跑**（会自动退回 `sdpa`），
但装了之后批大小能开更大。

```bash
# 官方推荐方式；内存小/核少时用 MAX_JOBS 限制并行编译，避免 OOM
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

> ⚠️ 从源码编译通常要 **10~30 分钟**，很吃 CPU 与内存。
> 如果你的实例 CPU 核数少、内存 < 96G，`MAX_JOBS=4` 是必须的。
>
> 想跳过编译，可以找与你的 **torch 版本 + CUDA 版本 + Python 版本**三者都匹配的
> 预编译 wheel（社区有维护，见 [flash-attention](https://github.com/Dao-AILab/flash-attention)
> 仓库的 Releases 与各 wheelhouse 项目）。**版本必须三者都对上**，否则会 import 失败。

### 验证

```bash
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
```

失败也不影响使用——程序会自动用 `sdpa`，只是慢一些。

### 3.5 下载模型（**推荐下到数据盘**）

```bash
cd /root/autodl-tmp

pip install -U modelscope       # 国内快
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --local_dir ./Qwen3-TTS-12Hz-1.7B-Base

# 或用 HuggingFace（必要时先设镜像：export HF_ENDPOINT=https://hf-mirror.com）
# huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
```

### 验证

```bash
ls -lh /root/autodl-tmp/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors
# 应看到约 3.x GB 的文件
du -sh /root/autodl-tmp/Qwen3-TTS-12Hz-1.7B-Base
```

---

## 4. 上传项目

### 4.1 一键部署（推荐）

仓库里带了 `scripts/deploy_remote.sh`，它做的正是下面 4.2~4.3 的事，
但**会自动排除 `config.yaml` 与 `config.local.yaml`**——这是关键：
手敲 tar 很容易把 `config.yaml` 一起覆盖过去，把云端改好的
`model_path` 冲回 HuggingFace 仓库 id，而云端连不上 HF，模型直接加载失败。

```bash
./scripts/deploy_remote.sh                    # 默认 host=autodl
./scripts/deploy_remote.sh --tests            # 传完顺便在远程跑测试
./scripts/deploy_remote.sh --dry-run          # 只打包，看会传什么
```

它只传代码/测试/文档（约 200 KB），不传小说、参考音频、模型与配置。
首次部署仍需按 4.2 把小说与参考音频传上去。

### 4.2 手动打包（在**你自己的 Mac** 上执行）

```bash
cd /Users/wanghaoxin/Documents/AustinProject/NovelTTS

# 打包含源码、配置、参考音频；排除模型权重、输出、git 历史
tar --exclude=output \
    --exclude=.git \
    --exclude=config.yaml \
    --exclude=config.local.yaml \
    --exclude='Qwen3-TTS*' \
    --exclude='__pycache__' \
    --exclude='.graphflow-cache' \
    -czf ~/noveltts.tar.gz \
    noveltts tests docs config.yaml pyproject.toml README.md run_tests.py \
    assets sources/.gitkeep sources/README.md

ls -lh ~/noveltts.tar.gz      # 约 13 MB（主要是参考音频）
```

> **参考音频必须一起上传**（`assets/ref/doubao.wav`）。没有它无法克隆音色。

### 4.3 上传

在 **VSCode 的远程窗口** 里操作最省事：直接把 `noveltts.tar.gz`
从本地文件管理器拖进远程的文件树（VSCode 会自动传输）。

或者用命令行（本地执行）：

```bash
scp -P <端口> ~/noveltts.tar.gz root@<HostName>:/root/autodl-tmp/
```

> AutoDL 也支持把文件传到 `/root/autodl-tmp` 后用 JupyterLab 的文件面板上传。

### 4.4 解压与安装

```bash
mkdir -p /root/autodl-tmp/NovelTTS
tar -xzf /root/autodl-tmp/noveltts.tar.gz -C /root/autodl-tmp/NovelTTS
cd /root/autodl-tmp/NovelTTS

pip install -e . --no-deps      # 安装 noveltts 命令
```

### 验证（第 2 层：项目完整性）

```bash
cd /root/autodl-tmp/NovelTTS

# ① 命令可用
noveltts --version

# ② 参考音频在位
ls -lh assets/ref/doubao.wav

# ③ 测试全绿（440 个用例，不需要 GPU、不加载模型）
python run_tests.py | tail -3
```

期望：`合计：440 通过 / 0 失败 / 0 跳过`

**这一步极其重要**——它在不烧 GPU 时间的前提下，证明代码传全了、
依赖装对了。如果这里就失败，别急着跑模型。

### 4.5 把产物目录改到数据盘（强烈建议）

AutoDL 的**系统盘很小**（本项目实测 30G，模型就占了 14G），而**数据盘**才是
大而持久的。整本无损音频约 1.6 GB，产出放系统盘很快就会满。

改法：写进**不入库**的 `config.local.yaml`（**不要改 config.yaml**，
否则下次更新代码会被冲掉——这是实际踩过的坑）。

仓库里带了模板，直接复制改名即可（模板里已按 AutoDL 布局填好）：

```bash
cp config.local.example.yaml config.local.yaml
```

```yaml
# /root/autodl-tmp/NovelTTS/config.local.yaml（可从 config.local.example.yaml 复制）
project:
  output_dir: /root/autodl-tmp/NovelTTS-output/audio    # 绝对路径 -> 数据盘
  work_dir: /root/autodl-tmp/NovelTTS-output/work       # 中间产物也搬过去

generation:
  model_path: /root/models/Qwen3-TTS-12Hz-1.7B-Base
  device: cuda
  batch_size: auto
```

要点：

* **绝对路径**会被原样采用（不会拼到项目目录下）；相对路径仍相对项目根目录
* 产物在项目之外时，`manifest.json` 里存**绝对路径**；在项目内则存相对路径
  （便于整个目录搬家）——两种情况 `concat` 都能正确处理
* 换成绝对路径后，**旧的 output/ 不会自动搬过去**，需要手动 `mv`（见下）

```bash
# 迁移已有产物（如果之前跑过）
mkdir -p /root/autodl-tmp/NovelTTS-output
mv /root/autodl-tmp/NovelTTS/output/* /root/autodl-tmp/NovelTTS-output/ 2>/dev/null
rmdir /root/autodl-tmp/NovelTTS/output 2>/dev/null

# 验证路径已生效
noveltts info | grep 输出目录        # 应显示数据盘路径
```

> 为什么 `work_dir` 也搬：`chapters.json` 与片段索引虽然不大，但它们与产物
> 同生命周期；放在一起便于整体备份或迁移到新实例。

### 4.6 指向模型路径并改成 CUDA

```bash
cd /root/autodl-tmp/NovelTTS

python - <<'EOF'
from pathlib import Path
p = Path("config.yaml"); t = p.read_text(encoding="utf-8")
t = t.replace("model_path: Qwen3-TTS-12Hz-1.7B-Base",
              "model_path: /root/autodl-tmp/Qwen3-TTS-12Hz-1.7B-Base")
p.write_text(t, encoding="utf-8")
print("已指向数据盘模型")
EOF
```

> `device` / `batch_size` / `tf32` / `attn_implementation` 保持 `auto` 即可，
> 程序会自己判断。想固定也可以显式写 `cuda` / `9` / `true`。

### 4.7 把小说放进去

```bash
# 上传你的 epub / txt
cp /root/autodl-tmp/某本小说.epub sources/
noveltts info
```

`noveltts info` 应显示：

```
素材文件：
  - 某本小说.epub (...)
依赖检查：
  音频读写 : OK - ok
  音频编码 : OK - ffmpeg
模型设置（synth / run 才会真正加载）：
  路径     : /root/autodl-tmp/Qwen3-TTS-12Hz-1.7B-Base
  设备     : auto  精度: bfloat16
```

---

## 5. 开跑（先试听，再全书）

### 5.1 前三步是纯 CPU，先跑掉

```bash
noveltts parse --list                     # 看章节 ID
noveltts normalize --chapter c003         # 看洗稿效果（不写文件）
noveltts segment --chapter c003 --preview 20
```

### 5.2 第 4 步：试听 5 段（第一次用 GPU）

```bash
noveltts synth --chapter c003 --limit 5
```

**重点看这段日志**：

```
硬件与加速方案：
设备：cuda:NVIDIA GeForce RTX 3090 | 显存 24.0G（可用 23.8G） | 算力 sm_86 | torch ... | cuda ...
配置：device=cuda:0 dtype=bfloat16 attn=flash_attention_2 batch_size=8
开关：tf32=True cudnn_benchmark=True empty_cache=True clear_every=50
...
模型加载完成，用时 x.x s
音色 prompt 构建完成，用时 x.x s
```

对比本机 Mac 的基线（RTF 2.2），云端应该能到 **RTF 0.05~0.3**，
也就是快 **10~40 倍**。

### 验证（第 3 层：真的出声了）

```bash
# 文件都在
ls -lh output/audio/segments/c003/*.wav

# 时长合理（不是 0 秒，也不是几十秒的异常值）
python - <<'EOF'
from pathlib import Path
import soundfile as sf
for p in sorted(Path("output/audio/segments/c003").glob("*.wav"))[:5]:
    info = sf.info(p)
    print(f"{p.name}: {info.duration:.2f}s @ {info.samplerate}Hz")
EOF
```

把某个 wav 下载到本地听（VSCode 里右键 → Download），确认音色与语速。

### 5.3 跑全书

```bash
# 用 nohup 防止 SSH 断开导致中断
nohup noveltts run > run.log 2>&1 &

tail -f run.log          # Ctrl-C 只退出查看，任务继续
```

**可随时中断**：已生成的片段都在磁盘上，重跑同一条命令会自动跳过。

```bash
noveltts run             # 断了就重跑，看到「跳过 N」即正常续跑
```

跑完再合并整本：

```bash
noveltts concat --merge-book --book-name "我的有声书"
```

### 成本估算

以 3 小时 49 分音频（本书）为例：

**实测数据**（RTX 3090 24G，本书 c003 整章 44 段 / 3 分 13 秒音频）：

```
完成 c003 | 新生成 44 | 音频 03:13 | 耗时 01:43 | RTF 0.53
配置：device=cuda:0 dtype=bfloat16 attn=flash_attention_2 batch_size=12
```

| 环境 | RTF | 整本 3.82 小时音频的合成耗时 |
|---|---|---|
| 本机 Mac MPS | ~2.2 | 约 8.4 小时 |
| 3090（**修复前**，逐段生成） | ~1.74 | 约 6.6 小时 |
| **3090（批大小 12）** | **0.53** | **约 2.0 小时** |

批处理是关键：同一批 5 段，逐段 RTF 1.74、批量 RTF 0.53，快约 3.3 倍，
而峰值显存只从 4.7G 涨到 7.7G。

按 3090 约 1.5 元/小时算，**整本书成本约 3 元**。

---

## 6. 显存调优（30G 卡该开多大批大小）

程序会按显存自动推一个保守值。想榨干性能，照下面三步实测一遍：

### 第 1 步：看自动推的值

```bash
noveltts synth --chapter c003 --limit 20 2>&1 | grep -A 3 "硬件与加速方案"
```

假设你的卡是 30G，会看到类似 `batch_size=9`。

### 第 2 步：另开一个终端盯显存

```bash
watch -n 1 nvidia-smi
```

（或 `nvidia-smi -l 1`）

跑起来后看 `Memory-Usage` 那一列。**判定标准**：

| 观察到的显存占用 | 结论 | 动作 |
|---|---|---|
| 明显留有余量（例如用了 18G / 30G） | 还能再开大 | `batch_size` 加 2~4 再试 |
| 贴到 90% 以上 | 太满了，有 OOM 风险 | 减 2，或就用自动值 |
| 远低于预期（例如只用了 10G） | 可能是逐段合成，批处理没吃满 | 正常，见下方说明 |

### 第 3 步：写进配置

```yaml
generation:
  batch_size: 12          # 实测稳得住的值
```

> **注意一个反直觉的点**：本项目是**逐段合成**的（每段生成完立刻落盘），
> 所以 `batch_size` 影响的是「批次之间」的内存整理节奏，而不是
> 「一次前向处理多少段」。**真正决定显存占用的是 `max_new_tokens`**——
> 它决定单次生成的序列长度上限。
>
> 因此：
> * 显存吃紧 -> 先把 `max_new_tokens` 从 1024 降到 512
> * `batch_size` 保持 `auto` 就够用，不必手工调
> * 如果 OOM，程序会**自动把批大小减半**并继续（日志有警告），不会整章失败

### 30G 卡的推荐起手配置

```yaml
generation:
  device: cuda
  batch_size: auto         # 会自动推出 9 左右
  attn_implementation: auto
  tf32: auto
  max_new_tokens: 1024     # 显存紧就降到 512
```

---

## 7. 排错

### `attn=sdpa` 但我装了 flash-attn

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

报错通常是**版本三方不匹配**（torch / CUDA / Python）。用
`pip show torch` 看清版本，再找对应 wheel。程序不会因此崩溃，只是慢一点。

### 显存不足（OOM）

程序会自动把批大小减半并继续（日志里会有警告）。也可以手工压：

```yaml
generation:
  batch_size: 2            # 从 auto 改成固定小值
  max_new_tokens: 512      # 片段很短，512 通常够
```

### `CUDA out of memory` 但显存看起来够

可能是**别的进程占着卡**。看谁在占：

```bash
nvidia-smi
```

程序推导批大小时已经优先用「可用显存」而不是「总显存」，
所以这种情况它会自动算小一些。

### SSH 断了 / VSCode 掉线

任务在 `nohup` 下继续跑。重连后：

```bash
tail -f /root/autodl-tmp/NovelTTS/run.log
```

### 端口/密码变了连不上

AutoDL 每次开机 SSH 端口可能变，回控制台抄新的，更新 VSCode 的
`~/.ssh/config` 里对应的 `Port`。

### 磁盘满了

```bash
df -h
du -sh /root/autodl-tmp/NovelTTS/output/*
```

音频很占地方：无损 wav 约 **48 KB/秒**。整本 3.8 小时约 **1.6 GB**。
只想留成品的话：

```bash
noveltts encode              # 压成 m4a（约 1/8 体积）
rm -rf output/audio/segments # 确认无误后删掉片段 wav
```

---

## 8. 验证清单（照这个顺序勾）

```bash
# ── 第 1 层：环境 ──────────────────────────────
nvidia-smi                                          # 有卡
python -c "import torch; print(torch.cuda.is_available())"   # True
python -c "import qwen_tts"                         # 无报错
ffmpeg -version | head -1                           # 有输出

# ── 第 2 层：项目 ──────────────────────────────
cd /root/autodl-tmp/NovelTTS
noveltts --version                                  # 1.0.0
ls -lh assets/ref/doubao.wav                        # 参考音频在位
ls -d /root/autodl-tmp/Qwen3-TTS-12Hz-1.7B-Base     # 模型在位
python run_tests.py | tail -2                       # 440 通过 / 0 失败

# ── 第 3 层：真的能出声 ────────────────────────
noveltts info                                       # 素材与依赖 OK
noveltts parse --list                               # 切出章节
noveltts synth --chapter c003 --limit 5             # 打印加速方案 + 生成 5 段
ls -lh output/audio/segments/c003/*.wav             # 文件存在且非空
```

三层全过，就可以放心跑全书了。

---

## 9. 一句话流程（云端）

```bash
# 连上实例后
cd /root/autodl-tmp/NovelTTS
conda activate noveltts                     # 若建了独立环境

noveltts info                               # ① 环境自检
python run_tests.py | tail -2               # ② 项目自检（不烧 GPU）
cp /root/autodl-tmp/某本小说.epub sources/   # ③ 放书
noveltts parse --list                       # ④ 看章节 ID
noveltts normalize --chapter c003           # ⑤ 看洗稿
noveltts synth --chapter c003 --limit 5     # ⑥ 试听（看加速方案日志）
nohup noveltts run > run.log 2>&1 &         # ⑦ 全书
noveltts concat --merge-book                # ⑧ 合并整本
```
