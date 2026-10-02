# 技术债与改进清单

> **这份文档是给自己人看的，不是宣传材料。**
> 它记录当前实现的弱点、重复造轮子的地方、以及明确知道但没做的事。
>
> 编写原则：**每条都给出证据、代价和修法**，而不是笼统地说"可以优化"。
> 如果你要接手这个项目，请先读这份，再读 [overview.md](overview.md)。
>
> 当前版本：`v2.0.0`

---

## 0. 优先级总览

按 **（影响 × 修复成本）** 排序。**建议严格按 P0 → P3 顺序处理**。

| 优先级 | 事项 | 类型 | 预估 | 为什么这个优先级 |
|---|---|---|---|---|
| **P0** | 无任何音频质量自动检查 | 功能缺口 | 0.5 天 | 现在 3220 段的质量只能靠人耳抽查，这是最大风险 |
| **P0** | 测试全部用假引擎，模型零覆盖 | 验证缺口 | 1 天 | "457 通过"给了过强信心 |
| **P1** | 迁移到 pytest | 技术债 | 0.5 天 | 越晚迁越贵；当前框架扩展性差 |
| **P1** | 删除自写 YAML 解析器 | 重复造轮子 | 2 小时 | 已埋下静默出错风险 |
| **P1** | GPU 利用率只有 30%，性能未挖尽 | 性能 | 1 天 | 可再快约 2 倍 |
| **P2** | 批大小启发式只拟合了一个数据点 | 健壮性 | 0.5 天 | 换卡可能偏差很大 |
| **P2** | API-REFERENCE 会随代码腐烂 | 文档债 | 0.5 天 | 1700 行里的字段表必然过时 |
| **P2** | 只有 1 本书 / 1 音色 / 1 张卡的验证样本 | 验证缺口 | 视情况 | 泛化能力未知 |
| **P3** | 部署方式仍是 tar 覆盖 | 工程债 | 1 天 | 可用但有更好做法 |
| **P3** | `synthesize_chapter` 要求预先切分 | API 设计瑕疵 | 1 小时 | 我自己写文档时就被绊倒过 |
| **P3** | 情绪控制完全没做 | 功能缺口 | 见产品文档 | 属于产品化范围 |

---

## 1. P0：没有任何音频质量自动检查

### 现状与证据

生成了 3220 个片段（4 小时 29 分音频），**零失败**——但"零失败"只意味着
"没抛异常"，**不意味着音频是对的**。

当前对音频质量的唯一保障是：**我人工抽听了几段**。

`SynthesisReport` 里能看到的只有：

```python
total / generated / skipped / failed / audio_seconds / elapsed_seconds / errors
```

`errors` 只在**异常**时才有值。以下情况**完全没有检测**：

| 潜在缺陷 | 现有检测 | 后果 |
|---|---|---|
| 某段漏读了几个字 | ❌ 无 | 听众察觉内容缺失 |
| 某段音质崩坏（金属声/噪声） | ❌ 无 | 需要人工发现 |
| 某段时长异常（0.3 秒或 40 秒） | ❌ 无 | 明显不自然 |
| 某段全是静音 | ❌ 无 | 突兀的空白 |
| 音色在某段漂移 | ❌ 无 | 听起来像换了人 |
| 语速在某段失控 | ❌ 无 | 突兀 |

### 为什么这是 P0

一本 4.5 小时的有声书，**人工听完需要 4.5 小时**，而且注意力会衰减 ——
第 3 小时之后基本听不出问题。**没有自动质检，质量的唯一保证就是运气。**

### 修法（0.5 天，投入产出比最高）

用**已有的 manifest 数据**就能做出大部分检查，不需要任何新模型：

```python
# noveltts/qa.py（新模块）
"""基于 manifest 的自动质检。不需要加载模型。"""

@dataclass
class QAIssue:
    segment_id: str
    kind: str          # "duration_outlier" | "silence" | "too_short" | "clip" | "loudness"
    detail: str
    severity: str      # "warn" | "error"

def check_chapter(config, chapter_id) -> list[QAIssue]:
    """对一章做质检，返回问题列表。"""
```

**具体检查项**（都能从 manifest + 音频文件算出来）：

| 检查 | 方法 | 阈值建议 |
|---|---|---|
| **时长离群** | 字符数 → 音频时长 的比值，与本章中位数比较 | 偏离 > 40% 报警 |
| **异常短** | `duration < 0.3s` 但 `char_count > 10` | 必报 |
| **异常长** | `duration > 30s` | 必报（可能有重复/卡住） |
| **数字静音** | 音频全零或 RMS < -60 dBFS | 必报 |
| **削波** | 峰值 ≥ -0.1 dBFS | 报警 |
| **响度异常** | RMS 偏离本章中位数 > 6 dB | 报警 |
| **首尾静音过长** | 首/尾静音 > 1.5s | 提示 |

**输出**：`output/qa/c003-issues.json` + 一个 CLI 子命令

```bash
noveltts qa --chapter c003              # 质检一章
noveltts qa --all                        # 质检全书，输出汇总
noveltts qa --all --json > qa.json       # 给 GUI 用
```

**验收标准**：拿现在这本已生成的书跑一遍，应该报出**少量**真实问题
（如果报出 0 个问题，说明阈值太松；如果报出几百个，说明太紧）。
**这条验收很重要** —— 它能验证质检器本身有没有用。

### 顺带收益

有了质检，就能实现"只重做有问题的段"：

```bash
noveltts qa --all --json > qa.json
noveltts synth --chapter c003 --segments $(jq -r '.issues[].segment_id' qa.json)
```

这比 `--retry-failed` 更有价值，因为它能发现**没抛异常但质量差**的段。

---

## 2. P0：测试全部用假引擎，模型行为零覆盖

### 现状与证据（实测统计）

```
test_pipeline.py         53 个测试，其中 49 处使用假引擎
test_engine_contract.py  29 个测试，全部使用 stub 模块
真实加载模型             0 处
```

我在汇报时说「**457 个测试全部通过**」。这句话**字面正确但容易误导**：
它证明的是**我写的那部分逻辑**是对的，**模型是黑盒，一行都没测**。

### 这带来的具体盲区

| 盲区 | 后果 |
|---|---|
| `generate_voice_clone` 的实际参数名/行为 | 换 qwen-tts 版本可能直接崩（0.0.3 → 0.1.1 时就差点踩到） |
| 批量输入时模型返回的段数、顺序 | 虽然我加了段数核对，但"顺序是否与输入一致"**未验证** |
| 模型对超长文本 / 特殊字符的行为 | 生产环境可能遇到 |
| 空结果、重复内容等边界 | 只用假引擎模拟过 |

### 修法（1 天）

**核心思路：把"慢但真实"的测试隔离出来，默认不跑，但能一键跑。**

```python
# tests/test_real_model.py
"""需要真实模型 + GPU 的测试。默认跳过。

运行方式：
    NOVELTTS_REAL_MODEL=1 python run_tests.py real_model
"""

import os, pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NOVELTTS_REAL_MODEL"),
    reason="需要设置 NOVELTTS_REAL_MODEL=1 且准备好 GPU 与模型",
)

def test_clone_batch_order_matches_input():
    """关键：批量返回的顺序必须与输入顺序一致。

    这是"把 B 的音频写进 A 的文件名"这类静默错误的前提条件。
    """
```

**必须覆盖的 5 个真实测试**（总共只跑几秒，成本极低）：

1. **批量顺序一致性** —— 输入 3 段可区分的文本（如 "one two three" / "four five six"），
   验证返回顺序与文本对应（可用时长单调性近似判断）
2. **API 契约冒烟** —— 真的调一次 `generate_voice_clone`，确认参数名没变
3. **空文本 / 超长文本** 的行为
4. **`create_voice_clone_prompt` 的 ICL vs xvec** 两种模式都能用
5. **`non_streaming_mode` 两种取值** 都能跑

**验收标准**：这 5 个测试在 3090 上跑通，且**总耗时 < 30 秒**。

### 另外：诚实化项目文档里的说法

把所有文档里的"457 个测试全部通过"改成类似：

> **457 个测试通过**（覆盖流水线逻辑；**模型行为需另跑 `NOVELTTS_REAL_MODEL=1`**）

---

## 3. P1：迁移到 pytest —— 我重复造了轮子

### 现状与证据

`run_tests.py` **550 行、48 个自定义函数**，其中相当一部分是在手搓 pytest 的功能：

| 我手写的 | 对应的 pytest 现成能力 |
|---|---|
| `_MonkeyPatch` 类 | `monkeypatch` fixture |
| `_CaptureFixture` + `_Tee` 类 | `capsys` fixture |
| `_RaisesContext` | `pytest.raises(match=...)` |
| 自写的 `tmp_path` fixture | `tmp_path` fixture |
| 自写的参数化（处理 `(7,5)` 元组） | `@pytest.mark.parametrize` |
| 自写的测试发现与执行 | pytest 全部搞定 |
| 自写的失败输出格式 | pytest 的 diff / 断言重写 |

**当时"不装 pytest"的理由**（"用户环境可能没装"）**是站不住脚的**：
`pip install pytest` 是一行命令，而我为此写了 550 行且功能是残缺的。

### 已经因此踩到的坑

从 commit 历史看，我为这个自写框架**修了至少 5 次 bug**：

- `_MonkeyPatch` 漏了 `os` 导入
- `capsys` 替换 buffer 而非原地清空，导致 logging 持有旧引用
- 单参数 `parametrize` 把元组参数 `(7,5)` 撕成 `7`
- 缺少 `monkeypatch` / `capsys` fixture
- 跨测试污染：`test_cli` 断言"进程内没有 torch"，而 `test_accelerator` 导入了 torch

**每一个都是"重新实现别人已经解决好的问题"的代价。**

### 现在缺什么（pytest 有而我这里没有）

| 能力 | 影响 |
|---|---|
| `@pytest.mark.skipif` | 真实模型测试无法优雅跳过 |
| fixture 依赖注入 | 无法组合测试装置 |
| `conftest.py` 分层 | 无法按目录共享 fixture |
| `-k` / `-m` 表达式筛选 | 无法按标记筛选测试 |
| 插件生态（`pytest-cov` 等） | 没有覆盖率报告 |
| `--lf`（只跑上次失败） | 开发循环变慢 |

### 修法（0.5 天）

```bash
pip install pytest pytest-cov
```

```python
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = [
    "real_model: 需要真实模型与 GPU（默认跳过）",
    "slow: 耗时较长的测试",
]
```

迁移步骤：

1. 把 `run_tests.py` 的 fixture 换成 pytest 内置（删掉约 300 行）
2. `tests/conftest.py` 放共享 fixture（`project_root` / `workdir` / `config`）
3. 保留 `run_tests.py` 作为**向后兼容入口**（内部 `subprocess` 调 pytest），
   这样文档里的命令不用改
4. 加 `pytest-cov`，看真实覆盖率（**预计会发现不少未覆盖分支**）

---

## 4. P1：自写 YAML 解析器 —— 应该删掉

### 现状与证据

`noveltts/config.py` 里有：

```python
def _load_yaml(path: Path) -> dict[str, Any]:        # 515 行
def _mini_yaml_load(text: str) -> dict[str, Any]:    # 536 行
```

约 60 行手写解析器，只支持 YAML 子集。**理由是"让包只依赖标准库"**。

### 为什么这是负债

| 问题 | 说明 |
|---|---|
| **静默出错** | 遇到不支持的 YAML 特性可能**不报错而是解析成错的结果** |
| **已修的 bug** | 块序列（`- item`）不支持 → 修过一次；升级占位 dict 的栈层级错 → 修过一次 |
| **功能缺失** | 多行字符串 `\|`、锚点 `&`/`*`、`>` 折叠、复杂嵌套列表的列表 |
| **测试负担** | 为它写了 45 个用例（`test_config_utils.py` 大半） |
| **PyYAML 本来就是依赖** | `qwen-tts` 间接依赖 PyYAML；实际环境里 `PyYAML 6.0.2` 一直都在 |

**最关键的反驳**：项目**已经在用** `yaml.safe_dump` 之类的库了吗？
不，但 `pyproject.toml` 里 `PyYAML` 是**显式依赖**（`config.py` 顶部有 `import yaml` 的
可选分支）。所以"省依赖"这个理由**根本不成立**。

### 修法（2 小时）

```python
import yaml   # 从可选依赖改为必需依赖

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层必须是映射，实际是 {type(data).__name__}")
    return data
```

删掉 `_mini_yaml_load` 及其测试，**净减约 100 行**。

> 如果确实想保留"零依赖"，正确做法是**把解析交回调用方**：
> `AppConfig.load(..., parser=yaml.safe_load)`，而不是自己实现一个残缺版。

---

## 5. P1：GPU 只用了 30%，性能远未挖尽

### 现状与证据

整本生成时的监控数据（每 5 分钟采样）：

```
GPU 利用率：27% ~ 39%（全程）
显存占用：7.1 ~ 7.4 GB / 24 GB
实时率 RTF：0.37
```

**瓶颈不在 GPU 算力。** 显存只用了 30%，GPU 空转 60%+。
这说明时间花在 **Python 层调度、逐批往返、数据搬运**上。

### 我当时的判断失误

我在文档里写「RTF 0.37，整本约 2 小时」，语气像在描述**已达成的成果**，
但它其实是**当前实现的水平**，不是硬件极限。按显存余量估算：

```
可用预算 = 24G × 0.8 = 19.2G
减去固定占用 4.5G = 14.7G
按实测 1.2G/样本 → 可开 12 批（当前值）
若按乐观的 0.75G/样本 → 可开 19 批
显存实际只用了 7.4G → 远未吃满
```

### 可尝试的优化（1 天）

| 方向 | 做法 | 预期收益 |
|---|---|---|
| **提高批大小** | 把 `max_batch` 从 16 提到 32~48，实测 OOM 边界 | 1.5~2× |
| **按字符数分桶** | 同批内文本长度相近，减少 padding 浪费 | 1.2~1.4× |
| **`torch.compile`** | 编译 talker 前向 | 1.2~1.5×（首次编译慢） |
| **vLLM 集成** | 官方文档有 `vLLM Integration` 章节，专门做高吞吐 | 可能 2~4× |
| **CUDA Graph** | 固定形状下减少 launch 开销 | 1.1~1.3× |

> ⚠️ **注意**：官方文档有 [vLLM Integration](https://mintlify.wiki/QwenLM/Qwen3-TTS/advanced/vllm-integration)
> 与 [Performance Optimization](https://mintlify.wiki/QwenLM/Qwen3-TTS/advanced/performance)
> 两页专门讲这个，**我完全没读过**。如果要做性能优化，**第一步应该是读它们**，
> 而不是自己试参数。

### 为什么不是 P0

1.6 小时 vs 0.8 小时，对一次性生成一本书来说**体感差异不大**。
但如果要做成产品（多用户、批量），这就是**成本翻倍**的差别。

---

## 6. P2：批大小启发式只拟合了一个数据点

### 现状与证据

```python
_BASE_VRAM_GB = 4.5          # 固定占用
_PER_SAMPLE_VRAM_GB = 1.2    # 每样本增量
```

这两个常数的来源：**在 3090 上测了 3 个点**。

| 批大小 | 峰值显存 | 增量 |
|---|---|---|
| 1 | 4.7 G | — |
| 2 | 5.4 G | +0.70 G |
| 5 | 7.7 G | +0.77 G / 样本 |

然后我把 `0.75 G/样本` **上浮 60% 取 1.2**，用于**所有显卡**。

### 为什么这是隐患

| 问题 | 说明 |
|---|---|
| **单点外推** | 从 3 个点拟出一条直线，外推到 13 甚至 16 批 |
| **未考虑文本长度** | 显存随**实际音频时长**变化，而本书片段平均只有 60 字符。换成中文或长句，系数可能差很多 |
| **未考虑 `max_new_tokens`** | 用户调到 2048 时，估算完全失效 |
| **不同架构差异** | 4090 / A100 / H100 的显存效率不同 |
| **`total_vram_gb` 与实际可用不符** | 代码里取 `min(总显存×0.8, 可用显存)`，但"可用显存"在容器里有额外开销 |

文档里那张"8G→2, 12G→5, 24G→13"的表**看起来像规格，实际是外推**。
这是我在叙述上把启发式说得比实际更可靠。

### 修法（0.5 天）

1. **加运行时自适应**（比静态估算可靠得多）：

```python
# 首次批失败后，不只用"减半"，而是记录实际峰值显存，重算系数
torch.cuda.reset_peak_memory_stats()
# ... 跑一批 ...
peak = torch.cuda.max_memory_allocated() / 1024**3
# 用实测峰值反推更准的 per_sample，供后续批次使用
```

2. **保守起步 + 快速爬升**：从 `max(1, auto//2)` 开始，成功后逐步加倍
3. **把常数写进可配置项**，允许现场标定：

```yaml
generation:
  vram_base_gb: 4.5           # 可覆盖
  vram_per_sample_gb: 1.2     # 可覆盖
```

4. **文档改成诚实表述**：

> 这两个常数来自 3090 上的一次实测（3 个数据点），**是启发式而非规格**。
> 换硬件请用 `--batch-size` 显式指定，或观察日志里的实际峰值显存后标定。

---

## 7. P2：API-REFERENCE 会随代码腐烂

### 现状与证据

`docs/API-REFERENCE.md` 约 **1700 行**，其中：
- 全部配置字段与默认值表（8 个 dataclass，约 90 个字段）
- 全部数据模型字段表（14 个 dataclass）
- 全部 CLI 参数

**这些内容全部是手工抄写/生成的。没有任何机制保证它们同步。**

### 为什么这是个真问题

我在写这份文档时**已经发现 7 处与实现不符**（虚构的方法、错的返回值类型、
错的示例输出）。那次是因为我逐个 `import` + `inspect` 核对了。

**但下一次代码改动，没人会再核对一遍。** 三个月后这份文档会变成
"看起来详尽但充满错误的陷阱"——比没有文档更糟。

### 修法（0.5 天）

**思路：让文档的一部分由代码生成，而不是手写。**

```python
# scripts/gen_api_docs.py
"""从 dataclass 定义自动生成字段表，注入到 API-REFERENCE.md。"""

import dataclasses as dc
from noveltts import config, models

TEMPLATE = "<!-- AUTO-GENERATED:config-tables -->\n{content}\n<!-- /AUTO-GENERATED -->"

def render_config_tables() -> str:
    """渲染所有配置类的字段表。"""
    ...

def main():
    doc = Path("docs/API-REFERENCE.md")
    text = doc.read_text(encoding="utf-8")
    text = replace_between_markers(text, "config-tables", render_config_tables())
    doc.write_text(text, encoding="utf-8")
```

配合**测试**保证不腐烂：

```python
def test_api_doc_is_up_to_date():
    """文档里的自动生成区块必须与代码一致。

    失败时提示：运行 python scripts/gen_api_docs.py
    """
    expected = render_config_tables()
    actual = extract_between_markers(Path("docs/API-REFERENCE.md").read_text(), "config-tables")
    assert actual == expected, "文档过期，请运行 scripts/gen_api_docs.py"
```

**同时**：手写的部分（设计约定、集成方式、GUI 建议）保留——
那才是文档真正的价值所在，而字段表这种机械内容应该自动生成。

### 顺带：文档总量失衡

文档 **4954 行** vs 源码 **9424 行**。比例超过 1:2。

而且这些文档**全部在同一次会话里生成，读者只有我一个人**。
没有经过真实用户使用检验的文档，会系统性地**高估清晰度**。

**建议**：不要再增加文档总量。优先把自动生成做好，把会腐烂的部分交给机器。

---

## 8. P2：验证样本只有 1 本书 / 1 音色 / 1 张卡

### 现状与证据

| 维度 | 验证过的 | 未验证的 |
|---|---|---|
| 书 | 1 本英文轻小说（EPUB，19 万字） | 中文 / 日文 / TXT / PDF / 排版异常的 EPUB / 多卷本 |
| 音色 | `doubao.wav`（1 个，男声，英文） | 女声 / 带噪参考 / 非英语参考 / 3 秒超短参考 |
| 语言 | English | 另外 9 种（模型支持中英日韩德法俄葡西意） |
| GPU | RTX 3090 24G | 4090 / A100 / 低显存卡 / Mac MPS（只跑过 5 段） |
| 场景 | 单角色朗读 | 多角色、对话、情绪 |

### 修法

**1. 建一个"回归语料"**（低成本，高价值）：

```text
tests/fixtures/books/
├── en_light_novel.epub       # 现有这本，抽 1 章
├── zh_web_novel.epub          # 中文，测洗稿的中文规则
├── weird_structure.epub       # 章节嵌套怪异，测解析健壮性
├── no_toc.txt                 # 无目录 TXT，测正则切章
└── dialogue_heavy.epub        # 对话密集，测切分的对话分离
```

每个只保留 **1 章**（避免版权与体积问题），跑"解析 → 洗稿 → 切分"三步，
断言片段数、字符数分布、无超硬上限。**这不需要 GPU，可以进 CI。**

**2. 音色鲁棒性检查**：

```python
def test_reference_audio_quality(config):
    """参考音频的基本质量检查，提前发现问题。"""
    info = probe_audio(config.voice.ref_audio)
    assert info.duration >= 3.0, "模型要求至少 3 秒参考音频"
    assert info.duration <= 30.0, "过长的参考音频无益且拖慢 prompt 构建"
    # 检查是否数字静音 / 削波 / 信噪比过低
```

**3. 显式记录"未验证范围"**：在 README 或本文档里明确写清，
避免后续使用者以为"支持所有格式"。

---

## 9. P3：部署方式仍是 tar 覆盖

### 现状

`scripts/deploy_remote.sh` 用 `tar` + `scp` 把代码传到远程。
它已经比手敲好（自动排除 `config.yaml`），但本质仍是**文件覆盖**。

| 问题 | 说明 |
|---|---|
| 无原子性 | 传到一半断了，远程是"半个版本" |
| 无回滚 | 想回退只能重新传旧包 |
| 无版本对应 | 远程代码与 git commit 没有强关联（靠时间戳猜） |
| 依赖手动指定路径 | host/dir/python 三个参数可能写错 |
| **刻意不传 `.git`** | 所以远程无法 `git log` 确认自己是什么版本 |

### 修法（1 天）

**方案 A：远程 git clone（推荐）**

```bash
# 一次性
ssh autodl 'git clone <repo> /root/autodl-tmp/NovelTTS'

# 每次更新
ssh autodl 'cd /root/autodl-tmp/NovelTTS && git fetch && git checkout <tag>'
```

需要解决**私有仓库认证**（部署密钥或 token）。好处是版本明确、可回滚、原子。

**方案 B：打包时写入版本**

```bash
# deploy_remote.sh 里加
COMMIT=$(git rev-parse --short HEAD)
DIRTY=$(git status --porcelain | wc -l)
cat > .deploy-info <<EOF
commit: $COMMIT
dirty_files: $DIRTY
deployed_at: $(date -Iseconds)
EOF
```

远程可 `cat .deploy-info` 自查版本，`noveltts info` 也把它显示出来。

**建议先做 B**（1 小时），再做 A。

---

## 10. P3：`synthesize_chapter` 要求预先切分

### 现状与证据

```python
def synthesize_chapter(self, chapter: Chapter, ...):
    segments = chapter.segments
    if not segments:
        raise ValueError(f"章节 {chapter.chapter_id} 还没有切分，请先调用 segment_chapter()")
```

**这个设计本身是合理的**（职责分离：切分是独立阶段，结果要落盘复用）。

**问题是 API 命名有误导性**：`synthesize_chapter` 听起来像"给我一章，我给你音频"，
但它其实要求调用方先做一步别的事。

**证据**：我自己在写 `API-REFERENCE.md` 的示例时就被绊倒了 ——
写了个 `parse_book` → `synthesize_chapter` 的例子，跑起来直接抛异常。

### 修法（1 小时）

**方案 A：加一个真正端到端的方法**（推荐）

```python
def synthesize_chapter_auto(self, chapter: Chapter, **kwargs) -> SynthesisReport:
    """端到端：没有切分就自动切分。给"我就想合成这一章"的场景用。"""
    if not chapter.segments:
        segmenter = TextSegmenter(self.config.segment, language=chapter.language)
        chapter.segments = segmenter.segment_chapter(chapter)
    return self.synthesize_chapter(chapter, **kwargs)
```

**方案 B：在异常消息里给出可用代码**

```python
raise ValueError(
    f"章节 {chapter.chapter_id} 还没有切分。请先执行：\n"
    f"    from noveltts.segmenter import TextSegmenter\n"
    f"    chapter.segments = TextSegmenter(config.segment).segment_chapter(chapter)\n"
    f"或改用 noveltts.run 命令（会自动切分）"
)
```

**建议两个都做** —— 方案 B 成本几乎为零，能立刻减少困惑。

---

## 11. 已知的设计取舍（不是 bug，但要知道）

这些是**我明确选择的结果**，记录在此避免被误认为缺陷：

| 取舍 | 选择 | 代价 |
|---|---|---|
| **语速在生成后处理** | 用相位声码器（音高不变） | 改语速必须**重新生成**（不能只重处理） |
| **批处理 vs 失败隔离** | 默认批处理，失败时降级逐段 | 失败时该批要多花一次全量时间 |
| **manifest 存相对路径** | 项目内相对、项目外绝对 | 两种语义并存，读代码时需注意 |
| **配置三层优先级** | 默认 < config.yaml < local < CLI | 概念多，但解决了"部署冲掉配置"的真问题 |
| **惰性 import** | 包只依赖标准库 | 部分模块里 import 位置分散，可读性略差 |
| **不传入 `.git` 到远程** | 减小传输体积 | 远程无法自查版本（见 §9） |

---

## 12. 不建议做的事

避免把精力花在收益低的地方：

| 不建议 | 原因 |
|---|---|
| **继续增加文档总量** | 已经失衡（4954 行 vs 9424 行源码），且会自动腐烂 |
| **在没有质检前支持更多输入格式** | 质量问题比格式支持重要得多 |
| **自研 TTS 相关算法** | 官方文档有 vLLM/性能优化章节，先读再动手 |
| **为"零依赖"继续造轮子** | 已经证明代价大于收益（YAML 解析器、测试框架） |
| **过度优化 Mac 端性能** | Mac 是开发/验证环境，真正的产出在云端 GPU |

---

## 13. 建议的执行顺序

```text
第 1 周（打地基，都是低成本高收益）
  ├─ P0-1  音频质检器 + noveltts qa 命令         0.5 天
  ├─ P0-2  5 个真实模型测试（默认跳过）           1 天
  ├─ P1-1  迁移到 pytest + 装 pytest-cov         0.5 天
  ├─ P1-2  删除自写 YAML 解析器                   2 小时
  └─ P2-1  文档自动生成 + 防腐烂测试              0.5 天
     合计约 3 天。做完这五项，项目从"能跑"变成"可信"。

第 2 周（提性能与健壮性）
  ├─ P1-3  读官方 vLLM/性能文档，做性能优化        1 天
  ├─ P2-2  运行时自适应标定显存系数                0.5 天
  └─ P2-3  建回归语料（多语言/多结构，仅 CPU）      0.5 天

第 3 周（工程化）
  ├─ P3-1  部署写入 .deploy-info，再考虑 git clone  1 天
  └─ P3-2  synthesize_chapter_auto + 友好异常      1 小时
```

---

## 附：本清单的证据来源

所有"现状与证据"部分的数据都来自实际检查，不是估计：

| 数据 | 来源 |
|---|---|
| 测试用假引擎的比例 | `grep -c "FakeSynthesisEngine\|dry_run=True" tests/*.py` |
| GPU 利用率 27~39% | 云端整本生成期间每 5 分钟采样的 `nvidia-smi` |
| 显存 7.1~7.4 G | 同上 |
| 显存-批大小 3 个数据点 | `bench_batch.py` 在 3090 上的实测输出 |
| `run_tests.py` 48 个函数 | `grep -c "def " run_tests.py` |
| 代码/文档行数 | `wc -l noveltts/*.py tests/*.py docs/*.md` |
| 自写框架修的 5 个 bug | `git log --oneline --grep="run_tests"` |
| API 文档 7 处不符 | 逐个 `import` + `inspect` 核对 68 个接口的脚本输出 |
