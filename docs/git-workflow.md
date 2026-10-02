# Git 使用规范

> 本仓库的版本管理约定。目标：**每完成一个功能、且所有测试通过后，留下一个可回溯的提交。**

---

## 1. 仓库基本信息

| 项目 | 值 |
|---|---|
| 仓库根目录 | 即项目根目录（`.git` 与 `config.yaml` 同级） |
| 默认分支 | `main` |
| 提交身份 | 仓库本地配置（不污染全局） |

查看当前身份：

```bash
cd /Users/wanghaoxin/Documents/AustinProject/NovelTTS
git config --local --get user.name
git config --local --get user.email
```

> ⚠️ **当前 `user.email` 是占位符 `wanghaoxin@localhost`。**
> 这台机器上没有全局 git 身份，为了不擅自改动你的全局配置，我把它设成了
> **仅对本仓库生效**的占位值。推到 GitHub / Gitee 之前请改成你自己的：

```bash
# 只改本仓库（推荐，不动全局）
git config user.name  "你的名字"
git config user.email "你的邮箱@example.com"

# 想全局生效（影响这台机器上所有仓库）
# git config --global user.name  "你的名字"
# git config --global user.email "你的邮箱@example.com"
```

> 为什么要配身份：git 会把它写进每个提交，用于 `git log` / `git blame` 追责。
> 邮件填错不会导致推送失败，但 GitHub 无法把它归到你的账号下
> （提交记录里会显示成一个无头像的陌生作者）。

---

## 2. 提交什么、不提交什么

判断标准（写在 [.gitignore](../.gitignore) 顶部）：

| 类别 | 处理 | 例子 |
|---|---|---|
| 可再生成的产物 | **忽略** | `output/`（含 chapter JSON、片段音频、manifest） |
| 大二进制、可外部获取 | **忽略** | `Qwen3-TTS-*/`（数 GB 模型权重） |
| 第三方版权素材 | **忽略** | `sources/*.epub`、`sources/*.txt` |
| 小而关键的复现资源 | **提交** | `assets/ref/*.wav`（参考音频，决定音色） |
| 源码 / 配置 / 测试 / 文档 | **提交** | `noveltts/`、`config.yaml`、`tests/`、`docs/` |

### ⚠️ 两个踩过的坑

**坑 1：`.gitignore` 不支持行尾注释。**
下面这条规则**完全失效**（会被当成"路径里含 `# 注释`"的字面量）：

```gitignore
output/          # 产物目录        ← ✗ 规则无效！
```

注释必须单独成行：

```gitignore
# 产物目录
output/                          ← ✔
```

**坑 2：`.gitattributes` 与 `.gitignore` 的注释规则不同。**
`.gitattributes` **支持**行尾注释，`.gitignore` **不支持**。迁移规则时别搞混。

### 检查某文件会不会被提交

```bash
git check-ignore -v output/audio/c003.wav
#   .gitignore:24:output/**	output/audio/c003.wav     ← 命中规则，不会提交
#   无输出 = 不会被忽略，会提交
```

看完整待提交清单：

```bash
git status --short
git add -A --dry-run          # 只显示会加什么，不真的加
```

---

## 3. 分支规范

采用简化的 Git Flow：`main` 始终可用，功能在分支上开发。

| 分支前缀 | 用途 | 例子 |
|---|---|---|
| `feature/` | 新功能 | `feature/cuda-accel`、`feature/speech-rate` |
| `fix/` | 修 bug | `fix/numba-cache-import` |
| `chore/` | 杂务、重构、文档布局 | `chore/docs-layout` |
| `docs/` | 纯文档 | `docs/cloud-deployment` |

命名规则：全小写、单词用 `-` 连接、不加日期（日期看提交时间即可）。

```bash
# 开新分支
git checkout -b feature/speech-rate

# 看所有分支
git branch -vv

# 切回主干
git checkout main

# 合并（功能做完且测试通过后）
git checkout main
git merge --no-ff feature/speech-rate -m "merge: 语速调节功能"
git branch -d feature/speech-rate          # 删掉已合并的分支
```

`--no-ff` 会强制产生一个合并提交，`git log --graph` 里能清楚看出"这是一个功能"，
而不是一堆散落的提交。

---

## 4. 提交信息规范（Conventional Commits）

```
<类型>(<范围>): <简短描述>

<正文：为什么这么改，而不是改了什么>

<页脚：Breaking change / 关联 issue>
```

### 类型

| 类型 | 含义 |
|---|---|
| `feat` | 新功能 |
| `fix` | 修 bug |
| `docs` | 只改文档 |
| `test` | 只改测试 |
| `refactor` | 重构（外部行为不变） |
| `perf` | 性能优化 |
| `chore` | 构建、依赖、配置、仓库杂务 |
| `merge` | 合并分支 |

### 示例（本仓库真实提交）

```
feat(engine): 支持 CUDA 显存自适应批大小与 TF32 加速

- 按实测显存自动决定 batch_size，避免 24G 卡上跑 batch=1 浪费吞吐
- 自动开启 TF32 与 cudnn.benchmark，仅在 CUDA 上生效
- 全部开关由 config.yaml 的 generation 段驱动，可用 auto 让它自己判断

影响：CUDA 上吞吐提升明显；MPS/CPU 行为不变（有测试守护）
```

```
fix(init): 修复 numba 缓存导致 qwen_tts 完全无法导入

librosa 的 @numba.jit(cache=True) 在导入期就要选缓存目录，首选
librosa/core/__pycache__。该目录不可写时 numba 抛
"cannot cache function '__o_fold': no locator available"，
导致 import librosa 失败、进而 qwen_tts 也进不来，模型根本加载不起来。

改为在包导入时检测该目录可写性，不可写则把 NUMBA_CACHE_DIR 指向临时目录。
```

### 写法要点

* **描述用祈使句**："支持…" / "修复…"，不要写"我改了…"
* **正文写"为什么"**：`git log` 是给未来的自己看的，代码本身能说明"改了什么"
* **一次提交只做一件事**：功能与格式化分开，便于 `git revert`
* **提交前必须测试通过**（本仓库的硬性要求）

---

## 5. 标准工作流（每做一个功能）

```bash
# ① 确认当前状态干净
cd /Users/wanghaoxin/Documents/AustinProject/NovelTTS
git status

# ② 从 main 开功能分支
git checkout main
git checkout -b feature/你的功能名

# ③ 写代码 ……

# ④ 跑全部测试（必须全绿才提交）
conda activate qwen3-tts
python run_tests.py
# 若装了 pytest：pytest tests/ -q

# ⑤ 查看改了什么，确认没有误加产物
git status --short
git diff                     # 工作区改动
git diff --cached            # 已暂存的改动

# ⑥ 暂存 + 提交
git add -A
git commit -F - <<'MSG'
feat(范围): 简短描述

为什么这么改。
MSG

# ⑦ 合并回 main
git checkout main
git merge --no-ff feature/你的功能名 -m "merge: 你的功能名"
git branch -d feature/你的功能名

# ⑧ （可选）打标签记录里程碑
git tag -a v1.0.0 -m "首个可用版本"
```

---

## 6. 怎么查看

### 最常用

```bash
git log --oneline                    # 一行一条，最近的在最上
git log --oneline --graph --all      # 带分支图（推荐）
git log --stat -3                    # 最近 3 次提交改了哪些文件
git log -p noveltts/engine.py        # 某个文件的完整变更历史
```

### 看某次提交具体改了什么

```bash
git show HEAD                        # 最近一次提交的完整 diff
git show HEAD --stat                 # 只看文件清单
git show <commit>                    # 指定提交（前 7 位哈希就够）
```

### 看某一行是谁、什么时候改的

```bash
git blame noveltts/engine.py
git blame -L 200,230 noveltts/engine.py    # 只看 200~230 行
```

### 对比两个版本

```bash
git diff main feature/cuda-accel           # 两个分支的差异
git diff v1.0.0 HEAD --stat                # 两个标签之间改了哪些文件
git diff HEAD~3 HEAD                       # 最近 3 次提交合起来改了什么
```

### 找回被删的东西

```bash
git checkout <commit> -- path/to/file      # 从某次提交恢复单个文件
git revert <commit>                        # 安全地"反做"一次提交（会新建一个提交）
git reflog                                 # 所有 HEAD 移动记录——误 reset 后的救命稻草
```

### 打标签（里程碑）

```bash
git tag -a v1.0.0 -m "首个可用版本：本地跑通 5 段试听"
git tag                              # 列出所有标签
git show v1.0.0                      # 看某个标签
```

---

## 7. 推到远端（GitHub / Gitee）

```bash
# 先在网站上建一个**空仓库**（不要勾选 README/gitignore），然后：
git remote add origin git@github.com:你的用户名/NovelTTS.git
git push -u origin main
git push origin --tags               # 连标签一起推

# 之后日常推送
git push
```

### 推之前检查体积

仓库里有 `assets/ref/*.wav`（约 13 MB），首次推送前确认没有误加更大的文件：

```bash
# 看仓库里最大的 10 个已跟踪文件
git ls-files -z | xargs -0 du -h | sort -rh | head -10
```

> ⚠️ **GitHub 单文件硬上限 100 MB**。若将来要加更长的参考音频，
> 请改用 [Git LFS](https://git-lfs.com/)：
> ```bash
> git lfs install
> git lfs track "assets/ref/*.wav"
> ```

### 万一误提交了大文件

```bash
# 只从"最新提交"里移除，但保留本地文件
git rm --cached 大文件.bin
echo "大文件.bin" >> .gitignore
git commit -m "chore: 移除误提交的大文件"

# 注意：这只影响之后的提交，历史里仍然存在。
# 要彻底清除需要 git filter-repo 并强推，会影响所有协作者。
```

---

## 8. 本仓库的开发历史

功能分支开发完就合并并删除，所以 `git branch` 平时只会看到 `main`。
**历史记录在 log 里**——用下面这条命令看得最清楚：

```bash
git log --oneline --graph --decorate
```

输出形如（`|\` 与 `|/` 之间就是那条功能分支）：

```
*   d25a354 merge: NVIDIA/CUDA 自适应加速与语速调节
|\
| * b9e7cdf feat(accel): 新增 NVIDIA/CUDA 自适应加速与语速调节
|/
*   17a406d merge: 文档布局重构
|\
| * 07e3afc chore(docs): 重构文档布局，抽出云端部署与 git 规范
|/
* 98eebe8 chore: 初始化仓库，纳入完整可运行版本
```

其他常用查看方式：

```bash
git branch -vv                       # 本地分支（含各自指向的提交）
git tag -n                           # 标签及说明
git log --oneline --all --graph      # 含所有分支
git log --merges --oneline           # 只看合并提交（= 每个已完成的"功能"）
```

---

## 9. 每次做完功能的固定动作（本仓库约定）

```bash
# ① 开分支
git checkout -b feature/xxx

# ② 写代码

# ③ 跑全部测试（硬性要求：必须全绿）
python run_tests.py

# ④ 提交（信息写清"为什么"）
git add -A
git commit -F - <<'MSG'
feat(范围): 简短描述

为什么这么改。
MSG

# ⑤ 合并回 main 并删分支
git checkout main
git merge --no-ff feature/xxx -m "merge: xxx"
git branch -d feature/xxx

# ⑥ 里程碑打标签
git tag -a v1.1.0 -m "说明"
```

**测试不通过就不提交**——这是本仓库唯一的硬性纪律。
好处是 `main` 上的任意一个提交都是可运行的，随时能回退。
