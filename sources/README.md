# sources/ —— 放你的小说

把 `.epub` 或 `.txt` 文件**直接丢进这个目录**即可。

```bash
cp ~/Downloads/某本小说.epub sources/
noveltts info          # 应该能看到它
noveltts parse --list  # 解析并查看章节 ID
```

## 这个目录的内容不进 git

`sources/` 下除本文件与 `.gitkeep` 之外的内容都被 `.gitignore` 忽略。

原因：小说原文是**第三方版权内容**，随仓库扩散不合适。仓库只提供工具，
素材请自行准备。

## 支持格式

| 格式 | 说明 |
|---|---|
| `.epub` | **推荐**。EPUB 的目录（nav / ncx）能提供完整章节标题，切章质量最好 |
| `.txt` | 支持。会按 `Loss 1:` / `Chapter 3` / `第X章` 这类标题行切章；被排版成两行的标题会自动接回 |
| `.md` | 同 `.txt` |

## 多个文件时用哪个

`sources/` 里同时存在 epub 与 txt 时，程序默认**优先 EPUB**
（章节标题更完整），并会在日志里说明选了哪个：

```
素材目录里有 2 个文件，自动选择 xxx.epub（可用 --source 指定其它）
```

想指定具体文件：

```bash
noveltts parse --source "sources/某本书.txt"
```
