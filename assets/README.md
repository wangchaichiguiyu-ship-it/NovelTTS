# assets/ —— 随仓库一起版本化的资源

这个目录里的东西**会被提交进 git**，所以只放"复现所必需、且体积极小"的文件。

## 目录结构

```
assets/
└── ref/                        参考音频（声音克隆用）
    ├── doubao.wav              默认参考音色（45.5s / 48kHz / 立体声 / 12.5 MB）
    └── test.wav                备用短参考（8.8s / 24kHz / 单声道 / 0.4 MB）
```

## 为什么参考音频要入库

声音克隆的输出**完全取决于参考音频**。如果参考音频不入库，别人（或几个月后的你自己）
克隆出来的音色就会不一样，有声书也就无法复现。

权衡结果：`doubao.wav` 12.5 MB，对 git 来说偏大但可以接受；相比"音色无法复现"的代价，
这个体积是值得的。

> 如果打算把仓库推到 GitHub 等远端，注意 **GitHub 单文件硬上限 100 MB**。
> 若将来要换成更长的参考音频（> 50 MB），请改用
> [Git LFS](https://git-lfs.com/)：
> ```bash
> git lfs install
> git lfs track "assets/ref/*.wav"
> git add .gitattributes
> ```

## 换成自己的音色

1. 准备一段 **3~15 秒**的干净人声（无背景音乐、无混响、无多人对话）
2. 放到 `assets/ref/我的音色.wav`
3. 改 `config.yaml`：

```yaml
voice:
  ref_audio: assets/ref/我的音色.wav
  ref_text: "这段音频里实际说的话，逐字写在这里"
```

`ref_text` 与音频内容越一致，克隆相似度越高。留空会自动退回
`x_vector_only` 模式（只用说话人向量，相似度会打折）。
