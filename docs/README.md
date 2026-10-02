# NovelTTS 文档索引

| 文档 | 内容 | 适合谁看 |
|---|---|---|
| [../README.md](../README.md) | 设计原理、目录结构、内存策略、洗稿规则表 | 想了解"为什么这么设计" |
| [user-guide.md](user-guide.md) | **上手操作**：命令规范、逐步流程、产物说明、故障排查 | 第一次使用 |
| [cloud-deployment.md](cloud-deployment.md) | **云端部署**：AutoDL + VSCode Remote SSH、NVIDIA 优化、验证清单 | 要上 GPU 跑全书 |
| [git-workflow.md](git-workflow.md) | 本仓库的 git 规范、提交约定、查看命令 | 参与开发 / 管理版本 |

## 按场景找文档

**"我只想赶紧生成有声书"**
→ [user-guide.md](user-guide.md) 第 6 节「建议的执行顺序」

**"`--chapter c003` 里的 c003 是什么？"**
→ [user-guide.md](user-guide.md) 第 1 节「参数命名规范」

**"我要在 AutoDL 上跑，怎么配环境？"**
→ [cloud-deployment.md](cloud-deployment.md)

**"跑出来的音频有问题"**
→ [user-guide.md](user-guide.md) 第 5 节「故障排查」

**"怎么知道我的改动没搞坏别的东西？"**
→ `python run_tests.py`（零依赖，372 个用例，不需要 GPU）

**"怎么回滚到上一个能用的版本？"**
→ [git-workflow.md](git-workflow.md) 第 6 节「怎么查看」
