# NovelTTS 文档索引

| 文档 | 内容 | 适合谁看 |
|---|---|---|
| **[overview.md](overview.md)** | **完整工作流**：导入电子书 → 云端生成 → 成果交付，含常见陷阱与复现检查清单 | **所有人，先看这个** |
| **[API-REFERENCE.md](API-REFERENCE.md)** | **接口规范**：全部 Python API / CLI 参数 / 数据模型 / 异常约定 | 开发可视化界面、二次开发 |
| [user-guide.md](user-guide.md) | 本机操作：命令命名规范、逐步流程、产物说明、故障排查 | 第一次使用 |
| [cloud-deployment.md](cloud-deployment.md) | 云端部署：AutoDL + VSCode Remote SSH、NVIDIA 优化、防断线、验证清单 | 要上 GPU 跑全书 |
| **[TECH-DEBT.md](TECH-DEBT.md)** | **技术债与改进清单**：弱点、重复造轮子、已知取舍，含证据与修法 | **接手项目前必读** |
| **[PRODUCTIZATION.md](PRODUCTIZATION.md)** | **产品化设计**：做成拖拽式软件的架构、技术栈、风险与实施计划 | 想把它包装成产品 |
| [git-workflow.md](git-workflow.md) | git 规范、提交约定、查看与回滚命令 | 参与开发 / 管理版本 |
| [../README.md](../README.md) | 设计原理、目录结构、内存策略、洗稿规则表 | 想了解"为什么这么设计" |

## 按问题查找

| 我想知道… | 去哪 |
|---|---|
| **从零做一本有声书，完整流程是什么** | [overview.md](overview.md) |
| 怎么把成品交给别人 / 怎么复现 | [overview.md §9](overview.md)（复现检查清单） |
| **怎么开发一个可视化界面** | [API-REFERENCE.md §14](API-REFERENCE.md)（GUI 开发建议） |
| 某个函数/参数的准确签名 | [API-REFERENCE.md](API-REFERENCE.md) |
| `--chapter c003` 里的 c003 是什么 | [user-guide §1](user-guide.md) · [overview §2.3](overview.md) |
| 怎么调语速 | [overview §6.2](overview.md) → `--rate 1.15 --force` |
| 云端怎么配环境、怎么传项目 | [cloud-deployment.md](cloud-deployment.md) |
| **SSH 断了任务会不会死** | [cloud-deployment §5.3](cloud-deployment.md)（setsid 方案） |
| 怎么验证环境/项目没问题 | [cloud-deployment §8](cloud-deployment.md)（三层验证清单） |
| 硬件加速有没有生效 | [user-guide §6.3](user-guide.md) |
| 跑出来的音频不对 | [user-guide §6](user-guide.md) |
| 哪些坑千万别踩 | [overview §10](overview.md)（常见陷阱汇总） |
| 怎么回滚 / 看历史 | [git-workflow §6](git-workflow.md) |
| **这个项目哪里做得不好** | [TECH-DEBT.md](TECH-DEBT.md) |
| **想做成软件，需要什么** | [PRODUCTIZATION.md](PRODUCTIZATION.md) |
| 情绪控制 / 分角色朗读能不能做 | [PRODUCTIZATION §2.2](PRODUCTIZATION.md)（**关键结论**） |
| 先做哪些改进最值 | [TECH-DEBT §0](TECH-DEBT.md)（优先级总览） |

## 按角色找文档

**「我只想赶紧生成有声书」**
→ [overview.md](overview.md) §0 一页速览，照着命令敲

**「我要在 AutoDL 上跑」**
→ [overview.md §1.2](overview.md) 配环境 → [cloud-deployment.md](cloud-deployment.md) 看细节

**「我要做个网页/桌面界面」**
→ [API-REFERENCE.md](API-REFERENCE.md)，重点看 §1 设计约定、§2 三种集成方式、§14 GUI 建议

**「我要接手这个项目」**
→ [overview.md §9](overview.md) 复现检查清单 → `python run_tests.py`

**「我怎么知道改动没搞坏别的东西」**
→ `python run_tests.py`（零依赖，**457 个用例**，不需要 GPU、不加载模型）

## 关键事实速查

```text
命令入口            noveltts <命令>
章节 ID 格式        c003（字母 c + 三位补零数字）
配置优先级          内置默认 < config.yaml < config.local.yaml < 命令行
机器相关设置放哪     config.local.yaml（不入库，不会被部署覆盖）
默认输出目录        output/audio/
需要设置的环境变量   NUMBA_CACHE_DIR（项目已自动兜底，通常不用管）

性能参考（RTX 3090，片段平均 60 字符）
  逐段生成          RTF 1.74
  批量生成（推荐）   RTF 0.37 ~ 0.53
  整本 4.5 小时音频  ≈ 1.6 小时

体积参考（每 1 小时音频）
  无损 wav          约 165 MB
  m4a (68kbps)      约 30 MB
```
