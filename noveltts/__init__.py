"""NovelTTS —— 小说文本预处理 + Qwen3-TTS 有声书生成流水线。

流水线总览（每一环都可以单独运行，中间产物全部落盘）：

    EPUB/TXT ──parse──▶ Chapter ──normalize──▶ 洗稿文本 ──segment──▶ segments.json
                                                                          │
                                             audio/segments/*.wav ◀──synth──┘
                                                       │
                                        chapter.wav ◀──concat──┘
                                                       │
                                        chapter.m4a ◀──encode──┘

设计约束（针对 16 GB 统一内存的 Mac，同时也适用于云端）：

* **不在导入期加载重型依赖**。``noveltts`` 包本身只依赖标准库；
  numpy / soundfile / torch / qwen_tts 全部在使用处延迟导入。
  因此 ``parse`` / ``normalize`` / ``segment`` 三个子命令是纯 CPU 轻量操作，
  且测试可以在不安装 torch 的环境里跑。
* **逐段落盘，绝不把整本有声书读进内存**。合成阶段每生成一段就立刻
  以 PCM_16 写盘并释放张量；拼接与压缩阶段以固定块流式读写。
* **可断点续跑**。已存在的片段 wav 会被跳过，中断后重跑不会重复烧算力。
"""

from __future__ import annotations

import os as _os
import sys as _sys
import tempfile as _tempfile

__all__ = [
    "AUDIO_DTYPE",
    "SAMPLE_RATE",
    "__version__",
    "ensure_numba_cache_dir",
]

#: 输出音频采样率。Qwen3-TTS-12Hz 语音 tokenizer 的解码输出固定为 24 kHz。
SAMPLE_RATE: int = 24_000

#: 落盘音频位深。PCM_16 相比 float32 省一半磁盘与 IO，且听感无损。
AUDIO_DTYPE: str = "PCM_16"

__version__ = "1.0.0"


def ensure_numba_cache_dir() -> str | None:
    """确保 ``import librosa``（进而 ``import qwen_tts``）不会因为 numba 缓存而失败。

    背景
    ----

    ``librosa`` 里有一段被 ``@numba.jit(cache=True)`` 装饰的代码。
    numba 在**导入期**就会为它选择一个缓存位置，而首选位置是
    ``<librosa>/core/__pycache__``。如果那个目录不可写——例如 conda 环境装在
    只读位置、或者进程没有该目录的写权限——numba 找不到任何可用位置，
    直接抛::

        RuntimeError: cannot cache function '__o_fold': no locator available

    这个异常发生在 ``import librosa`` 期间，于是**整个 ``qwen_tts`` 都导入不了**，
    表现为模型还没开始加载就崩掉。它跟本项目无关，但会让 ``synth`` 完全跑不起来。

    处理方式
    --------

    检测 «librosa 的 ``__pycache__`` 是否可写»，不可写时把 ``NUMBA_CACHE_DIR``
    指向一个可写的临时目录。numba 的 ``UserProvidedCacheLocator`` 会优先使用
    这个变量，于是导入恢复正常，且**不会改动 librosa 自身的任何文件**。

    这个函数在 ``import noveltts`` 时自动执行，所以用户不需要手工设环境变量。

    Returns:
        被设置的缓存目录；无需处理时返回 ``None``。
    """
    # 已经显式设置过就尊重用户的配置
    if _os.environ.get("NUMBA_CACHE_DIR"):
        return None

    # 没有 librosa / numba 的环境（例如只跑 parse/segment）完全不需要处理
    if "librosa" not in _sys.modules:
        try:
            import importlib.util as _util

            spec = _util.find_spec("librosa")
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.origin:
            return None
        librosa_core = _os.path.dirname(spec.origin)
    else:
        librosa_core = _os.path.dirname(_sys.modules["librosa"].__file__ or "")

    if not librosa_core:
        return None

    pycache = _os.path.join(librosa_core, "__pycache__")
    if _os.path.isdir(pycache) and _os.access(pycache, _os.W_OK):
        return None  # 可写，numba 用默认位置即可，不必打扰它

    cache_dir = _os.path.join(_tempfile.gettempdir(), "noveltts-numba-cache")
    try:
        _os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return None

    _os.environ["NUMBA_CACHE_DIR"] = cache_dir
    return cache_dir


# 在包导入时就把 numba 缓存问题处理掉。
# 必须放在这里：librosa 是在 engine 里延迟导入的，而这个函数要在那之前生效。
ensure_numba_cache_dir()
