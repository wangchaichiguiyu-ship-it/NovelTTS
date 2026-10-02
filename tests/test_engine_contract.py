"""引擎 API 契约测试：用**桩模块**验证对 ``qwen_tts`` 的调用方式是否正确。

为什么需要这个文件
------------------

:class:`~noveltts.engine.QwenTTSEngine` 是本项目唯一依赖外部 TTS 库的地方。
如果参数名写错（``dtype`` 写成 ``torch_dtype``、``voice_clone_prompt`` 写成
``prompt``…），本机因为**不能加载模型**而不会暴露，到了云端才会在加载完
3.4 GB 权重之后炸掉——那是最贵的失败方式。

所以这里用 ``sys.modules`` 注入一个**假的 ``qwen_tts``**，把真实调用签名
逐条断言下来。它完全不碰 torch 权重、不联网、不占显存，属于纯静态契约测试。

断言依据来自 ``qwen_tts 0.1.1`` 的源码（``inference/qwen3_tts_model.py``）：

* ``Qwen3TTSModel.from_pretrained(path, device_map=..., dtype=..., attn_implementation=...)``
* ``create_voice_clone_prompt(ref_audio=..., ref_text=..., x_vector_only_mode=...)``
* ``generate_voice_clone(text=..., language=..., voice_clone_prompt=...,
  non_streaming_mode=..., **generate_kwargs)``
* 两者都返回 ``(wavs, sample_rate)`` / ``list[VoiceClonePromptItem]``
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from noveltts.config import GenerationConfig, VoiceConfig
from noveltts.engine import QwenTTSEngine, SynthesisResult

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------- #
# 桩模块
# --------------------------------------------------------------------------- #


class _StubModel:
    """假的 ``Qwen3TTSModel``：记录调用参数，返回可控结果。"""

    last_init_kwargs: dict = {}
    last_prompt_kwargs: dict = {}
    last_generate_kwargs: dict = {}
    instances: list["_StubModel"] = []

    def __init__(self) -> None:
        self.prompt_items = [object()]
        self.raises: Exception | None = None
        self.wavs = [np.zeros(2400, dtype=np.float32)]
        self.sample_rate = 24_000
        # 模拟真实模型上用于探测采样率的属性
        self.model = types.SimpleNamespace(sample_rate=24_000)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "_StubModel":
        cls.last_init_kwargs = {"path": path, **kwargs}
        instance = cls()
        cls.instances.append(instance)
        return instance

    def create_voice_clone_prompt(self, **kwargs) -> list:
        type(self).last_prompt_kwargs = dict(kwargs)
        return self.prompt_items

    def generate_voice_clone(self, **kwargs):
        type(self).last_generate_kwargs = dict(kwargs)
        if self.raises is not None:
            raise self.raises
        return list(self.wavs), self.sample_rate


@pytest.fixture
def stub_qwen_tts(monkeypatch: pytest.MonkeyPatch):
    """把假的 ``qwen_tts`` 注入 ``sys.modules``，并在测试结束后恢复原状。"""
    _StubModel.instances.clear()
    _StubModel.last_init_kwargs = {}
    _StubModel.last_prompt_kwargs = {}
    _StubModel.last_generate_kwargs = {}

    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = _StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    return _StubModel


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


@pytest.fixture
def ref_audio(tmp_path: Path) -> Path:
    """参考音频：只要求「文件存在」，桩模型不会真的读它。"""
    path = tmp_path / "ref.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")  # 内容无关紧要
    return path


def make_engine(ref_audio: Path, **voice_kwargs) -> QwenTTSEngine:
    voice = VoiceConfig(ref_audio=str(ref_audio), ref_text="reference transcript", **voice_kwargs)
    return QwenTTSEngine(GenerationConfig(model_path="stub-model", device="cpu"), voice)


# --------------------------------------------------------------------------- #
# 模型加载契约
# --------------------------------------------------------------------------- #


def test_from_pretrained_uses_documented_kwargs(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    kwargs = stub_qwen_tts.last_init_kwargs

    assert kwargs["path"] == "stub-model"
    # 这三个是 qwen_tts 真正接受的参数名；写错会在云端加载后才报错
    assert "device_map" in kwargs, "缺少 device_map"
    assert "dtype" in kwargs, "缺少 dtype（不是 torch_dtype）"
    assert "attn_implementation" in kwargs
    assert kwargs["device_map"] == "cpu"
    assert kwargs["dtype"] is not None
    engine.close()


def test_dtype_maps_to_torch_dtype(stub_qwen_tts, ref_audio: Path) -> None:
    import torch

    engine = make_engine(ref_audio)
    assert stub_qwen_tts.last_init_kwargs["dtype"] is torch.bfloat16
    engine.close()


def test_sample_rate_read_from_model(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    assert engine.sample_rate == 24_000
    engine.close()


def test_model_loaded_only_once(stub_qwen_tts, ref_audio: Path) -> None:
    """权重只能加载一次——这是 16 GB 机器的底线。"""
    engine = make_engine(ref_audio)
    for _ in range(3):
        engine.synthesize(["hello"])
    assert len(stub_qwen_tts.instances) == 1
    engine.close()


# --------------------------------------------------------------------------- #
# 音色 prompt 契约
# --------------------------------------------------------------------------- #


def test_prompt_built_once_at_construction(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    # 构造时就该建好 prompt（而不是每次合成都重算）
    assert stub_qwen_tts.last_prompt_kwargs["ref_audio"] == str(ref_audio)
    assert stub_qwen_tts.last_prompt_kwargs["ref_text"] == "reference transcript"
    assert stub_qwen_tts.last_prompt_kwargs["x_vector_only_mode"] is False
    engine.close()


def test_prompt_is_reused_across_calls(stub_qwen_tts, ref_audio: Path) -> None:
    """反复合成不应重算 prompt（否则每段都要重编码参考音频）。"""
    engine = make_engine(ref_audio)
    first = stub_qwen_tts.last_prompt_kwargs
    for _ in range(5):
        engine.synthesize(["a sentence"])
    assert stub_qwen_tts.last_prompt_kwargs == first
    engine.close()


def test_xvec_mode_flag(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio, x_vector_only_mode=True)
    assert stub_qwen_tts.last_prompt_kwargs["x_vector_only_mode"] is True
    engine.close()


def test_mode_xvec_alias(stub_qwen_tts, ref_audio: Path) -> None:
    """``mode: xvec`` 与 ``x_vector_only_mode: true`` 等价。"""
    engine = make_engine(ref_audio, mode="xvec")
    assert stub_qwen_tts.last_prompt_kwargs["x_vector_only_mode"] is True
    engine.close()


def test_missing_ref_text_falls_back_to_xvec(stub_qwen_tts, tmp_path: Path) -> None:
    """ref_text 为空时应自动退回 xvec 模式，而不是让 ICL 模式报错。"""
    path = tmp_path / "ref.wav"
    path.write_bytes(b"x")
    engine = QwenTTSEngine(
        GenerationConfig(model_path="stub-model", device="cpu"),
        VoiceConfig(ref_audio=str(path), ref_text=""),
    )
    kwargs = stub_qwen_tts.last_prompt_kwargs
    assert kwargs["x_vector_only_mode"] is True
    assert kwargs["ref_text"] is None
    engine.close()


def test_missing_ref_audio_raises_before_loading(stub_qwen_tts, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        QwenTTSEngine(
            GenerationConfig(model_path="stub-model", device="cpu"),
            VoiceConfig(ref_audio=str(tmp_path / "nope.wav"), ref_text="x"),
        )


def test_empty_ref_audio_raises(stub_qwen_tts) -> None:
    with pytest.raises(ValueError):
        QwenTTSEngine(
            GenerationConfig(model_path="stub-model", device="cpu"),
            VoiceConfig(ref_audio="", ref_text="x"),
        )


# --------------------------------------------------------------------------- #
# 生成契约
# --------------------------------------------------------------------------- #


def test_generate_uses_documented_kwargs(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    engine.synthesize(["Hello there."])
    kwargs = stub_qwen_tts.last_generate_kwargs

    assert kwargs["text"] == ["Hello there."]
    assert kwargs["language"] == ["English"]
    assert "voice_clone_prompt" in kwargs, "缺少 voice_clone_prompt"
    assert kwargs["voice_clone_prompt"] is not None
    assert "non_streaming_mode" in kwargs
    assert "max_new_tokens" in kwargs
    engine.close()


def test_language_is_a_list_matching_texts(stub_qwen_tts, ref_audio: Path) -> None:
    """批量时 language 必须与 text 等长（qwen_tts 会校验长度）。"""
    engine = make_engine(ref_audio)
    engine.synthesize(["one", "two", "three"])
    assert stub_qwen_tts.last_generate_kwargs["language"] == ["English"] * 3
    engine.close()


def test_language_override(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    engine.synthesize(["你好"], language="Chinese")
    assert stub_qwen_tts.last_generate_kwargs["language"] == ["Chinese"]
    engine.close()


def test_sampling_params_only_passed_when_set(stub_qwen_tts, ref_audio: Path) -> None:
    """不设采样参数时不要传 None 进去覆盖模型自带的 generate_config。"""
    engine = make_engine(ref_audio)
    engine.synthesize(["x"])
    kwargs = stub_qwen_tts.last_generate_kwargs
    for name in ("temperature", "top_k", "top_p", "repetition_penalty"):
        assert name not in kwargs, f"{name} 为 None 时不应传给模型"
    engine.close()


def test_sampling_params_passed_when_set(stub_qwen_tts, ref_audio: Path) -> None:
    engine = QwenTTSEngine(
        GenerationConfig(
            model_path="stub-model",
            device="cpu",
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
        ),
        VoiceConfig(ref_audio=str(ref_audio), ref_text="x"),
    )
    engine.synthesize(["x"])
    kwargs = stub_qwen_tts.last_generate_kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_k"] == 50
    assert kwargs["top_p"] == 0.9
    assert kwargs["repetition_penalty"] == 1.1
    engine.close()


def test_max_new_tokens_and_non_streaming_passed(stub_qwen_tts, ref_audio: Path) -> None:
    engine = QwenTTSEngine(
        GenerationConfig(model_path="stub-model", device="cpu", max_new_tokens=512,
                         non_streaming_mode=False),
        VoiceConfig(ref_audio=str(ref_audio), ref_text="x"),
    )
    engine.synthesize(["x"])
    kwargs = stub_qwen_tts.last_generate_kwargs
    assert kwargs["max_new_tokens"] == 512
    assert kwargs["non_streaming_mode"] is False
    engine.close()


def test_synthesize_returns_result_object(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    result = engine.synthesize(["Hello there."])

    assert isinstance(result, SynthesisResult)
    assert len(result) == 1
    assert result.sample_rate == 24_000
    assert result.texts == ["Hello there."]
    assert result.generate_seconds >= 0
    engine.close()


def test_synthesize_batch_returns_all_waveforms(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    engine._model.wavs = [np.zeros(100, dtype=np.float32) for _ in range(3)]
    result = engine.synthesize(["a", "b", "c"])
    assert len(result.waveforms) == 3
    engine.close()


def test_synthesize_strips_whitespace(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    engine.synthesize(["  padded text  "])
    assert stub_qwen_tts.last_generate_kwargs["text"] == ["padded text"]
    engine.close()


def test_synthesize_rejects_empty_list(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    with pytest.raises(ValueError):
        engine.synthesize([])
    engine.close()


def test_synthesize_rejects_blank_text(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    with pytest.raises(ValueError):
        engine.synthesize(["   "])
    engine.close()


# --------------------------------------------------------------------------- #
# 生命周期
# --------------------------------------------------------------------------- #


def test_closed_engine_refuses_work(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    engine.close()
    with pytest.raises(RuntimeError):
        engine.synthesize(["x"])


def test_close_is_idempotent(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    engine.close()
    engine.close()  # 不应抛异常


def test_context_manager_closes(stub_qwen_tts, ref_audio: Path) -> None:
    with make_engine(ref_audio) as engine:
        assert engine.synthesize(["x"])
    with pytest.raises(RuntimeError):
        engine.synthesize(["x"])


def test_describe_mentions_key_settings(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio)
    text = engine.describe()
    assert "stub-model" in text
    assert "cpu" in text
    assert "voice_mode=icl" in text
    engine.close()


def test_describe_reports_xvec_mode(stub_qwen_tts, ref_audio: Path) -> None:
    engine = make_engine(ref_audio, mode="xvec")
    assert "voice_mode=xvec" in engine.describe()
    engine.close()


def test_engine_failure_propagates(stub_qwen_tts, ref_audio: Path) -> None:
    """引擎异常必须向上抛出，交给 synthesis 层做重试与隔离。"""
    engine = make_engine(ref_audio)
    engine._model.raises = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        engine.synthesize(["x"])
    engine.close()


def test_real_module_is_not_used(stub_qwen_tts, ref_audio: Path) -> None:
    """确保测试期间 sys.modules 里是我们注入的桩。"""
    assert sys.modules["qwen_tts"].Qwen3TTSModel is stub_qwen_tts  # type: ignore[attr-defined]
