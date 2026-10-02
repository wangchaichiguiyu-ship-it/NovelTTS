"""零依赖测试运行器：不需要安装 pytest 也能跑完整测试套件。

为什么要有这个文件
------------------

``tests/`` 下的用例是按 **pytest 风格**写的（``test_*`` 函数、``assert``、
``@pytest.fixture``、``@pytest.mark.parametrize``），因此在 CI 或本地装了
pytest 时直接 ``pytest tests/`` 即可。

但你的 conda 环境（``qwen3-tts``）里没有 pytest，而且沙箱不允许往
``~/.local`` 写包。为了不必为了跑测试而改动环境，这个脚本内置了一个
**极简 pytest 兼容层**，把 ``pytest`` 注入 ``sys.modules``，
然后自己发现并执行用例。

用法::

    python run_tests.py              # 跑全部
    python run_tests.py normalize    # 只跑名字含 normalize 的文件
    python run_tests.py -v           # 打印每个用例名

支持的 pytest 特性（够用即可，不做完整实现）：

* ``@pytest.fixture``（含函数参数注入、fixture 依赖 fixture）
* ``@pytest.mark.parametrize``（含 ``pytest.param`` 与 ``id``）
* ``pytest.raises`` / ``pytest.approx``
* ``pytest.skip`` / ``pytest.importorskip``

**本运行器绝不会加载 Qwen3-TTS 模型**：涉及推理的用例都通过
``FakeSynthesisEngine`` 注入，产出的是正弦波。
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import os
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# 极简 pytest 兼容层
# --------------------------------------------------------------------------- #


class Skipped(Exception):
    """对应 pytest.skip。"""


class _RaisesContext:
    """``with pytest.raises(Exc, match="..."):`` 上下文管理器。"""

    def __init__(
        self,
        expected: type[BaseException] | tuple[type[BaseException], ...],
        match: str | None = None,
    ) -> None:
        self.expected = expected
        self.match = match
        self.value: BaseException | None = None

    def __enter__(self) -> "_RaisesContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            names = (
                self.expected.__name__
                if isinstance(self.expected, type)
                else "/".join(e.__name__ for e in self.expected)
            )
            raise AssertionError(f"期望抛出 {names}，但没有异常")
        if not issubclass(exc_type, self.expected):
            return False
        self.value = exc
        if self.match is not None:
            import re as _re

            if not _re.search(self.match, str(exc)):
                raise AssertionError(
                    f"异常信息不匹配：期望匹配 {self.match!r}，实际为 {str(exc)!r}"
                )
        return True


class _Approx:
    """``pytest.approx`` 的简化版。"""

    def __init__(self, expected: Any, rel: float = 1e-6, abs: float = 1e-12) -> None:
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other: Any) -> bool:
        return abs(float(other) - float(self.expected)) <= max(
            self.abs, self.rel * abs(float(self.expected))
        )

    def __repr__(self) -> str:
        return f"approx({self.expected})"


class _Param:
    """``pytest.param(...)`` 的返回值。"""

    def __init__(self, values: tuple[Any, ...], id: str | None = None) -> None:
        self.values = values
        self.id = id


class _Mark:
    """``@pytest.mark.parametrize``。"""

    def parametrize(
        self,
        argnames: str | list[str],
        argvalues: list[Any],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        names = (
            [name.strip() for name in argnames.split(",")]
            if isinstance(argnames, str)
            else list(argnames)
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cases: list[tuple[dict[str, Any], str]] = []
            for position, raw in enumerate(argvalues):
                if isinstance(raw, _Param):
                    values, custom_id = raw.values, raw.id
                elif len(names) == 1:
                    # 单参数：整个 argvalue 就是那一个值，**不拆元组**。
                    # 否则参数化成 (7, 5) 这种元组会被撕成两个值，
                    # 函数收到 7 而不是 (7, 5)。
                    values, custom_id = (raw,), None
                else:
                    values, custom_id = tuple(raw), None
                label = custom_id or (ids[position] if ids and position < len(ids) else None)
                if label is None:
                    # 用能区分用例的短标签：去掉空白后截断，避免 "a b" 与 "a  b"
                    # 因为空格被压缩而看起来一样
                    label = "-".join(
                        "".join(str(v).split())[:28] or "empty" for v in values
                    )
                cases.append((dict(zip(names, values)), label))
            func._parametrize = getattr(func, "_parametrize", []) + [(cases,)]  # type: ignore[attr-defined]
            return func

        return decorator

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """忽略未实现的 marker（如 ``@pytest.mark.slow``）。"""

        def noop(*args: Any, **kwargs: Any) -> Any:
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def inner(func: Callable[..., Any]) -> Callable[..., Any]:
                return func

            return inner

        return noop


_FIXTURE_REGISTRY: dict[str, Callable[..., Any]] = {}


class _PytestShim(types.ModuleType):
    """一个够用的 pytest 替身。"""

    def __init__(self) -> None:
        super().__init__("pytest")
        self.mark = _Mark()
        self.param = _Param

    # --- 断言辅助 ---
    @staticmethod
    def raises(
        expected: type[BaseException] | tuple[type[BaseException], ...],
        match: str | None = None,
    ) -> _RaisesContext:
        return _RaisesContext(expected, match=match)

    @staticmethod
    def approx(expected: Any, rel: float = 1e-6, abs: float = 1e-12) -> _Approx:
        return _Approx(expected, rel=rel, abs=abs)

    # --- 跳过 ---
    @staticmethod
    def skip(reason: str = "") -> None:
        raise Skipped(reason)

    @staticmethod
    def importorskip(name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return __import__(name)
        except ImportError:
            raise Skipped(f"缺少模块 {name}") from None

    @staticmethod
    def fail(reason: str = "") -> None:
        raise AssertionError(reason)

    # --- fixture ---
    def fixture(
        self,
        func: Callable[..., Any] | None = None,
        *,
        scope: str = "function",
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        def register(target: Callable[..., Any]) -> Callable[..., Any]:
            _FIXTURE_REGISTRY[name or target.__name__] = target
            return target

        if func is not None:
            return register(func)
        return register


def install_shim() -> None:
    """把替身装进 sys.modules（若真 pytest 可用则不装）。"""
    try:
        import pytest  # noqa: F401

        return
    except ImportError:
        pass
    sys.modules["pytest"] = _PytestShim()


# --------------------------------------------------------------------------- #
# 用例发现与执行
# --------------------------------------------------------------------------- #


def load_module(path: Path) -> types.ModuleType | None:
    """按文件路径导入测试模块。"""
    name = f"nttests_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_fixture(name: str, cache: dict[str, Any], depth: int = 0) -> Any:
    """递归解析 fixture（fixture 也能依赖 fixture）。"""
    if name in cache:
        return cache[name]
    if depth > 10:
        raise RuntimeError(f"fixture 依赖过深（是否成环？）：{name}")
    factory = _FIXTURE_REGISTRY.get(name)
    if factory is None:
        raise LookupError(f"找不到 fixture：{name}")

    kwargs = {
        param: resolve_fixture(param, cache, depth + 1)
        for param in inspect.signature(factory).parameters
        if param not in {"self"}
    }
    value = factory(**kwargs)
    cache[name] = value
    return value


def iter_cases(module: types.ModuleType) -> Iterator[tuple[str, Callable[..., Any], dict[str, Any], bool]]:
    """产出 ``(名字, 函数, 绑定参数, 是否需要 fixture)``。"""
    for name, func in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(func):
            continue
        if inspect.isfunction(func) and func.__module__ != module.__name__:
            continue
        param_sets = getattr(func, "_parametrize", None)
        if param_sets:
            for cases in param_sets:
                for values, label in cases[0]:
                    yield f"{name}[{label}]", func, values, True
        else:
            yield name, func, {}, True


@dataclass
class _Captured:
    """``capsys.readouterr()`` 的返回值。"""

    out: str = ""
    err: str = ""


class _CaptureFixture:
    """``capsys`` 的等价实现：捕获 stdout / stderr。

    关键点：logging 的 ``StreamHandler`` 会**长期持有** ``sys.stderr`` 的引用
    （第一次 ``setup_logging`` 时就绑定了）。如果 :meth:`readouterr` 只清空
    自己的缓冲对象，那么日志会继续写进旧缓冲，第二次读就永远是空的。
    因此这里让 tee 在每次读取时**就地**从底层缓冲取值并清空。
    """

    def __init__(self, passthrough: bool = True) -> None:
        self._out = io.StringIO()
        self._err = io.StringIO()
        self._passthrough = passthrough

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def readouterr(self) -> _Captured:
        out, err = self._out.getvalue(), self._err.getvalue()
        # 就地清空：保持同一对象，让仍然持有引用的 StreamHandler 继续可用
        self._out.seek(0)
        self._out.truncate(0)
        self._err.seek(0)
        self._err.truncate(0)
        return _Captured(out=out, err=err)


class _Tee(io.TextIOBase):
    """同时写入捕获缓冲与真实流。"""

    def __init__(self, buffer: io.StringIO, passthrough: Any) -> None:
        self._buffer = buffer
        self._passthrough = passthrough

    def write(self, data: str) -> int:  # noqa: D102
        self._buffer.write(data)
        try:
            self._passthrough.write(data)
        except Exception:  # noqa: BLE001
            pass
        return len(data)

    def flush(self) -> None:  # noqa: D102
        try:
            self._passthrough.flush()
        except Exception:  # noqa: BLE001
            pass


_MISSING = object()

#: 当前正在执行的用例的 monkeypatch 实例。
#: pytest 把 ``monkeypatch`` 实现为内置 fixture，我们也照做：注册一个
#: 返回当前实例的 fixture，这样 ``monkeypatch`` 既能当参数注入，也能被
#: 其它 fixture 依赖。
_CURRENT_PATCHER: "_MonkeyPatch | None" = None


class _MonkeyPatch:
    """``monkeypatch`` 的等价实现。

    记录所有改动，用例结束后由 runner 调用 :meth:`undo` 回滚。
    支持最常用的 ``setattr`` / ``delattr`` / ``setitem`` / ``delitem`` /
    ``setenv`` / ``delenv`` / ``syspath_prepend`` / ``chdir``。
    """

    def __init__(self) -> None:
        self._undo: list[Callable[[], None]] = []

    # --- 属性 ---
    def setattr(self, target: Any, name: str, value: Any = _MISSING, raising: bool = True) -> None:
        # 兼容 monkeypatch.setattr(obj, "name", value) 与 monkeypatch.setattr("pkg.mod", value)
        if value is _MISSING and isinstance(target, str):
            module_name, _, attr = target.rpartition(".")
            target, name, value = __import__(module_name, fromlist=[attr]), attr, name
        old = getattr(target, name, _MISSING)
        if old is _MISSING and raising:
            raise AttributeError(f"{target!r} 没有属性 {name!r}")
        setattr(target, name, value)
        if old is _MISSING:
            self._undo.append(lambda: delattr(target, name))
        else:
            self._undo.append(lambda: setattr(target, name, old))

    def delattr(self, target: Any, name: str, raising: bool = True) -> None:
        old = getattr(target, name, _MISSING)
        if old is _MISSING:
            if raising:
                raise AttributeError(f"{target!r} 没有属性 {name!r}")
            return
        delattr(target, name)
        self._undo.append(lambda: setattr(target, name, old))

    # --- 容器项 ---
    def setitem(self, mapping: Any, key: Any, value: Any) -> None:
        old = mapping.get(key, _MISSING)
        mapping[key] = value
        if old is _MISSING:
            self._undo.append(lambda: mapping.pop(key, None))
        else:
            self._undo.append(lambda: mapping.__setitem__(key, old))

    def delitem(self, mapping: Any, key: Any, raising: bool = True) -> None:
        old = mapping.get(key, _MISSING)
        if old is _MISSING:
            if raising:
                raise KeyError(key)
            return
        del mapping[key]
        self._undo.append(lambda: mapping.__setitem__(key, old))

    # --- 环境变量 ---
    def setenv(self, name: str, value: str, prepend: bool = False) -> None:
        old = os.environ.get(name, _MISSING)
        os.environ[name] = (
            f"{value}{os.pathsep}{old}" if prepend and old is not _MISSING else value
        )
        if old is _MISSING:
            self._undo.append(lambda: os.environ.pop(name, None))
        else:
            self._undo.append(lambda: os.environ.__setitem__(name, old))

    def delenv(self, name: str, raising: bool = True) -> None:
        old = os.environ.get(name, _MISSING)
        if old is _MISSING:
            if raising:
                raise KeyError(name)
            return
        del os.environ[name]
        self._undo.append(lambda: os.environ.__setitem__(name, old))

    # --- 杂项 ---
    def syspath_prepend(self, path: Any) -> None:
        self.setitem(sys.path, 0, str(path))

    def chdir(self, path: Any) -> None:
        old = os.getcwd()
        os.chdir(path)
        self._undo.append(lambda: os.chdir(old))

    def undo(self) -> None:
        while self._undo:
            self._undo.pop()()


def register_builtin_fixtures() -> None:
    """注册 pytest 内置 fixture 的等价物（只实现用到的那些）。"""
    import tempfile

    def factory() -> Path:
        tmp = tempfile.mkdtemp()
        _TEMP_DIRS.append(tmp)
        return Path(tmp)

    def monkeypatch() -> "_MonkeyPatch":
        if _CURRENT_PATCHER is None:  # pragma: no cover - 正常流程不会走到
            raise RuntimeError("monkeypatch 只能在用例执行期间使用")
        return _CURRENT_PATCHER

    _FIXTURE_REGISTRY.setdefault("tmp_path", factory)
    _FIXTURE_REGISTRY.setdefault("tmpdir", factory)
    _FIXTURE_REGISTRY.setdefault("monkeypatch", monkeypatch)


_TEMP_DIRS: list[str] = []


def run_module(path: Path, *, verbose: bool = False) -> tuple[int, int, int]:
    """执行一个测试文件，返回 ``(通过, 失败, 跳过)``。"""
    module = load_module(path)
    if module is None:
        print(f"  !! 无法加载 {path.name}")
        return 0, 1, 0

    passed = failed = skipped = 0

    for case_name, func, bound, _ in iter_cases(module):
        cache: dict[str, Any] = {}
        code, ok = _invoke(func, bound, cache, case_name, verbose)

        if code == 0:
            passed += 1
            if verbose:
                print(f"    PASS {case_name}")
        elif code == 1:
            failed += 1
        else:
            skipped += 1
            if verbose:
                print(f"    SKIP {case_name}")

    return passed, failed, skipped


def _cleanup_temp_dirs() -> None:
    """删除本次运行创建的临时目录。"""
    import shutil

    while _TEMP_DIRS:
        path = _TEMP_DIRS.pop()
        shutil.rmtree(path, ignore_errors=True)


def _invoke(
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    cache: dict[str, Any],
    case_name: str,
    verbose: bool,
) -> tuple[int, bool]:
    """调用一个用例，返回 ``(状态码, 是否成功)``；状态码 0 通过 / 1 失败 / 2 跳过。"""
    signature = inspect.signature(func)
    capture: _CaptureFixture | None = None
    patcher = _MonkeyPatch()
    global _CURRENT_PATCHER
    _CURRENT_PATCHER = patcher
    try:
        for param in signature.parameters:
            if param in kwargs:
                continue
            if param in {"capsys", "capfd"}:
                capture = _CaptureFixture()
                capture.start()
                kwargs[param] = capture
                continue
            if param == "monkeypatch":
                kwargs[param] = patcher
                continue
            kwargs[param] = resolve_fixture(param, cache)
    except Skipped as exc:
        if verbose:
            print(f"    SKIP {case_name}: {exc}")
        return 2, False
    except Exception as exc:  # noqa: BLE001
        print(f"    ERROR {case_name}: fixture 解析失败：{exc}")
        return 1, False

    original_out, original_err = sys.stdout, sys.stderr
    if capture is not None:
        sys.stdout = _Tee(capture._out, original_out)   # type: ignore[assignment]
        sys.stderr = _Tee(capture._err, original_err)   # type: ignore[assignment]

    try:
        func(**kwargs)
        return 0, True
    except Skipped as exc:
        if verbose:
            print(f"    SKIP {case_name}: {exc}")
        return 2, False
    except AssertionError:
        print(f"    FAIL {case_name}")
        traceback.print_exc(limit=3)
        return 1, False
    except Exception:  # noqa: BLE001
        print(f"    ERROR {case_name}")
        traceback.print_exc(limit=3)
        return 1, False
    finally:
        patcher.undo()
        _CURRENT_PATCHER = None
        if capture is not None:
            sys.stdout, sys.stderr = original_out, original_err


def main(argv: list[str]) -> int:
    """运行器入口。"""
    args = [a for a in argv[1:] if not a.startswith("-")]
    verbose = "-v" in argv or "--verbose" in argv

    install_shim()
    register_builtin_fixtures()

    if not TESTS_DIR.is_dir():
        print(f"找不到测试目录：{TESTS_DIR}")
        return 2

    files = sorted(TESTS_DIR.glob("test_*.py"))
    if args:
        files = [f for f in files if any(token in f.stem for token in args)]
    if not files:
        print("没有匹配的测试文件")
        return 2

    # 先加载 conftest.py，把它注册的 fixture 放进全局注册表。
    # pytest 会自动做这件事，我们这里要手动来一次。
    conftest = TESTS_DIR / "conftest.py"
    if conftest.is_file():
        try:
            load_module(conftest)
        except Exception:  # noqa: BLE001
            print("!! 加载 conftest.py 失败")
            traceback.print_exc(limit=3)
            return 1

    total_passed = total_failed = total_skipped = 0
    print()
    try:
        for path in files:
            passed, failed, skipped = run_module(path, verbose=verbose)
            total_passed += passed
            total_failed += failed
            total_skipped += skipped
            status = "OK  " if failed == 0 else "FAIL"
            extra = f" ({skipped} skipped)" if skipped else ""
            print(f"  [{status}] {path.name:<28} {passed:>3} passed{extra}")
    finally:
        _cleanup_temp_dirs()

    print()
    print(f"合计：{total_passed} 通过 / {total_failed} 失败 / {total_skipped} 跳过")
    print()
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
