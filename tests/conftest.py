"""pytest 公共 fixture。

**重要约定：本测试套件绝不加载 Qwen3-TTS 模型。**
所有涉及推理的测试都用 :class:`noveltts.engine.FakeSynthesisEngine`
（产出正弦波），因此可以在没有 GPU、没装 torch 的机器上跑，
也不会占用一分钱算力、一兆显存。

需要 numpy / soundfile 的测试用 ``pytest.importorskip`` 守护，
缺依赖时自动跳过而不是报错。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让测试可以直接 import noveltts（无需安装）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noveltts.config import (  # noqa: E402
    AppConfig,
    AudioConfig,
    GenerationConfig,
    ProjectConfig,
    SegmentConfig,
    TextRulesConfig,
    VoiceConfig,
)


@pytest.fixture
def project_root() -> Path:
    """仓库根目录。"""
    return PROJECT_ROOT


@pytest.fixture
def tmp_config(tmp_path: Path) -> AppConfig:
    """指向临时目录的配置（不会污染真实 output/）。"""
    config = AppConfig(
        project=ProjectConfig(
            root=str(tmp_path),
            source_dir="sources",
            work_dir="output/work",
            output_dir="output/audio",
            cache_dir="output/cache",
        ),
        parse=None,  # type: ignore[arg-type]
        text_rules=TextRulesConfig(),
        segment=SegmentConfig(),
        voice=VoiceConfig(ref_audio="ref.wav", ref_text="test"),
        generation=GenerationConfig(model_path="dummy", device="cpu", num_retries=0),
        audio=AudioConfig(),
    )
    from noveltts.config import ParseConfig

    config.parse = ParseConfig()
    config.project.resolve(tmp_path)
    config.project.ensure_dirs()
    return config


@pytest.fixture
def sample_epub(tmp_path: Path) -> Path:
    """生成一个结构完整的迷你 EPUB（含 nav 目录、正文、图片页）。"""
    import zipfile

    path = tmp_path / "sample.epub"
    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Novel</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="Text/nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c1" href="Text/c1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="Text/c2.xhtml" media-type="application/xhtml+xml"/>
    <item id="img" href="Text/img.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/><itemref idref="img"/><itemref idref="c2"/>
  </spine>
</package>"""

    nav = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Contents</title></head><body>
<nav epub:type="toc"><ol>
  <li><a href="c1.xhtml">Chapter One: The Start</a></li>
  <li><a href="c2.xhtml">Chapter Two: The End</a></li>
</ol></nav></body></html>"""

    def chapter(title: str, body: str) -> str:
        paragraphs = "".join(f"<p>{para}</p>" for para in body.split("\n\n"))
        return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{title}</title></head>
<body><h1>{title}</h1>{paragraphs}</body></html>"""

    c1 = chapter(
        "Chapter One: The Start",
        "The 1500 meter race began at 3 p.m. Vol. 8 of the guide says so.\n\n"
        '"I am ready," she said. Mr. Smith nodded.',
    )
    c2 = chapter(
        "Chapter Two: The End",
        "She won by 870-120 points in 1990-1995 style.\n\n"
        "It cost $1,500.00 and 100 percent of her savings.",
    )
    img = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Image</title></head>
<body><p><img src="../Images/x.jpg" alt=""/></p></body></html>"""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/Text/nav.xhtml", nav)
        archive.writestr("OEBPS/Text/c1.xhtml", c1)
        archive.writestr("OEBPS/Text/c2.xhtml", c2)
        archive.writestr("OEBPS/Text/img.xhtml", img)
    return path


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """一个带多行标题的 TXT 样本。"""
    path = tmp_path / "sample.txt"
    path.write_text(
        """Table of Contents

Loss 1:
The First Loss

The 1500 meter race began. "I am ready," she said.

Mr. Smith nodded and left.

Intermission:
A Short Break

Nothing much happened here. It was quiet.
""",
        encoding="utf-8",
    )
    return path
