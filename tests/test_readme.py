"""The README must keep up with the code: modules, scripts, settings, environment variables and endpoints."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf8")
APP = (ROOT / "app.py").read_text(encoding="utf8")
DB = (ROOT / "database.py").read_text(encoding="utf8")
SETTINGS = (ROOT / "templates" / "settings.html").read_text(encoding="utf8")
BACKENDS = (ROOT / "llm_backends.py").read_text(encoding="utf8")


def test_every_module_and_script_is_in_the_project_structure():
    for folder in (ROOT, ROOT / "scripts"):
        for f in folder.glob("*.py"):
            assert f.name in README, f"{f.name} is not mentioned in the README"
    for d in (ROOT / "data").iterdir():
        assert f"{d.name}/" in README, f"data/{d.name} is not mentioned in the README"


def test_every_setting_is_documented():
    keys = set(re.findall(r'get_setting\("([a-z_]+)"', APP))
    keys |= set(re.findall(r"payload\.([a-z_]+)\s*=", SETTINGS))
    keys |= set(re.findall(r'\bs\.get\("([a-z_]+)"', BACKENDS))          # read by the local backends
    assert {"api_provider", "max_rpm", "rag_strictness", "n_ctx", "chunk_chars"} <= keys
    for k in sorted(keys):
        assert f"`{k}`" in README, f"setting {k} is not documented in the README"


def test_every_environment_variable_is_documented():
    env = set(re.findall(r'environ(?:\.get)?[\("\[]+"([A-Z_]+)"', APP + DB))
    assert {"SD_DB_DIR", "SD_PORT", "SD_CHUNK_CHARS", "SD_MAX_RPM"} <= env
    for e in sorted(env):
        assert e in README, f"environment variable {e} is not documented in the README"


def test_every_endpoint_is_documented():
    for route in re.findall(r'@app\.route\("(/api/[^"]+)"', APP):
        last = [s for s in route.split("/") if s and not s.startswith("<")][-1]
        assert last in README, f"endpoint {route} is not documented in the README"


def test_every_corpus_and_licence_is_stated():
    for needle in ("data/greek", "data/vulgate", "data/syriac", "LXX", "`Vg`", "`PES`", "CC BY-SA 4.0",
                   "CC BY-NC 4.0", "non-commercial", "LICENSES.md"):
        assert needle in README, needle


def test_no_stale_statements_from_the_original_readme():
    for stale in ("The only external dependency is a free Gemini API key", "Python 3.10+", "SD_BIBLE_DIR",
                  "Greek-specific", "Yale color palette"):
        assert stale not in README, stale
    assert "Python 3.11.9" in README                                   # as in pyproject.toml
    assert "requires-python" not in README
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf8")
    assert 'requires-python = ">=3.11.9"' in pyproject
