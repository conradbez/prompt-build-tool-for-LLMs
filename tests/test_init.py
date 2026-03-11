"""Tests for `pbt init`."""

from pathlib import Path

import pytest

from tests.conftest import run_pbt, init_project


# ---------------------------------------------------------------------------
# Expected files
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    "README.md",
    "client.py",
    "models/articles.prompt",
    "models/summaries.prompt",
    "tests/summary_has_bullets.prompt",
    "validation/articles.py",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_creates_expected_files(tmp_path: Path) -> None:
    result = run_pbt("init", "myproject", cwd=tmp_path)
    assert result.returncode == 0
    proj = tmp_path / "myproject"
    for rel in EXPECTED_FILES:
        assert (proj / rel).exists(), f"Missing: {rel}"


def test_init_default_provider_is_gemini(tmp_path: Path) -> None:
    run_pbt("init", "proj", cwd=tmp_path)
    client = (tmp_path / "proj" / "client.py").read_text()
    assert "genai" in client or "GEMINI_API_KEY" in client


@pytest.mark.parametrize("provider,marker", [
    ("gemini",    "GEMINI_API_KEY"),
    ("openai",    "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
])
def test_init_provider_client(tmp_path: Path, provider: str, marker: str) -> None:
    run_pbt("init", "proj", "--provider", provider, cwd=tmp_path)
    client = (tmp_path / "proj" / "client.py").read_text()
    assert marker in client


@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic"])
def test_init_force_all_providers(tmp_path: Path, provider: str) -> None:
    """--force with each provider overwrites all files without error."""
    run_pbt("init", "proj", "--provider", provider, cwd=tmp_path)
    result = run_pbt("init", "proj", "--force", "--provider", provider, cwd=tmp_path)
    assert result.returncode == 0
    assert "skipped" not in result.stdout
    proj = tmp_path / "proj"
    for rel in EXPECTED_FILES:
        assert (proj / rel).exists(), f"Missing after --force: {rel}"
    client = (proj / "client.py").read_text()
    assert {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider] in client


def test_init_skips_existing_files(tmp_path: Path) -> None:
    run_pbt("init", "proj", cwd=tmp_path)
    article = tmp_path / "proj" / "models" / "articles.prompt"
    article.write_text("custom content")

    result = run_pbt("init", "proj", cwd=tmp_path)
    assert result.returncode == 0
    assert article.read_text() == "custom content", "Existing file was overwritten without --force"
    assert "skipped" in result.stdout


def test_init_force_overwrites(tmp_path: Path) -> None:
    run_pbt("init", "proj", cwd=tmp_path)
    article = tmp_path / "proj" / "models" / "articles.prompt"
    article.write_text("custom content")

    run_pbt("init", "proj", "--force", cwd=tmp_path)
    assert article.read_text() != "custom content", "--force should have overwritten the file"


def test_init_default_project_name(tmp_path: Path) -> None:
    run_pbt("init", cwd=tmp_path)
    assert (tmp_path / "generate_articles_example" / "models" / "articles.prompt").exists()


def test_init_output_mentions_run_command(tmp_path: Path) -> None:
    result = run_pbt("init", "proj", cwd=tmp_path)
    assert "pbt run" in result.stdout
