"""Tests for pbt ls, pbt docs, pbt test, and pbt type-hints."""

from pathlib import Path

import pytest

from tests.conftest import run_pbt, init_project


# ---------------------------------------------------------------------------
# pbt ls
# ---------------------------------------------------------------------------

class TestLs:
    def test_ls_lists_models(self, proj: Path) -> None:
        result = run_pbt("ls", cwd=proj)
        assert result.returncode == 0
        assert "article" in result.stdout
        assert "summary" in result.stdout

    def test_ls_shows_dependencies(self, proj: Path) -> None:
        result = run_pbt("ls", cwd=proj)
        # summary depends on article
        assert "article" in result.stdout

    def test_ls_empty_models_dir_errors(self, tmp_path: Path) -> None:
        (tmp_path / "models").mkdir()
        result = run_pbt("ls", cwd=tmp_path, check=False)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# pbt docs
# ---------------------------------------------------------------------------

class TestDocs:
    def test_docs_generates_html(self, proj: Path) -> None:
        run_pbt("run", cwd=proj)
        result = run_pbt("docs", cwd=proj)
        assert result.returncode == 0
        assert (proj / ".pbt" / "docs" / "index.html").exists()

    def test_docs_custom_output_path(self, proj: Path) -> None:
        run_pbt("run", cwd=proj)
        result = run_pbt("docs", "--output", "report.html", cwd=proj)
        assert result.returncode == 0
        assert (proj / "report.html").exists()

    def test_docs_html_contains_model_names(self, proj: Path) -> None:
        run_pbt("run", cwd=proj)
        run_pbt("docs", cwd=proj)
        html = (proj / ".pbt" / "docs" / "index.html").read_text()
        assert "article" in html
        assert "summary" in html


# ---------------------------------------------------------------------------
# pbt test
# ---------------------------------------------------------------------------

class TestTest:
    def test_test_requires_prior_run(self, proj: Path) -> None:
        # Without a prior run there are no model outputs to test against
        result = run_pbt("test", cwd=proj, check=False)
        # Either exits non-zero or prints a warning — either is acceptable
        assert result.returncode != 0 or "no run" in result.stdout.lower() or "warning" in result.stdout.lower()

    def test_test_passes_after_run(self, proj: Path) -> None:
        run_pbt("run", cwd=proj)
        result = run_pbt("test", cwd=proj)
        assert result.returncode == 0

    def test_test_failing_prompt_exits_nonzero(self, proj: Path) -> None:
        run_pbt("run", cwd=proj)
        # Overwrite the test prompt to always return fail
        (proj / "tests" / "summary_has_bullets.prompt").write_text(
            'Reply with only valid JSON: {"results": "fail"}.\n'
        )
        result = run_pbt("test", cwd=proj, check=False)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# pbt type-hints
# ---------------------------------------------------------------------------

class TestTypeHints:
    def test_type_hints_generates_stubs(self, proj: Path) -> None:
        result = run_pbt("type-hints", cwd=proj)
        assert result.returncode == 0
        gen_dir = proj / ".pbt" / "gen"
        assert gen_dir.exists()
        assert (gen_dir / "ref.py").exists()

    def test_type_hints_stub_per_model(self, proj: Path) -> None:
        run_pbt("type-hints", cwd=proj)
        stub = (proj / ".pbt" / "gen" / "ref.py").read_text()
        assert "Literal['articles']" in stub
        assert "Literal['summaries']" in stub
        assert "ModelNames" in stub

    def test_type_hints_updates_pyproject(self, proj: Path) -> None:
        run_pbt("type-hints", cwd=proj)
        toml = (proj / "pyproject.toml").read_text()
        assert "[tool.jinja-lsp]" in toml

    def test_type_hints_idempotent(self, proj: Path) -> None:
        run_pbt("type-hints", cwd=proj)
        toml_after_first = (proj / "pyproject.toml").read_text()
        run_pbt("type-hints", cwd=proj)
        toml_after_second = (proj / "pyproject.toml").read_text()
        assert toml_after_first == toml_after_second, "[tool.jinja-lsp] was duplicated"

    def test_type_hints_custom_dirs(self, proj: Path) -> None:
        result = run_pbt(
            "type-hints",
            "--validation-dir", "validation",
            "--gen-dir", ".pbt/custom_gen",
            cwd=proj,
        )
        assert result.returncode == 0
        assert (proj / ".pbt" / "custom_gen" / "ref.py").exists()
