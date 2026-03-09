"""Tests for `pbt run`."""

import textwrap
from pathlib import Path

import pytest

from tests.conftest import run_pbt, init_project, STUB_CLIENT_JSON_PY


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_succeeds(proj: Path) -> None:
    result = run_pbt("run", cwd=proj)
    assert result.returncode == 0
    assert "succeeded" in result.stdout


def test_run_creates_output_files(proj: Path) -> None:
    run_pbt("run", cwd=proj)
    outputs = list((proj / "outputs").glob("*.txt"))
    assert len(outputs) >= 2, "Expected output files for article and summary"


def test_run_all_models_succeed(proj: Path) -> None:
    result = run_pbt("run", cwd=proj)
    assert "errored" not in result.stdout
    assert "skipped" not in result.stdout


def test_run_with_promptdata(proj: Path) -> None:
    result = run_pbt("run", "--promptdata", "topic=quantum computing", cwd=proj)
    assert result.returncode == 0
    assert "succeeded" in result.stdout


def test_run_select_single_model(proj: Path) -> None:
    # First full run to populate cache
    run_pbt("run", cwd=proj)
    # Select only summary (depends on article)
    result = run_pbt("run", "--select", "summary", cwd=proj)
    assert result.returncode == 0
    assert "succeeded" in result.stdout


def test_run_select_unknown_model_errors(proj: Path) -> None:
    result = run_pbt("run", "--select", "nonexistent_model", cwd=proj, check=False)
    assert result.returncode != 0


def test_run_caches_identical_prompts(proj: Path) -> None:
    run_pbt("run", cwd=proj)
    result = run_pbt("run", cwd=proj)
    # Second run should show 0 ms times (all cached)
    assert "(0 ms)" in result.stdout or "0ms" in result.stdout


def test_run_json_output_format(tmp_path: Path) -> None:
    proj = init_project(tmp_path)
    # Override summary to declare json output and use a JSON-returning stub
    (proj / "models" / "summary.prompt").write_text(
        '{{ config(output_format="json") }}\n'
        'Return JSON {"title": "t", "summary": "s"}.\n'
        "{{ ref('article') }}\n"
    )
    (proj / "models" / "client.py").write_text(STUB_CLIENT_JSON_PY)
    result = run_pbt("run", cwd=proj)
    assert result.returncode == 0


def test_run_stores_results_in_db(proj: Path) -> None:
    run_pbt("run", cwd=proj)
    assert (proj / ".pbt" / "pbt.db").exists()


def test_run_invalid_promptdata_format_errors(proj: Path) -> None:
    result = run_pbt("run", "--promptdata", "badformat", cwd=proj, check=False)
    assert result.returncode != 0


def test_run_validation_failure_marks_model_error(tmp_path: Path) -> None:
    proj = init_project(tmp_path)
    # Write a validation that always fails
    (proj / "validation" / "article.py").write_text(
        "def validate(prompt: str, result: str) -> bool:\n    return False\n"
    )
    result = run_pbt("run", cwd=proj, check=False)
    assert "errored" in result.stdout or result.returncode != 0


def test_run_downstream_skipped_on_error(tmp_path: Path) -> None:
    proj = init_project(tmp_path)
    # Make article always fail validation — summary depends on article
    (proj / "validation" / "article.py").write_text(
        "def validate(prompt: str, result: str) -> bool:\n    return False\n"
    )
    result = run_pbt("run", cwd=proj, check=False)
    assert "skipped" in result.stdout
