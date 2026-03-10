"""
FastAPI application factory for the pbt server.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, List, Optional

try:
    from fastapi import FastAPI, File, Form, Query, UploadFile
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "utils.server requires FastAPI and uvicorn. "
        "Install them with: pip install fastapi uvicorn"
    ) from exc

import pbt


class RunResponse(BaseModel):
    outputs: dict[str, Any]
    errors: list[str] = []


def _serialise(outputs: dict) -> tuple[dict[str, Any], list[str]]:
    serialised: dict[str, Any] = {}
    errors: list[str] = []
    for name, value in outputs.items():
        if isinstance(value, pbt.ModelStatus):
            serialised[name] = value.value
            errors.append(f"{name}: {value.value}")
        else:
            serialised[name] = value
    return serialised, errors


def _parse_promptdata(promptdata_json: str | None) -> dict | None:
    """Parse the JSON-encoded promptdata form field, or return None."""
    if not promptdata_json:
        return None
    try:
        parsed = json.loads(promptdata_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"promptdata must be a valid JSON object, got: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("promptdata must be a JSON object (dict), not a list or scalar.")
    return parsed or None


def _parse_promptfiles(uploads: list[UploadFile] | None) -> dict | None:
    """
    Convert a list of uploaded files into the ``promptfiles`` dict expected by
    ``pbt.run()``.  Each file's ``filename`` (without extension) is used as the
    promptfile key, so the caller should name files to match the keys declared
    in the model's ``# pbt:config promptfiles:`` block.

    Example: upload a file named ``doc.pdf`` → key ``"doc"``.
    If the filename has no extension the full filename is used as the key.
    """
    if not uploads:
        return None
    result: dict = {}
    for upload in uploads:
        raw_name = upload.filename or ""
        # Strip extension to get the key (e.g. "doc.pdf" → "doc")
        key = raw_name.rsplit(".", 1)[0] if "." in raw_name else raw_name
        if not key:
            key = raw_name
        result[key] = upload.file  # SpooledTemporaryFile satisfies IO[bytes]
    return result or None


def _build_run_endpoint(
    models_dir: str,
    validation_dir: str,
    dag_promptdata: list[str],
    dag_promptfiles: list[str],
):
    """
    Dynamically build a /run function whose signature lists each detected
    promptdata() key as an optional query parameter. FastAPI reads __signature__
    to generate the OpenAPI schema, so every key shows up in /docs.

    promptfiles cannot be passed via GET (no file uploads), but they are
    documented in the endpoint description.
    """

    def _run(**kwargs: Any) -> RunResponse:
        provided = {k: v for k, v in kwargs.items() if v is not None}
        try:
            outputs = pbt.run(
                models_dir=models_dir,
                promptdata=provided or None,
                validation_dir=validation_dir,
                verbose=False,
            )
        except Exception as exc:
            return RunResponse(outputs={}, errors=[str(exc)])
        serialised, errors = _serialise(outputs)
        return RunResponse(outputs=serialised, errors=errors)

    # Build a signature: one Optional[str] query param per detected promptdata key
    params = [
        inspect.Parameter(
            key_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=Query(None, description=f"Template variable: `{{{{ promptdata('{key_name}') }}}}`"),
            annotation=Optional[str],
        )
        for key_name in dag_promptdata
    ]
    _run.__signature__ = inspect.Signature(params)

    promptfiles_note = (
        "\n\n**Required promptfiles** (upload via `POST /run`):\n"
        + "\n".join(f"- `{v}`" for v in dag_promptfiles)
        if dag_promptfiles
        else ""
    )

    description = (
        "Run all pbt prompt models and return their outputs.\n\n"
        + (
            "**Detected template variables** (from `promptdata()` usage in `.prompt` files):\n"
            + "\n".join(f"- `{v}`" for v in dag_promptdata)
            if dag_promptdata
            else "_No promptdata() variables detected in current models._"
        )
        + promptfiles_note
        + (
            "\n\n> **Note:** File uploads (`promptfiles`) are only supported via `POST /run`."
            if dag_promptfiles
            else ""
        )
    )
    _run.__doc__ = description

    return _run


def create_app(
    models_dir: str = "models",
    validation_dir: str = "validation",
) -> FastAPI:
    """
    Create and return a FastAPI app that exposes pbt over HTTP.

    The /run endpoint's query parameters are built dynamically from the vars
    detected across all .prompt files via static promptdata() scanning at startup.
    """
    # Detect promptdata() keys and promptfile names at startup so the OpenAPI
    # schema is accurate.
    try:
        from pbt.executor.graph import load_models, get_dag_promptdata, get_dag_promptfiles
        models = load_models(models_dir)
        dag_promptdata = get_dag_promptdata(models)
        dag_promptfiles = get_dag_promptfiles(models)
    except Exception:
        dag_promptdata = []
        dag_promptfiles = []

    app = FastAPI(
        title="pbt server",
        description=(
            "Run pbt prompt models via HTTP. "
            "Query parameters on `/run` are auto-generated from `promptdata()` "
            "usage detected in your `.prompt` files. "
            "File inputs (`promptfiles`) are accepted as multipart uploads on `POST /run`."
        ),
        version=pbt.__version__,
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "pbt_version": pbt.__version__,
            "dag_promptdata": dag_promptdata,
            "dag_promptfiles": dag_promptfiles,
        }

    # POST /run — multipart/form-data so that both text variables (promptdata)
    # and file uploads (promptfiles) can be included in the same request.
    #
    # promptdata  : JSON-encoded dict, e.g. '{"country": "USA"}'
    # select      : repeated form field, e.g. select=tweet&select=haiku
    # <any file>  : uploaded file; filename (without extension) is the promptfile key
    @app.post("/run", response_model=RunResponse, summary="Run models (multipart form)")
    async def run_post(
        promptdata: Optional[str] = Form(
            None,
            description=(
                "JSON-encoded dict of template variables, e.g. "
                '`{"country": "USA", "tone": "formal"}`. '
                "Matches `promptdata()` calls in `.prompt` templates."
            ),
        ),
        select: Optional[List[str]] = Form(
            None,
            description="Limit execution to these model names (and their upstream dependencies).",
        ),
        promptfiles: Optional[List[UploadFile]] = File(
            None,
            description=(
                "Files required by models that declare `promptfiles` in their config block. "
                "Each file's **filename** (without extension) is used as the promptfile key, "
                "so name your upload to match the key declared in the model "
                "(e.g. upload `doc.pdf` for `promptfiles: doc`)."
                + (
                    f" Detected keys in current models: {', '.join(f'`{v}`' for v in dag_promptfiles)}."
                    if dag_promptfiles
                    else ""
                )
            ),
        ),
    ) -> RunResponse:
        """
        Run pbt models. Accepts multipart/form-data so file inputs (promptfiles)
        can be uploaded alongside text variables (promptdata).

        **promptdata** — JSON object string with template variables.\n
        **select** — repeated field to limit which models run.\n
        **promptfiles** — one file per declared promptfile key; use the key as the filename
        (with any extension), e.g. `doc.pdf` → key `doc`.
        """
        try:
            pd = _parse_promptdata(promptdata)
        except ValueError as exc:
            return RunResponse(outputs={}, errors=[str(exc)])

        pf = _parse_promptfiles(promptfiles)

        try:
            outputs = pbt.run(
                models_dir=models_dir,
                select=select,
                promptdata=pd,
                promptfiles=pf,
                validation_dir=validation_dir,
                verbose=False,
            )
        except Exception as exc:
            return RunResponse(outputs={}, errors=[str(exc)])
        serialised, errors = _serialise(outputs)
        return RunResponse(outputs=serialised, errors=errors)

    # GET /run — dynamic query params per detected promptdata key (for the docs UI).
    # File uploads are not possible via GET; use POST /run for promptfiles.
    run_get = _build_run_endpoint(models_dir, validation_dir, dag_promptdata, dag_promptfiles)
    app.get("/run", response_model=RunResponse, summary="Run models (query params)")(run_get)

    return app
