from __future__ import annotations

from types import ModuleType
from pyodide.ffi import create_proxy
from js import document
import json
import sys
import traceback

status_el = document.getElementById("status")
output_el = document.getElementById("output")
button_el = document.getElementById("run-model")
model_source_el = document.getElementById("model-source")
button_proxy = None


def set_status(message: str, ok: bool | None = None) -> None:
    status_el.textContent = message
    if ok is True:
        status_el.className = "status ok"
    elif ok is False:
        status_el.className = "status fail"
    else:
        status_el.className = "status"


def install_browser_db_shim() -> None:
    state = {
        "dags": {},
        "runs": {},
        "results": {},
        "cache": {},
        "counter": 0,
    }

    def init_db() -> None:
        return None

    def load_dag(dag_hash: str) -> str | None:
        return state["dags"].get(dag_hash)

    def save_dag(dag_hash: str, dag_json: str) -> None:
        state["dags"][dag_hash] = dag_json

    def create_run(model_count: int, dag_hash: str, git_sha: str | None = None) -> str:
        state["counter"] += 1
        run_id = f"browser-run-{state['counter']}"
        state["runs"][run_id] = {
            "model_count": model_count,
            "dag_hash": dag_hash,
            "git_sha": git_sha,
            "status": "running",
        }
        state["results"][run_id] = {}
        return run_id

    def finish_run(run_id: str, status: str) -> None:
        state["runs"][run_id]["status"] = status

    def upsert_model_pending(
        run_id: str,
        model_name: str,
        prompt_template: str,
        depends_on: list[str],
    ) -> None:
        state["results"][run_id][model_name] = {
            "status": "pending",
            "prompt_template": prompt_template,
            "depends_on": depends_on,
        }

    def mark_model_running(run_id: str, model_name: str) -> None:
        state["results"][run_id][model_name]["status"] = "running"

    def get_cached_llm_output(cache_key: str) -> str | None:
        return state["cache"].get(cache_key)

    def mark_model_success(
        run_id: str,
        model_name: str,
        prompt_rendered: str,
        llm_output: str,
        cache_key: str | None = None,
    ) -> None:
        state["results"][run_id][model_name].update(
            {
                "status": "success",
                "prompt_rendered": prompt_rendered,
                "llm_output": llm_output,
            }
        )
        if cache_key is not None:
            state["cache"][cache_key] = llm_output

    def mark_model_error(run_id: str, model_name: str, error: str) -> None:
        state["results"][run_id][model_name].update(
            {"status": "error", "error": error}
        )

    def mark_model_skipped(run_id: str, model_name: str) -> None:
        state["results"][run_id][model_name]["status"] = "skipped"

    db = ModuleType("pbt.db")
    db.init_db = init_db
    db.load_dag = load_dag
    db.save_dag = save_dag
    db.create_run = create_run
    db.finish_run = finish_run
    db.upsert_model_pending = upsert_model_pending
    db.mark_model_running = mark_model_running
    db.get_cached_llm_output = get_cached_llm_output
    db.mark_model_success = mark_model_success
    db.mark_model_error = mark_model_error
    db.mark_model_skipped = mark_model_skipped
    sys.modules["pbt.db"] = db


def llm_stub(prompt: str, **kwargs) -> str:
    return f"UNUSED::{prompt}"


def run_model(event=None) -> None:
    import pbt

    button_el.disabled = True
    set_status("Running inline model through pbt...", None)

    try:
        model_source = model_source_el.value
        models = {
            "modelinclude": model_source,
            "browser_result": (
                "{% set value = ref('modelinclude') %}\n"
                "{{ skip_and_set_to_value(value if value else '<empty string>') }}\n"
            ),
        }
        results = pbt.run(
            models_from_dict=models,
            llm_call=llm_stub,
            verbose=False,
        )
        output_el.textContent = json.dumps(
            {
                "submitted_model": "modelinclude",
                "model_source": model_source,
                "results": results,
                "note": "browser_result mirrors modelinclude, but shows '<empty string>' when the output is blank.",
            },
            indent=2,
            sort_keys=True,
        )
        set_status("pbt.run completed successfully.", True)
    except Exception:
        output_el.textContent = traceback.format_exc()
        set_status("pbt.run failed.", False)
    finally:
        button_el.disabled = False


try:
    install_browser_db_shim()
    import pbt

    button_proxy = create_proxy(run_model)
    button_el.addEventListener("click", button_proxy)
    button_el.disabled = False
    output_el.textContent = json.dumps(
        {
            "pbt_version": pbt.__version__,
            "ready": True,
            "message": "PyScript loaded local pbt files. Click the button to run models_from_dict.",
        },
        indent=2,
    )
    set_status("Local pbt package loaded successfully.", True)
except Exception:
    output_el.textContent = traceback.format_exc()
    set_status("Failed to load local pbt package.", False)
