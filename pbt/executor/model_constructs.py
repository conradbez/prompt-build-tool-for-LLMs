"""
Model construct handlers — specialised execution strategies keyed by model_type.

Each construct is an async function with the signature::

    async def execute(model, model_outputs, model_files, storage_backend,
                      run_id, llm_call, rag_call, promptdata,
                      prompt_skipped_models) -> ModelRunResult

Constructs mutate *model_outputs* in-place and return a ModelRunResult.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import TYPE_CHECKING, Callable

from pbt.executor.parser import render_prompt

if TYPE_CHECKING:
    from pbt.executor.executor import ModelRunResult
    from pbt.executor.graph import PromptModel
    from pbt.storage.base import StorageBackend


async def execute_loop_model(
    model: "PromptModel",
    model_outputs: dict,
    model_files: list | None,
    storage_backend: "StorageBackend",
    run_id: str,
    llm_call: Callable,
    rag_call: Callable | None,
    promptdata: dict | None,
    prompt_skipped_models: set[str],
    parse_json_output: Callable,
) -> "ModelRunResult":
    """Execute a loop model: call the LLM once per item in an upstream list."""
    from pbt.executor.executor import ModelRunResult

    # Find the single upstream dep that returns a JSON list.
    list_deps = {
        dep: model_outputs[dep]
        for dep in model.depends_on
        if dep in model_outputs and isinstance(model_outputs[dep], list)
    }
    if not list_deps:
        raise ValueError(
            f"Loop model '{model.name}': no upstream dependency returns a JSON list. "
            "Ensure an upstream model has output_format='json' and returns a list."
        )
    if len(list_deps) > 1:
        raise ValueError(
            f"Loop model '{model.name}': multiple dependencies return lists: {list(list_deps)}. "
            "Add loop_over='model_name' to config() to disambiguate."
        )
    list_dep_name, list_items = next(iter(list_deps.items()))

    loop_results: list = []
    total_elapsed_ms = 0
    all_cached = True
    rendered_prompts: list[str] = []
    output_format = model.config.get("output_format", "text")

    for item in list_items:
        # Override the list dep so ref('dep') returns the current item.
        item_outputs = {**model_outputs, list_dep_name: item}
        item_rendered, item_skip_state = render_prompt(
            model.source, item_outputs,
            promptdata=promptdata, rag_call=rag_call,
            prompt_skipped_models=prompt_skipped_models,
        )
        rendered_prompts.append(item_rendered)

        item_cache_key = item_rendered + "\x00" + json.dumps(model.config, sort_keys=True)

        if item_skip_state.skip_value is not None:
            item_output = item_skip_state.skip_value
        elif (cached_item := storage_backend.get_cached_llm_output(item_cache_key)) is not None:
            item_output = cached_item
        else:
            t0 = time.monotonic()
            _sig = inspect.signature(llm_call).parameters
            _kwargs: dict = {}
            if model_files and "files" in _sig:
                _kwargs["files"] = model_files
            if "config" in _sig:
                _kwargs["config"] = model.config
            raw = llm_call(item_rendered, **_kwargs)
            if inspect.isawaitable(raw):
                item_output = await raw
            else:
                item_output = raw
            total_elapsed_ms += int((time.monotonic() - t0) * 1000)
            all_cached = False

        if item_skip_state.skip_value is None and output_format == "json":
            item_output = parse_json_output(item_output)

        loop_results.append(item_output)

    # Store combined list as the model output (always JSON).
    model_outputs[model.name] = loop_results
    llm_output = json.dumps(loop_results)
    combined_rendered = (
        f"[loop over {len(list_items)} items from '{list_dep_name}']\n"
        + "\n---\n".join(rendered_prompts)
    )
    combined_cache_key = combined_rendered + "\x00" + json.dumps(model.config, sort_keys=True)
    storage_backend.mark_model_success(
        run_id, model.name, combined_rendered, llm_output,
        cache_key=combined_cache_key,
    )
    return ModelRunResult(
        model_name=model.name,
        status="success",
        prompt_rendered=combined_rendered,
        llm_output=llm_output,
        execution_ms=total_elapsed_ms,
        cached=all_cached,
        prompt_skipped=False,
    )
