"""
Jinja2-based prompt parser with ref() support.

Special template function
-------------------------
{{ ref('model_name') }}
    Injects the LLM output of another model into this prompt.
    pbt uses calls to ref() to build the dependency graph before execution.

All standard Jinja2 features (loops, conditionals, filters, macros, etc.)
are available in addition to ref().
"""

import re
from typing import Callable

from jinja2 import Environment, StrictUndefined, Undefined


SKIP_SENTINEL = "SKIP THIS MODEL"
_SKIP_OUTPUT  = "SKIPPED THIS MODEL"

# Regex that finds every ref('...') or ref("...") call in raw template text.
# Used for static dependency extraction WITHOUT executing the template.
_REF_PATTERN = re.compile(r"""\bref\(\s*['"](\w+)['"]\s*\)""")

# Regex to extract pbt config block: {# pbt:config ... #} at the top of file.
_CONFIG_PATTERN = re.compile(
    r"^\s*\{#\s*pbt:config\s*(.*?)\s*#\}", re.DOTALL
)


class VarSpy(dict):
    """
    A dict that records every key accessed via [] or .get().
    Used during a dry render to discover which vars a template uses.
    Returns a truthy dummy string for every key so rendering doesn't abort.
    """

    def __init__(self) -> None:
        super().__init__()
        self._accessed: list[str] = []

    def __getitem__(self, key: str) -> str:
        self._accessed.append(key)
        return f"__var_{key}__"

    def __contains__(self, key: object) -> bool:
        return True  # vars.key always "exists" during dry render

    def get(self, key: str, default=None) -> str:  # type: ignore[override]
        self._accessed.append(key)
        return f"__var_{key}__"

    @property
    def accessed(self) -> list[str]:
        """Deduplicated list of accessed keys in first-seen order."""
        return list(dict.fromkeys(self._accessed))


def detect_used_vars(template_source: str) -> list[str]:
    """
    Dry-render *template_source* with a VarSpy and dummy upstream outputs
    to discover which ``vars.*`` keys the template accesses.

    Best-effort: if the template errors mid-render, keys accessed up to that
    point are still returned. Control-flow branches not taken (e.g. the
    ``{% else %}`` side of a conditional that depends on a var) are not
    traversed — this is an inherent limitation of dynamic templates.
    """
    from collections import defaultdict

    env = _make_env()
    spy = VarSpy()
    dummy_outputs: dict = defaultdict(lambda: "dummy_output")

    context = {
        "ref": lambda name: dummy_outputs[name],
        "return_list_RAG_results": lambda *_: ["dummy_rag_result"],
        "was_skipped": lambda _: False,
        "skip_this_model": SKIP_SENTINEL,
        "vars": spy,
    }

    try:
        env.from_string(template_source).render(**context)
    except Exception:
        pass  # partial render still captures vars accessed so far

    return spy.accessed


def parse_model_config(template_source: str) -> dict:
    """
    Parse an optional config block at the top of a .prompt file.

    Format::

        {# pbt:config
        output_format: json
        #}

    Returns a dict of config keys (strings) to values (strings).
    Returns an empty dict if no config block is found.
    """
    match = _CONFIG_PATTERN.search(template_source)
    if not match:
        return {}
    config: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            config[key.strip()] = value.strip()
    return config


def extract_dependencies(template_source: str) -> list[str]:
    """
    Return the list of model names referenced via ref() in *template_source*.

    This is a static scan — no Jinja rendering happens here.
    Duplicate references are deduplicated while preserving first-seen order.
    """
    seen: dict[str, None] = {}
    for match in _REF_PATTERN.finditer(template_source):
        seen[match.group(1)] = None
    return list(seen)


def render_prompt(
    template_source: str,
    model_outputs: dict[str, str],
    extra_vars: dict | None = None,
    rag_call: "Callable[..., list[str]] | None" = None,
) -> str:
    """
    Render *template_source* as a Jinja2 template.

    Parameters
    ----------
    template_source:
        Raw contents of a .prompt file.
    model_outputs:
        Mapping of model_name → LLM output text for all upstream models.
    extra_vars:
        Additional variables injected into the template context.

    The ``ref(model_name)`` function is available inside templates and
    returns the corresponding entry from *model_outputs*.
    """
    env = _make_env()

    def ref(model_name: str) -> str:
        if model_name not in model_outputs:
            raise ValueError(
                f"ref('{model_name}') — model '{model_name}' has no output yet. "
                "This is likely a missing dependency or execution-order bug."
            )
        return model_outputs[model_name]

    def return_list_RAG_results(*args) -> list[str]:
        if rag_call is None:
            raise RuntimeError(
                "return_list_RAG_results() called but no rag_call was provided to render_prompt."
            )
        return rag_call(*args)

    def was_skipped(model_name: str) -> bool:
        return model_outputs.get(model_name) == _SKIP_OUTPUT

    context: dict = {
        "ref": ref,
        "return_list_RAG_results": return_list_RAG_results,
        "was_skipped": was_skipped,
        "skip_this_model": SKIP_SENTINEL,
        "vars": extra_vars or {},
    }

    template = env.from_string(template_source)
    return template.render(**context)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_env() -> Environment:
    """Return a Jinja2 Environment configured for prompt files."""
    return Environment(
        # Keep newlines in templates so multi-paragraph prompts render cleanly
        keep_trailing_newline=True,
        # Raise on unknown variables so typos are caught early
        undefined=StrictUndefined,
        # Standard Jinja2 delimiters — easy to read in plain-text prompts
        block_start_string="{%",
        block_end_string="%}",
        variable_start_string="{{",
        variable_end_string="}}",
        comment_start_string="{#",
        comment_end_string="#}",
        # Strip leading whitespace on blocks to keep prompts tidy
        trim_blocks=True,
        lstrip_blocks=True,
    )
