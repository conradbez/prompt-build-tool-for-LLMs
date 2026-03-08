# pbt — Architecture & Developer Notes

---

## Module map

```
pbt/
  __init__.py   Python API (pbt.run) — also resolves llm/rag backends
  cli.py        Click commands — orchestrates discovery, calls execute_run
  graph.py      PromptModel dataclass, DAG building, topological sort
  parser.py     Jinja2 rendering, VarSpy dry-render, config block parsing
  executor.py   Pure execution loop — no file discovery, no CLI concerns
  llm.py / rag.py / validator.py  Backend resolvers
  db.py         SQLite schema + queries
  tester.py / docs.py  pbt test and pbt docs implementations
```

---

## Key design decisions

**`executor.py` is a pure executor.** It takes callables (`llm_call`, `rag_call`, `validators`) and never touches the filesystem. File discovery lives in `cli.py` and `pbt/__init__.py`. This keeps the executor testable with mock callables.

**DAG hash covers structure, not content.** `compute_dag_hash()` hashes model names and edges only — not prompt text. The hash is used to verify `--select` can safely reuse a previous run's outputs. Content changes don't invalidate it.

**Prompt cache is content-addressed.** SHA256 of the *rendered* prompt (post-Jinja, pre-LLM) is the cache key. Identical rendered prompts across any run reuse the stored output. Independent of `--select` which operates at the run level.

**`model_outputs` is `dict[str, str | dict | list]`.** When a model declares `output_format: json`, its entry is a parsed Python object, not a string. Downstream `ref('model')` in Jinja receives the object, enabling `{{ ref('model').key }}` access. The DB always stores canonical JSON strings.

---

## Static promptdata() detection

Before any run, pbt scans each template with a regex to discover which `promptdata()` keys it uses — no Jinja rendering needed.

```python
_PROMPTDATA_PATTERN = re.compile(r"""\bpromptdata\(\s*['"](\w+)['"]\s*\)""")

def detect_used_promptdata(template_source: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _PROMPTDATA_PATTERN.finditer(template_source):
        seen[match.group(1)] = None
    return list(seen)
```

Results stored in `PromptModel.promptdata_used`, shown in `pbt ls`, and warned about in `pbt run` if not provided. This is simpler and more reliable than the previous VarSpy dry-render approach — all branches of conditionals are detected since it's a static scan.

---

## Validation vs. tests

| | `tests/` | `validation/` |
|---|---|---|
| **Format** | `.prompt` Jinja files | `.py` Python files |
| **When** | Explicit `pbt test` after a run | Automatically inside `pbt run` |
| **Input** | Model outputs from a previous run | Rendered prompt + LLM output of the current model |
| **Pass criterion** | LLM returns `{"results": "pass"}` | `validate(prompt, result) -> bool` |
| **On failure** | Non-zero exit from `pbt test` | Model marked `error`, downstream skipped |

---

## SQLite notes

- DB at `.pbt/pbt.db` relative to cwd.
- `PRAGMA journal_mode=WAL` — allows concurrent readers during a run.
- `_migrate()` applies idempotent `ALTER TABLE ADD COLUMN` for backward compat.
- `prompt_hash` and `dag_hash` are indexed for the cache and `--select` lookups.
