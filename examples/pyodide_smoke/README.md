# Pyodide `pbt.run()` example

This is a minimal PyScript page that:

- loads the local `pbt` package into the browser
- patches the `pbt.db` layer with an in-memory shim so SQLite is not required
- sends an inline model to `pbt.run(models_from_dict=...)`
- returns the run results in the page

The default model source is:

```jinja
{{ skip_and_set_to_value("") }}
```

That means the model executes through `pbt` without requiring any LLM call.

The example also loads the core Python dependencies used by `pbt`:

- `jinja2`
- `networkx`
- `pydantic`
- `rich`
- `click`
- `python-dotenv`

It does not try to run the full CLI or server stack. It is a browser-safe `pbt.run()` example around `models_from_dict`.

## Run it

From the repo root:

```bash
python -m http.server 8000
```

Then open:

`http://localhost:8000/examples/pyodide_smoke/`

## What success looks like

The page should show:

- a green success message
- a textarea containing the inline `modelinclude` source
- a working `Run in pbt` button
- JSON output from `pbt.run(models_from_dict={"modelinclude": ...})`

## What this is for

Use this as the first interactive browser check for `pbt`. If this page fails, the failure is in local package loading, PyScript configuration, or the browser-safe shim around `pbt` persistence.
