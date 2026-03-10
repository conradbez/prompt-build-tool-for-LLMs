# Tests

Used when you change prompts to make sure everything still works and passes quality standards.
Write prompts here that will run against your models to assess their quality.

Each `.prompt` file in this directory is an LLM-as-judge test.
The prompt should reference model outputs via `{{ ref('model_name') }}` and return JSON:
  - Pass: `{"results": "pass"}`
  - Fail: `{"results": "fail"}`

Example test file `tests/summary_has_bullets.prompt`:

    Does the following text contain at least 3 bullet points (lines starting with - or •)?

    {{ ref('summary') }}

    Reply with only valid JSON: {"results": "pass"} or {"results": "fail"}.
