# Validation

Optional Python code that runs *before* an LLM prompt output is passed to the next prompt.
Use this to check basic quality gates and avoid wasting tokens on bad intermediate results.

Each `.py` file must expose a `validate(prompt: str, result: str) -> bool` function.
Returning `False` stops the pipeline and reports a validation error.

Example `validation/article.py`:

    def validate(prompt: str, result: str) -> bool:
        """Article must be at least 200 characters and contain a markdown header."""
        return len(result) >= 200 and "#" in result
