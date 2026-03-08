def validate(prompt: str, result: str) -> bool:
    """Article must be at least 200 characters and contain a markdown header."""
    return len(result) >= 200 and "#" in result
