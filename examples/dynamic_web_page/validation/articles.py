# Validation files let you post-process and gate LLM outputs before they flow downstream.
# - Name this file after the prompt it validates (e.g. articles.py validates articles.prompt).
# - Return False (or raise) to fail the pipeline; return any value to replace the raw LLM text.
# - Returned values are stored and used by ref() in downstream prompts — parse, clean, or reshape freely.

# import json
# from pydantic import BaseModel, ValidationError
#
#
# class Article(BaseModel):
#     content: str
#     author: str
#     audience: str
#
#
# def validate(prompt: str, result: str) -> Article | bool:
#     """Validate and return the parsed Article dict, or False on failure."""
#     try:
#         data = json.loads(result)
#         article = Article(**data)
#     except (json.JSONDecodeError, ValidationError):
#         return False
#     if len(article.content) < 200:
#         return False
#     return article.model_dump()
