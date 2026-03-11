import os
from google import genai

# This function is automaticaaly picked up by `pbt` and used to run .prompt files
def llm_call(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents=prompt,
    ).text
