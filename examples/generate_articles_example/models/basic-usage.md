# Models

Write your prompts here. Each `.prompt` file defines one step in your pipeline.

You can:
- Reference other prompt outputs:  `{{ ref('other_prompt_name') }}`
- Access passed-in data:           `{{ promptdata("key") }}`
- Include files/data:              `{{ file("path/to/file.txt") }}`
- Configure output structure in the prompt (e.g. ask for JSON)

Example chain — `models/topic.prompt` → `models/article.prompt` → `models/summary.prompt`:

    # topic.prompt
    Generate a catchy blog post topic about AI.

    # article.prompt
    Write a detailed article about: {{ ref('topic') }}

    # summary.prompt
    Summarise this article in 3 bullet points:
    {{ ref('article') }}

Run with: `pbt run` or `pbt run --promptdata topic="your topic"`
