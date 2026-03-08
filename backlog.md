1. in cli on run, should generate a outputs dir witch each models results save in [modelname].md
2. refactors `files` to be called `promptfiles` to align with `promptdata` naming
3.  QuickQuestion: fastapi to use `vars` found per model to specify query params in it's api guide - how hard would this be?
4. make sure recurse into subdirectories - like in dbt - model name is still just the filename (unique across all sub directories)