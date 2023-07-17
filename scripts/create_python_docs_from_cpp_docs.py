#!/usr/bin/env python3

import re


FILES_TO_PARSE = [
    {"input_filepath": "docs/api_cpp/doomGame.md", "output_filepath": "docs/api_python/doomGame.md", "submodule": "DoomGame.", 
     "append_to_header": """
```{eval-rst}
.. autoclass:: vizdoom.DoomGame
```
"""},
    {"input_filepath": "docs/api_cpp/utils.md", "output_filepath": "docs/api_python/utils.md", "submodule": ""},
]
SECTION_REGEX = r"^##+ *([a-zA-Z ]+) *$"
FUNCTION_REGEX = r"^###+ *`([a-zA-Z]+)` *$"


if __name__ == "__main__":
    for fp in FILES_TO_PARSE:

        with open(fp["input_filepath"]) as input_file:
            lines = input_file.readlines()

        with open(fp["output_filepath"], "w") as output_file:
            start_lines = ""
            started = False
            for line in lines:
                # If lines match pattern, extract the function name and arguments
                match = re.match(SECTION_REGEX, line)
                if match:
                    if started:
                        output_file.write("```\n\n")
                    else:
                        started = True
                        output_file.write(start_lines)
                        if "append_to_header" in fp:
                            output_file.write(fp["append_to_header"])
                        output_file.write("\n\n")
                    section_name = match.group(1)
                    output_file.write(f"## {section_name}\n\n")
                    output_file.write("```{eval-rst}\n")

                elif not started:
                    start_lines += line
                
                else:
                    match = re.match(FUNCTION_REGEX, line)
                    if match:
                        function_name = match.group(1)
                        function_name = function_name.replace("ViZDoom", "Vizdoom")
                        function_name = re.sub(r'(?<!^)(?=[A-Z])', '_', function_name).lower()  # Convert CamelCase to snake_case
                        output_file.write(f".. autofunction:: vizdoom.{fp['submodule']}{function_name}\n")


