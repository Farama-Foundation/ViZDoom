#!/usr/bin/env python3
"""
Automated ViZDoom Type Stub Generator

This script uses pybind11-stubgen to generate stub file for ViZDoom, part of build target `generate_stubs`.

Workflow:
- using `pybind11-stubgen` for generation
- removing the `__all__ = ...` line
- adding `import numpy as np` after `import typing`
- optionally, `black` and `isort` for simple formatting
- adding a header comment
- replacing the `-> typing.Any` properties with actual return behaviour (`np.ndarray` or `typing.Optional[np.ndarray]`)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


vizdoom_stub_header = '''"""
ViZDoom Python Type Stubs

This file provides type information for static analysis and IDE support.
Auto-generated via pybind11-stubgen{formatted}.

For the official documentation, see: https://vizdoom.farama.org/
"""

'''


class ViZDoomStubGenerator:
    """Generator for ViZDoom type stubs with additional"""

    def __init__(
        self,
        output_file: str = "src/lib_python/vizdoom.pyi",
        verbose: bool = False,
        module: str = "vizdoom",
    ):
        self.output_file = output_file
        self.temp_dir = tempfile.mkdtemp()
        self.docstrings = {}
        self.enum_fixes = {}
        self.verbose = bool(verbose)
        self.module = module

    def generate_with_stubgen(self):
        """Generate basic type stubs using stubgen."""
        if self.verbose:
            print("Generating type stubs with stubgen...")

        # Add module directory to Python path if provided
        env: dict[str, str] = os.environ.copy()
        if os.path.isdir(self.module):
            print("module", self.module)
            module_path = self.module
            if self.verbose:
                print(f"Adding {module_path} to PYTHONPATH")
            current_path = env.get("PYTHONPATH", "")
            pathsep = os.pathsep
            paths = current_path.split(pathsep) if current_path else []
            if module_path not in paths:
                paths.insert(0, module_path)
                env["PYTHONPATH"] = pathsep.join(paths)

        # Run stubgen on the compiled vizdoom module
        cmd = ["pybind11-stubgen", "vizdoom", "-o", self.temp_dir]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            if not os.path.exists(os.path.join(self.temp_dir, "vizdoom.pyi")):
                self.temp_dir = os.path.join(self.temp_dir, "vizdoom")
            stub_file = os.path.join(self.temp_dir, "vizdoom.pyi")
            if os.path.exists(stub_file):
                with open(stub_file, "r", encoding="utf-8", errors="ignore") as f:
                    stub_content = []
                    for line in f:
                        if line.startswith("import typing"):
                            stub_content += (
                                "import typing\nfrom numpy.typing import NDArray\n"
                            )
                        elif not line.startswith("__all__ = "):
                            stub_content += line
                with open(stub_file, "w", encoding="utf-8", errors="ignore") as f:
                    f.writelines(stub_content)
            else:
                raise FileNotFoundError(
                    f"Stubgen did not create expected file: {stub_file}"
                )

        except subprocess.CalledProcessError as e:
            print(f"Stubgen failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise

    def reformat_generated_stub(self):
        """Reformat generated type stubs with black and isort."""
        finished_formatting: list[str] = []
        try:
            for tool in ["black", "isort"]:
                if self.verbose:
                    print(f"Reformatting type stubs with {tool}...")
                cmd = [tool, os.path.join(self.temp_dir, "vizdoom.pyi")]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                finished_formatting.append(tool)
        except subprocess.CalledProcessError as e:
            print(f"{tool} failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
        return finished_formatting

    def load_generated_stub(self) -> str:
        """Reformat generated type stubs with black."""
        if self.verbose:
            print("Loading generated type stubs...")
        stub_file = os.path.join(self.temp_dir, "vizdoom.pyi")
        if os.path.exists(stub_file):
            with open(stub_file, "r") as f:
                content = f.read().removeprefix('"""\nViZDoom Python module.\n"""\n')
            # Remove exact value of __version__ from the stub to avoid misguiding the user
            content = re.sub(r"^(__version__: str).*$", r"\1", content, count=1, flags=re.MULTILINE)
            return content
        else:
            raise FileNotFoundError(
                f"Stubgen did not create expected file: {stub_file}"
            )

    def add_module_header(self, content: str, formatted_with: list[str]) -> str:
        """Add a module header."""
        if self.verbose:
            print("Adding module header...")
        if formatted_with:
            formatted = f" and formatted with {', '.join(formatted_with)}"
        else:
            formatted = ""
        return vizdoom_stub_header.format(formatted=formatted) + content

    def annotate_gamestate_properties(self, content: str) -> str:
        """Additional treatment for properties of GameState."""
        if self.verbose:
            print("Annotating properties of the GameState class...", end=" ")
        replacements = []
        match_game_state_properties = re.finditer(
            r"^(\s*def (screen|depth|audio|automap|labels|game)_"
            r"(?:buffer|variables)\(self\)\s*->\s*)(typing\.)?Any\s*(:.*)$",
            content,
            flags=re.MULTILINE,
        )
        for match_property in match_game_state_properties:
            if match_property.group(2) == "screen":
                return_type = "NDArray"
            else:
                return_type = match_property.group(3) + "Optional[NDArray]"
            actual_return_type = (
                match_property.group(1) + return_type + match_property.group(4)
            )
            replacements.append((match_property.group(0), actual_return_type))

        for old_line, new_line in replacements:
            content = content.replace(old_line, new_line, 1)

        if self.verbose:
            print(f"(patched {len(replacements)} lines)")

        return content

    def generate(self) -> str:
        """Generate complete type stubs with docstrings."""
        if self.verbose:
            print("Starting ViZDoom stub generation...")

        try:
            # Step 1: Generate basic stubs with stubgen
            self.generate_with_stubgen()

            # Step 2: Reformat stub file with black & isort
            formatted_with = self.reformat_generated_stub()

            # Step 3: Load stub filex
            stub_content = self.load_generated_stub()

            # Step 4: Add module header
            stub_content = self.add_module_header(stub_content, formatted_with)

            # Step 5: Additionally treatment for properties of GameState object
            stub_content = self.annotate_gamestate_properties(stub_content)

            # Step 6: Write to output file
            output_dir = os.path.dirname(self.output_file)
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError:
                pass  # maybe the folder is in use or something OS-dependent
            with open(self.output_file, "w") as f:
                f.write(stub_content)

            if self.verbose:
                print(
                    f"Generated {self.output_file} ({len(stub_content.splitlines())} lines)"
                )

            return stub_content

        except Exception as e:
            print(f"Stub generation failed: {e}")
            raise
        finally:
            # Cleanup temp directory
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate ViZDoom type stubs")
    parser.add_argument(
        "-o",
        "--output",
        default="src/lib_python/vizdoom.pyi",
        help="Output file path (default: src/lib_python/vizdoom.pyi)",
    )
    parser.add_argument(
        "-m",
        "--module",
        default="vizdoom",
        help="Directory containing vizdoom module (default: vizdoom)",
    )
    parser.add_argument(
        "-p",
        "--patch",
        action="store_true",
        help="Patch ViZDoom library in current environment",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    generator = ViZDoomStubGenerator(args.output, args.verbose, args.module)
    try:
        import pybind11_stubgen

        assert pybind11_stubgen
        generator.generate()
        print("Successfully generated ViZDoom type stubs!")
        print(f"Output: {args.output}")
        if args.patch:
            import vizdoom

            vizdoom_loc = os.path.dirname(vizdoom.__file__)
            shutil.copyfile(
                args.output,
                os.path.join(vizdoom_loc, "vizdoom.pyi"),
                follow_symlinks=False,
            )
            with open(os.path.join(vizdoom_loc, "py.typed"), "w") as f:
                f.write("partial\n")
            print(f"Patched ViZDoom at: {vizdoom_loc}")
            print(f"Enjoy your type-hinted ViZDoom!")
        return 0
    except ImportError:
        print(
            f"Need to install pybind11-stubgen via:\npip install pybind11-stubgen\n"
        )
        print(
            "Optionally also install black and isort for formatting:\npip install black isort\n"
        )
    except Exception as e:
        print(f"Failed to generate stubs: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
