"""Tests for path utilities."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from doomgen.utils.paths import (
    get_project_root,
    get_bundled_compiler,
    validate_executable,
)


class TestGetProjectRoot(unittest.TestCase):
    """Tests for get_project_root function."""

    def test_get_project_root_returns_path(self):
        """Test that get_project_root returns a Path object."""
        result = get_project_root()
        self.assertIsInstance(result, Path)

    def test_get_project_root_contains_src_directory(self):
        """Test that the project root contains a src directory."""
        root = get_project_root()
        self.assertTrue((root / "src").is_dir())

    def test_get_project_root_contains_pyproject_toml(self):
        """Test that the project root contains pyproject.toml."""
        root = get_project_root()
        self.assertTrue((root / "pyproject.toml").is_file())

    def test_get_project_root_caching(self):
        """Test that get_project_root caches results."""
        # Call it twice and verify the cache is used
        result1 = get_project_root()
        result2 = get_project_root()
        self.assertIs(result1, result2)  # Should be the exact same object (cached)


class TestGetBundledCompiler(unittest.TestCase):
    """Tests for get_bundled_compiler function."""

    def test_get_bundled_compiler_acc_unix(self):
        """Test that get_bundled_compiler returns correct path for acc on Unix."""
        with patch("sys.platform", "linux"):
            path = get_bundled_compiler("acc")
            self.assertTrue(path.endswith("acc/acc") or path.endswith("acc\\acc"))
            self.assertNotIn(".exe", path)

    def test_get_bundled_compiler_acc_windows(self):
        """Test that get_bundled_compiler returns correct path for acc on Windows."""
        with patch("sys.platform", "win32"):
            path = get_bundled_compiler("acc")
            self.assertTrue(path.endswith("acc/acc.exe") or path.endswith("acc\\acc.exe"))

    def test_get_bundled_compiler_zdbsp_unix(self):
        """Test that get_bundled_compiler returns correct path for zdbsp on Unix."""
        with patch("sys.platform", "darwin"):
            path = get_bundled_compiler("zdbsp")
            self.assertTrue(path.endswith("zdbsp/zdbsp") or path.endswith("zdbsp\\zdbsp"))
            self.assertNotIn(".exe", path)

    def test_get_bundled_compiler_zdbsp_windows(self):
        """Test that get_bundled_compiler returns correct path for zdbsp on Windows."""
        with patch("sys.platform", "win32"):
            path = get_bundled_compiler("zdbsp")
            self.assertTrue(path.endswith("zdbsp/zdbsp.exe") or path.endswith("zdbsp\\zdbsp.exe"))

    def test_get_bundled_compiler_returns_string(self):
        """Test that get_bundled_compiler returns a string."""
        result = get_bundled_compiler("acc")
        self.assertIsInstance(result, str)

    def test_get_bundled_compiler_returns_absolute_path(self):
        """Test that get_bundled_compiler returns an absolute path."""
        result = get_bundled_compiler("acc")
        path_obj = Path(result)
        self.assertTrue(path_obj.is_absolute())

    def test_get_bundled_compiler_invalid_tool(self):
        """Test that get_bundled_compiler raises ValueError for unknown tool."""
        with self.assertRaises(ValueError):
            get_bundled_compiler("invalid_tool")

    def test_get_bundled_compiler_invalid_tool_message(self):
        """Test the error message for unknown tool."""
        with self.assertRaises(ValueError) as cm:
            get_bundled_compiler("invalid_tool")
        self.assertIn("Unknown tool", str(cm.exception))


class TestValidateExecutable(unittest.TestCase):
    """Tests for validate_executable function."""

    def test_validate_executable_with_existing_file(self):
        """Test validate_executable returns True for existing files."""
        # Use a file we know exists
        python_path = Path(__file__).resolve()
        self.assertTrue(validate_executable(str(python_path)))

    def test_validate_executable_with_nonexistent_file(self):
        """Test validate_executable returns False for nonexistent files."""
        self.assertFalse(validate_executable("/nonexistent/path/to/executable"))

    def test_validate_executable_uses_shutil_which(self):
        """Test that validate_executable uses shutil.which()."""
        with patch("doomgen.utils.paths.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/python"
            result = validate_executable("python")
            self.assertTrue(result)
            mock_which.assert_called_once_with("python")

    def test_validate_executable_falls_back_to_path_exists(self):
        """Test that validate_executable falls back to Path.exists() when shutil.which fails."""
        with patch("doomgen.utils.paths.shutil.which") as mock_which:
            mock_which.return_value = None
            # Use a valid path that we know exists
            python_path = Path(__file__).resolve()
            result = validate_executable(str(python_path))
            self.assertTrue(result)

    def test_validate_executable_returns_boolean(self):
        """Test that validate_executable always returns a boolean."""
        result = validate_executable("/some/path")
        self.assertIsInstance(result, bool)

    @patch("doomgen.utils.paths.shutil.which")
    def test_validate_executable_shutil_which_called_first(self, mock_which):
        """Test that shutil.which is called before checking file existence."""
        mock_which.return_value = "/usr/bin/test"
        with patch("doomgen.utils.paths.Path.exists") as mock_exists:
            validate_executable("test_tool")
            mock_which.assert_called_once()
            # Path.exists should not be called if shutil.which succeeds
            mock_exists.assert_not_called()


if __name__ == "__main__":
    unittest.main()
