"""
Tests for transjax.analyzer.parser.fortran_parser.

Uses a minimal synthetic Fortran snippet — no real Fortran codebase required.
"""

import textwrap
import tempfile
from pathlib import Path

import pytest

from transjax.analyzer.parser.fortran_parser import FortranParser, ModuleInfo
from transjax.analyzer.config.project_config import FortranProjectConfig


MINIMAL_FORTRAN = textwrap.dedent("""\
    module simple_math
      implicit none

      contains

      subroutine add(a, b, result)
        real, intent(in)  :: a, b
        real, intent(out) :: result
        result = a + b
      end subroutine add

      function square(x) result(y)
        real, intent(in) :: x
        real :: y
        y = x * x
      end function square

    end module simple_math
""")


@pytest.fixture
def fortran_tmp_dir():
    """Create a temporary directory with a minimal Fortran file."""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "simple_math.f90"
        f.write_text(MINIMAL_FORTRAN)
        yield tmp


@pytest.fixture
def parser(fortran_tmp_dir):
    config = FortranProjectConfig(
        project_name="test_project",
        project_root=fortran_tmp_dir,
        source_dirs=[fortran_tmp_dir],
        generate_graphs=False,
    )
    return FortranParser(config)


def test_find_fortran_files(parser, fortran_tmp_dir):
    """Parser must discover the .f90 file."""
    files = parser.find_fortran_files()
    assert len(files) == 1
    assert files[0].suffix.lower() == ".f90"


def test_parse_file_returns_module_info(parser, fortran_tmp_dir):
    """Parsing the file must return a ModuleInfo with the correct module name."""
    files = parser.find_fortran_files()
    module_info = parser.parse_file(files[0])

    assert module_info is not None
    assert isinstance(module_info, ModuleInfo)
    assert module_info.name.lower() == "simple_math"


def test_parse_detects_subroutine(parser, fortran_tmp_dir):
    """Parser must detect at least the 'add' subroutine."""
    files = parser.find_fortran_files()
    module_info = parser.parse_file(files[0])

    assert module_info is not None
    subroutine_names = [s.lower() for s in module_info.subroutines]
    assert "add" in subroutine_names


def test_parse_detects_function(parser, fortran_tmp_dir):
    """Parser must detect the 'square' function."""
    files = parser.find_fortran_files()
    module_info = parser.parse_file(files[0])

    assert module_info is not None
    function_names = [f.lower() for f in module_info.functions]
    assert "square" in function_names


def test_parse_project(parser):
    """parse_project must return a dict with 'modules' key."""
    results = parser.parse_project()
    assert "modules" in results
    assert len(results["modules"]) > 0


def test_module_line_count_positive(parser):
    """Each parsed module must have a positive line count."""
    results = parser.parse_project()
    for module_name, module_info in results["modules"].items():
        assert module_info.line_count > 0, (
            f"Module {module_name} has zero line count"
        )
