"""
Verify that every name advertised in transjax.__all__ is importable,
and that the package version is consistent with pyproject.toml.
"""

import importlib
import transjax


def test_all_names_importable():
    """Every name in __all__ must be importable from the top-level package."""
    for name in transjax.__all__:
        obj = getattr(transjax, name, None)
        assert obj is not None, (
            f"'{name}' is listed in transjax.__all__ but not found on the module"
        )


def test_version_is_string():
    """__version__ must be a non-empty string."""
    assert isinstance(transjax.__version__, str)
    assert len(transjax.__version__) > 0


def test_version_matches_pyproject():
    """Version in __init__.py must match pyproject.toml."""
    import tomllib  # Python 3.11+; falls back to tomli below
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        return  # skip if running outside the repo

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except ImportError:
        try:
            import tomli  # type: ignore
            with open(pyproject_path, "rb") as f:
                data = tomli.load(f)
        except ImportError:
            return  # skip if no toml parser available

    pyproject_version = data["project"]["version"]
    assert transjax.__version__ == pyproject_version, (
        f"transjax.__version__ ({transjax.__version__!r}) != "
        f"pyproject.toml version ({pyproject_version!r})"
    )
