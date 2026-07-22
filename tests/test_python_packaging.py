from pathlib import Path
import tomllib


def test_poetry_wheel_declares_every_top_level_source_package():
    repository = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        package["include"]
        for package in configuration["tool"]["poetry"]["packages"]
        if package.get("from") == "src"
    }
    source_packages = {
        child.name
        for child in (repository / "src").iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    }

    assert declared == source_packages
