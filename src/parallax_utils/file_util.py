import os
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    configured_root = os.environ.get("PARALLAX_SRC", "").strip()
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError(
                f"PARALLAX_SRC does not point to a Parallax source tree: {root}"
            )
        return root

    # Search for the project root by looking for pyproject.toml in parent directories
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent

    # If not found, fallback to current working directory
    return Path.cwd()
