from __future__ import annotations

try:
    from .gui_app import run_app
except ImportError:  # pragma: no cover - fallback for direct script execution
    from gui_app import run_app  # type: ignore


if __name__ == "__main__":
    run_app()
