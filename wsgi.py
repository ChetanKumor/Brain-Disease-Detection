"""WSGI entry point.

Production (behind a real server)::

    gunicorn "wsgi:app" --bind 0.0.0.0:8000 --workers 2

Local development::

    python wsgi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Load variables from a local .env file if python-dotenv is available.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional convenience dependency
    pass

from brain_disease_detection.config import load_config  # noqa: E402
from brain_disease_detection.web import create_app  # noqa: E402

_config = load_config()
app = create_app(_config)


if __name__ == "__main__":
    app.run(
        host=_config.app.host,
        port=_config.app.port,
        debug=not _config.app.is_production,
    )
