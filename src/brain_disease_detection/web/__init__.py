"""The Flask web application (server-rendered UI + JSON API)."""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
