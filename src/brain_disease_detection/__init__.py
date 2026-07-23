"""Brain Disease Detection.

A deep-learning system that classifies brain MRI / CT scans across three
diagnostic tasks — Alzheimer's disease, brain stroke, and brain tumor — and
serves predictions through a Flask web application and JSON API.

The package is organised into cohesive sub-modules:

* :mod:`brain_disease_detection.config`      — typed, YAML-driven configuration
* :mod:`brain_disease_detection.data`        — image preprocessing utilities
* :mod:`brain_disease_detection.models`      — model architecture and registry
* :mod:`brain_disease_detection.training`    — the training pipeline
* :mod:`brain_disease_detection.inference`   — the prediction service
* :mod:`brain_disease_detection.web`         — the Flask application
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
