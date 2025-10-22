
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "harmonic-rsi"
version = "0.2.0"
description = "Harmonic RSI — resonance & stability metrics with ISM Φ-layer and Meta-Agent"
authors = [{name="Freedom (Damjan)"}, {name="Harmonic Logos"}]
license = {text = "Apache-2.0"}
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "gradio>=4.0.0",
    "numpy>=1.23",
    "pandas>=1.5",
    "requests>=2.31",
    "openai>=1.40",
]

[project.optional-dependencies]
st = ["sentence-transformers>=2.2.2"]
dev = ["pytest>=7.0", "pytest-cov>=4.0"]

[project.scripts]
harmonic-rsi-app = "harmonic_rsi.app_gradio:main"

[tool.setuptools]
packages = ["harmonic_rsi", "harmonic_rsi.agents"]
include-package-data = true
