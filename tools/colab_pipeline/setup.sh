#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../.."
python3 -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ is required"'
python3 -m venv .venv-colab
.venv-colab/bin/python -m pip install 'google-colab-cli==0.8.0'
.venv-colab/bin/colab --help
printf '\nReady. Authenticate from your terminal with .venv-colab/bin/colab sessions\n'
printf 'Inspect the pilot with .venv-colab/bin/python -m tools.colab_pipeline plan\n'
