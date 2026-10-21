"""Builds the image and checks the container starts (slow: ~5 minutes).

Skipped unless RUN_DOCKER_TESTS=1, because it needs Docker and downloads
the base image and Python packages.
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(os.getenv("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1 to run")
def test_container_starts_and_serves_estimates():
    result = subprocess.run([str(ROOT / "scripts" / "smoke_test_docker.sh")], capture_output=True, text=True,
                            timeout=1800)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "Smoke test passed." in result.stdout
