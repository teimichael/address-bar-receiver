import shutil
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", default=False, help="run tests needing a live vLLM endpoint")
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch()
            b.close()
        return True
    except Exception:
        return False


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=15, check=True)
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs --run-gpu and a live vLLM endpoint")
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    browser_ok = None
    docker_ok = None
    for item in items:
        # markers, not keywords: item.keywords also holds node names (the tests/docker directory)
        if item.get_closest_marker("gpu") and not config.getoption("--run-gpu"):
            item.add_marker(skip_gpu)
        if item.get_closest_marker("slow") and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if item.get_closest_marker("browser"):
            if browser_ok is None:
                browser_ok = _chromium_available()
            if not browser_ok:
                item.add_marker(pytest.mark.skip(reason="Playwright Chromium unavailable"))
        if item.get_closest_marker("docker"):
            if docker_ok is None:
                docker_ok = _docker_available()
            if not docker_ok:
                item.add_marker(pytest.mark.skip(reason="Docker daemon unavailable"))
