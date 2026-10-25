# Lab image for the screenshot arms (S0/S1/S2): headful Chromium under Xvfb, the twin's local CA
# trusted through NSS (the real omnibox shows the normal secure state; no ignore-certificate flag),
# a steady (non-blinking) text caret so images depend on the page state only, and tesseract for the
# omnibox OCR manipulation check.
#
# Base pinned by digest (= RepoDigests of the local mcr.microsoft.com/playwright/python:v1.63.0-noble).
# It ships Chrome for Testing build 1243 in /ms-playwright but no Python playwright; the locked
# playwright==1.63.0 package installed below drives that bundled browser (PLAYWRIGHT_BROWSERS_PATH
# is set by the base image), so no browser is downloaded. Build:
#   docker build -t originblind-lab:latest -f docker/lab.Dockerfile --build-arg OB_SRC_SHA256=<hash> .
# (scripts/lab_run.sh computes the hash and rebuilds whenever the working tree differs.)
FROM mcr.microsoft.com/playwright/python:v1.63.0-noble@sha256:72bd171a9ffc2b4b59532aaa6210e21014d07093120dc25528870c0b840da1f0

ENV DEBIAN_FRONTEND=noninteractive
# Historical package availability is fixed at the retained lab-environment record's UTC time.
# https://snapshot.ubuntu.com/ documents --snapshot for Ubuntu 24.04 apt/apt-get.
ENV UBUNTU_SNAPSHOT=20261016T030740Z
COPY docker/dpkg-packages.lock /tmp/dpkg-packages.lock
RUN apt-get update --snapshot "$UBUNTU_SNAPSHOT" \
    && awk '{print $1 "=" $2}' /tmp/dpkg-packages.lock \
        | xargs apt-get install --snapshot "$UBUNTU_SNAPSHOT" -y --allow-downgrades --no-install-recommends \
    && dpkg-query -W | LC_ALL=C sort > /tmp/dpkg-packages.actual \
    && cmp /tmp/dpkg-packages.lock /tmp/dpkg-packages.actual \
    && rm -rf /var/lib/apt/lists/*
# A steady text caret. Chromium on Linux takes the caret blink from GTK; by default a focused field
# blinks (on/off every 500 ms), so a grab would show or hide the caret depending on WHEN it is taken
# and identical page states would differ in pixels across arms. With the blink off, an image is a
# function of the page state only (capture.py: page_sha256 parity between S0 and S1).
RUN set -e; for v in gtk-3.0 gtk-4.0; do \
        mkdir -p "/etc/xdg/$v"; printf '[Settings]\ngtk-cursor-blink = false\n' >"/etc/xdg/$v/settings.ini"; \
    done

# Poetry runs from its own virtualenv; with virtualenvs.create=false it installs into that venv's
# base interpreter, i.e. the image's system Python 3.12, exactly the versions in poetry.lock.
ENV POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1
RUN python3 -m virtualenv -q /opt/poetry \
    && /opt/poetry/bin/pip install -q "poetry==${POETRY_VERSION}"

WORKDIR /app
# dependency layer (cached across source edits): main group + the dev group (pytest and its plugins,
# needed by the pyproject pytest configuration) for running tests inside the lab
COPY pyproject.toml poetry.lock ./
RUN /opt/poetry/bin/poetry install --no-root --only main,dev
# the project itself (editable: /app/src/originblind)
COPY README.md ./
COPY src ./src
RUN /opt/poetry/bin/poetry install --only-root
COPY . .
# build-time self-check (every step must pass, else the build fails): the Python stack imports, the
# bundled Chromium launches, tesseract has its English model, and the X/NSS tools exist
RUN chmod 0755 docker/entrypoint.sh \
    && python3 -c "import originblind.browser.capture, mss, numpy, pytesseract, PIL; from playwright.sync_api import sync_playwright as s; p = s().start(); b = p.chromium.launch(); print('chromium', b.version); b.close(); p.stop()" \
    && tesseract --list-langs 2>/dev/null | grep -qx eng \
    && for t in certutil xdotool xdpyinfo Xvfb; do command -v "$t" >/dev/null || { echo "missing $t" >&2; exit 1; }; done

ENV ORIGIN_BLIND_IN_CONTAINER=1
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["bash"]

# Provenance: the image id cannot label itself; scripts/lab_run.sh prints it and records it next to
# the results. The source hash binds the image to the working tree it was built from.
ARG OB_SRC_SHA256=unknown
LABEL org.opencontainers.image.title="originblind-lab" \
      org.opencontainers.image.base.name="mcr.microsoft.com/playwright/python:v1.63.0-noble" \
      org.opencontainers.image.base.digest="sha256:72bd171a9ffc2b4b59532aaa6210e21014d07093120dc25528870c0b840da1f0" \
      org.originblind.src_sha256="${OB_SRC_SHA256}"
