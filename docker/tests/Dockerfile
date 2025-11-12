# syntax=docker/dockerfile:1.4

FROM python:3.11-slim AS tests

LABEL maintainer="HF-EOLUS Wind Resource Team"
LABEL description="Offline-friendly test runner for hf_wind_resource"

ARG ALLOW_EMPTY_WHEELS=0
ARG ALLOW_ONLINE_INSTALL=1

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

# Copy offline wheel cache (may be empty when ALLOW_EMPTY_WHEELS=1).
COPY vendor/wheels /opt/wheels
COPY docker/tests/requirements-tests.lock /tmp/requirements-tests.lock

RUN if find /opt/wheels -maxdepth 1 -type f -name '*.whl' | grep -q .; then \
        pip install --no-index --find-links /opt/wheels -r /tmp/requirements-tests.lock; \
    elif [ "$ALLOW_ONLINE_INSTALL" = "1" ]; then \
        echo "INFO: installing dependencies from PyPI (internet connection required)." ; \
        pip install -r /tmp/requirements-tests.lock; \
    elif [ "$ALLOW_EMPTY_WHEELS" = "1" ]; then \
        echo "WARNING: building test image without dependencies because /opt/wheels is empty." ; \
    else \
        echo "ERROR: expected wheel files under /opt/wheels to satisfy offline installation." && exit 1 ; \
    fi

# Drop the lock file to keep the final image clean.
RUN rm -f /tmp/requirements-tests.lock

# Provide a default entrypoint that runs pytest when no explicit command is given.
ENTRYPOINT ["pytest"]
