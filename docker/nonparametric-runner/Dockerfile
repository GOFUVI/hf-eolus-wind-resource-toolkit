# syntax=docker/dockerfile:1.4

FROM python:3.11-slim AS nonparametric-runner

LABEL maintainer="HF-EOLUS Wind Resource Team"
LABEL description="Containerised runner for non-parametric distribution generation"

ARG ALLOW_EMPTY_WHEELS=0
ARG ALLOW_ONLINE_INSTALL=1
ARG DOCKER_CLI_VERSION=26.1.4

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

COPY vendor/wheels /opt/wheels
COPY docker/tests/requirements-tests.lock /tmp/requirements.lock

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    if find /opt/wheels -maxdepth 1 -type f -name '*.whl' | grep -q .; then \
        pip install --no-index --find-links /opt/wheels -r /tmp/requirements.lock; \
    elif [ "$ALLOW_ONLINE_INSTALL" = "1" ]; then \
        echo "INFO: installing dependencies from PyPI (internet connection required)." ; \
        pip install -r /tmp/requirements.lock; \
    elif [ "$ALLOW_EMPTY_WHEELS" = "1" ]; then \
        echo "WARNING: building runner image without dependencies because /opt/wheels is empty." ; \
    else \
        echo "ERROR: expected wheel files under /opt/wheels to satisfy offline installation." && exit 1 ; \
    fi && \
    rm -rf /var/lib/apt/lists/*

RUN rm -f /tmp/requirements.lock

RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
    | tar -xz -C /usr/local/bin --strip-components=1 docker/docker

ENV DEBIAN_FRONTEND=

ENTRYPOINT ["python3"]
