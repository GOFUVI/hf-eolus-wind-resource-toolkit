#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-wind-resource-nonparametric}
DOCKERFILE=${DOCKERFILE:-docker/nonparametric-runner/Dockerfile}
ALLOW_ONLINE_INSTALL=${ALLOW_ONLINE_INSTALL:-1}
ALLOW_EMPTY_WHEELS=${ALLOW_EMPTY_WHEELS:-0}
FORCE_REBUILD=${FORCE_REBUILD:-0}
HOST_WORKSPACE_PATH=${HOST_WORKSPACE_PATH:-${PWD}}

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 || [[ "${FORCE_REBUILD}" == "1" ]]; then
    docker build \
        --build-arg ALLOW_ONLINE_INSTALL="${ALLOW_ONLINE_INSTALL}" \
        --build-arg ALLOW_EMPTY_WHEELS="${ALLOW_EMPTY_WHEELS}" \
        -t "${IMAGE_NAME}" \
        -f "${DOCKERFILE}" \
        .
fi

if [[ -n "${DOCKER_SOCKET:-}" ]]; then
    SOCKET_PATH="${DOCKER_SOCKET}"
else
    CANDIDATES=("/var/run/docker.sock" "${HOME}/.docker/run/docker.sock")
    SOCKET_PATH=""
    for candidate in "${CANDIDATES[@]}"; do
        if [[ -S "${candidate}" ]]; then
            SOCKET_PATH="${candidate}"
            break
        fi
    done
fi

if [[ -z "${SOCKET_PATH}" ]]; then
    echo "Docker socket not found. Set DOCKER_SOCKET to its path." >&2
    exit 1
fi

docker run --rm \
    -v "${PWD}":/workspace \
    -w /workspace \
    -v "${SOCKET_PATH}":/var/run/docker.sock \
    -e HOST_WORKSPACE_PATH="${HOST_WORKSPACE_PATH}" \
    "${IMAGE_NAME}" \
    scripts/generate_nonparametric_distributions.py "$@"
