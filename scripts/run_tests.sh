#!/usr/bin/env bash
set -euo pipefail

# Offline-friendly orchestration for the Dockerised pytest runner.
#
# Usage:
#   scripts/run_tests.sh [pytest-args...]
#   scripts/run_tests.sh --allow-empty-wheels -k io
#   IMAGE_NAME=custom-tests scripts/run_tests.sh -- --maxfail=1
#
# Environment variables:
#   IMAGE_NAME              Override the Docker image tag (default: wind-resource-tests)
#   ALLOW_EMPTY_WHEELS      Set to 1 to build without wheels (rare, for dev only)
#   ALLOW_ONLINE_INSTALL    Set to 1 (default) to let pip reach PyPI when the wheel cache is empty

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-wind-resource-tests}"
ALLOW_EMPTY_WHEELS="${ALLOW_EMPTY_WHEELS:-0}"
ALLOW_ONLINE_INSTALL="${ALLOW_ONLINE_INSTALL:-1}"
BUILD_ONLY=0
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --allow-empty-wheels)
      ALLOW_EMPTY_WHEELS=1
      shift
      ;;
    --allow-online-install)
      ALLOW_ONLINE_INSTALL=1
      shift
      ;;
    --no-online-install)
      ALLOW_ONLINE_INSTALL=0
      shift
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --)
      shift
      PYTEST_ARGS+=("$@")
      break
      ;;
    *)
      PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker binary not found. Please install Docker or Colima." >&2
  exit 1
fi

echo "==> Building ${IMAGE_NAME} (ALLOW_EMPTY_WHEELS=${ALLOW_EMPTY_WHEELS})"
docker build \
  --file "${ROOT_DIR}/docker/tests/Dockerfile" \
  --tag "${IMAGE_NAME}" \
  --build-arg "ALLOW_EMPTY_WHEELS=${ALLOW_EMPTY_WHEELS}" \
  --build-arg "ALLOW_ONLINE_INSTALL=${ALLOW_ONLINE_INSTALL}" \
  "${ROOT_DIR}"

if [[ "${BUILD_ONLY}" == "1" ]]; then
  exit 0
fi

echo "==> Running pytest inside ${IMAGE_NAME}"
if [[ ${#PYTEST_ARGS[@]} -eq 0 ]]; then
  docker run --rm \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}"
else
  docker run --rm \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    "${PYTEST_ARGS[@]}"
fi
