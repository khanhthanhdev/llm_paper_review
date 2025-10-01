#!/usr/bin/env bash
set -euo pipefail

GROBID_IMAGE="lfoppiano/grobid:0.8.2"
GROBID_CONTAINER_NAME="grobid_service"
GROBID_PORT="8070"
MINERU_HOST="0.0.0.0"
MINERU_PORT="8000"
MINERU_LOG_DIR="logs"
MINERU_LOG_FILE="${MINERU_LOG_DIR}/mineru_api.log"

cleanup() {
  local exit_code=$?
  if [[ -n "${MINERU_PID:-}" ]] && kill -0 "${MINERU_PID}" 2>/dev/null; then
    echo "Stopping MinerU API (PID ${MINERU_PID})..."
    kill "${MINERU_PID}" 2>/dev/null || true
    wait "${MINERU_PID}" 2>/dev/null || true
  fi

  if docker ps --format '{{.Names}}' | grep -Fxq "${GROBID_CONTAINER_NAME}"; then
    echo "Stopping GROBID container (${GROBID_CONTAINER_NAME})..."
    docker stop "${GROBID_CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

# Install required system dependencies
if ! ldconfig -p | grep -q libGL.so.1; then
  echo "Installing libgl1 (required for OpenCV and image processing)..."
  sudo apt-get update -qq && sudo apt-get install -y libgl1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found in PATH"
  exit 1
fi

if ! command -v mineru-api >/dev/null 2>&1; then
  echo "mineru-api command is required but not found in PATH"
  exit 1
fi

mkdir -p "${MINERU_LOG_DIR}"

if docker ps --format '{{.Names}}' | grep -Fxq "${GROBID_CONTAINER_NAME}"; then
  echo "GROBID container '${GROBID_CONTAINER_NAME}' already running. Skipping start."
else
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${GROBID_CONTAINER_NAME}"; then
    echo "Removing previously exited container '${GROBID_CONTAINER_NAME}'."
    docker rm "${GROBID_CONTAINER_NAME}" >/dev/null
  fi

  echo "Starting GROBID (${GROBID_IMAGE}) on port ${GROBID_PORT}..."
  docker run \
    --detach \
    --rm \
    --name "${GROBID_CONTAINER_NAME}" \
    --publish "${GROBID_PORT}:${GROBID_PORT}" \
    "${GROBID_IMAGE}" >/dev/null
fi

wait_for_http() {
  local url=$1
  local name=$2
  local max_attempts=${3:-30}
  local interval=${4:-2}

  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    if curl --silent --fail "${url}" >/dev/null 2>&1; then
      echo "${name} is responding at ${url}."
      return 0
    fi
    echo "Waiting for ${name} to become ready (${attempt}/${max_attempts})..."
    sleep "${interval}"
  done

  echo "Timed out waiting for ${name} at ${url}."
  return 1
}

MINERU_ARGS=(
  "--host" "${MINERU_HOST}"
  "--port" "${MINERU_PORT}"
)

echo "Starting MinerU API on ${MINERU_HOST}:${MINERU_PORT}..."
nohup mineru-api "${MINERU_ARGS[@]}" >"${MINERU_LOG_FILE}" 2>&1 &
MINERU_PID=$!
echo "MinerU API started with PID ${MINERU_PID}. Logs: ${MINERU_LOG_FILE}"

wait_for_http "http://localhost:${GROBID_PORT}/api/isalive" "GROBID"
wait_for_http "http://localhost:${MINERU_PORT}/docs" "MinerU API"

echo "Both services are up."

if [[ $# -gt 0 ]]; then
  echo "Running command: $*"
  "$@"
  echo "Command finished. Services remain running."
fi

echo "Press Ctrl+C to stop both services."

wait "${MINERU_PID}"
