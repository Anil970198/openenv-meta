#!/usr/bin/env bash
set -uo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"

printf "Step 1/3: Pinging HF Space %s/reset\n" "$PING_URL"
HTTP_CODE=$(curl -s -o /tmp/gradlab-validate-curl.out -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 || printf "000")
if [ "$HTTP_CODE" != "200" ]; then
  printf "FAILED -- /reset returned HTTP %s\n" "$HTTP_CODE"
  exit 1
fi
printf "PASSED -- HF Space reset endpoint responded\n"

printf "Step 2/3: Running docker build\n"
if ! command -v docker >/dev/null 2>&1; then
  printf "FAILED -- docker command not found\n"
  exit 1
fi
docker build "$REPO_DIR"
printf "PASSED -- Docker build succeeded\n"

printf "Step 3/3: Running openenv validate\n"
if ! command -v openenv >/dev/null 2>&1; then
  printf "FAILED -- openenv command not found. Install with: pip install openenv-core\n"
  exit 1
fi
(cd "$REPO_DIR" && openenv validate)
printf "PASSED -- openenv validate succeeded\n"

printf "All checks passed.\n"
