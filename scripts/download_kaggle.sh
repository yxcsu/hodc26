#!/usr/bin/env bash
set -euo pipefail

COMP="hyperspectral-object-detection-challenge-2026"
BASE="https://www.kaggle.com/api/v1/competitions/data/download/${COMP}"
LIST="${1:-/tmp/kaggle_names_200.txt}"
OUT="${2:-data/raw}"
JOBS="${JOBS:-24}"

if [[ ! -s "$HOME/.kaggle/access_token" ]]; then
  echo "Missing ~/.kaggle/access_token" >&2
  exit 2
fi
if [[ ! -s "$LIST" ]]; then
  echo "Missing file list: $LIST" >&2
  exit 2
fi

export TOKEN="$(cat "$HOME/.kaggle/access_token")"
export BASE OUT

download_one() {
  local f="$1"
  local enc="${f//\//%2F}"
  local target="$OUT/$f"
  local tmp="${target}.part"
  mkdir -p "$(dirname "$target")"
  [[ -s "$target" ]] && return 0
  curl -L \
    --retry 6 \
    --retry-all-errors \
    --retry-delay 1 \
    --connect-timeout 20 \
    --fail \
    --silent \
    --show-error \
    -H "Authorization: Bearer $TOKEN" \
    "$BASE/$enc" \
    -o "$tmp"
  mv "$tmp" "$target"
}
export -f download_one

mkdir -p "$OUT"
xargs -P "$JOBS" -I{} bash -c 'download_one "$1"' _ "{}" < "$LIST"

