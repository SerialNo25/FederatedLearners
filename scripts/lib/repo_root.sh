#!/usr/bin/env bash

repo_root_from() {
  local dir="$1"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/main.py" && -d "$dir/configs" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}
