#!/usr/bin/env bash
set -euo pipefail

candidates=(
  /usr/local/nvidia/lib64
  /usr/lib/x86_64-linux-gnu
  /lib/x86_64-linux-gnu
  /usr/lib64
  /usr/lib/wsl/lib
)

found_lib=""
for d in "${candidates[@]}"; do
  if [ -e "$d/libcuda.so.1" ]; then
    found_lib="$d/libcuda.so.1"
    break
  fi
done

# As a last resort, ask the loader cache.
if [ -z "$found_lib" ]; then
  if path=$(ldconfig -p | awk '/libcuda\.so\.1/{print $NF; exit}'); then
    found_lib="$path"
  fi
fi

if [ -n "$found_lib" ]; then
  target_dir="$(dirname "$found_lib")"
  if [ -w "$target_dir" ]; then
    [ -e "$target_dir/libcuda.so" ] || ln -s "$found_lib" "$target_dir/libcuda.so"
    echo "$target_dir" > /etc/ld.so.conf.d/nvidia.conf
  else
    mkdir -p /usr/local/lib
    [ -e /usr/local/lib/libcuda.so ] || ln -s "$found_lib" /usr/local/lib/libcuda.so
    echo "/usr/local/lib" > /etc/ld.so.conf.d/nvidia.conf
  fi
  ldconfig || true
fi

# diagnostics (non-fatal)
ldconfig -p | grep -E 'libcuda\.so(\.1)?' || true
for d in "${candidates[@]}" /usr/local/lib; do
  [ -d "$d" ] && ls -l "$d"/libcuda.so* 2>/dev/null || true
done

exec "$@"
