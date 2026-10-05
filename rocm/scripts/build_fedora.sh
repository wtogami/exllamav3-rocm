#!/bin/bash
# Build the exllamav3 ROCm extension on Fedora, where ROCm is packaged under /usr.
# Tested: Fedora 44 (kernel 7.2.8-200.fc44), Radeon RX 7900 XTX (gfx1100), Fedora's own
# ROCm 7.1.1 packages, Python 3.12 venv with torch 2.13.0+rocm7.2 wheels.
#
# This is a thin wrapper around rocm/scripts/build.sh: it exports what Fedora needs and hands
# over. Each adjustment is explained in rocm/BUILD-FEDORA.md.
#
# usage: rocm/scripts/build_fedora.sh [--clean]      (activate the venv first)
#   --clean  also delete build/, the hipified sources and any stale extension .so first
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ROCM_HOME=${ROCM_HOME:-/usr}
PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1100}
MAX_JOBS=${MAX_JOBS:-$(nproc)}

# 1. Host C++ must go through ROCm's clang (g++ cannot parse the HIP bf16 headers). Upstream's
#    layout is $ROCM_HOME/llvm/bin; Fedora ships it in /usr/lib64/rocm/llvm/bin, which is what
#    hipconfig reports as HIP_CLANG_PATH.
HIP_CLANG=$( { hipconfig 2>/dev/null || true; } | sed -n 's/^HIP_CLANG_PATH[[:space:]]*:[[:space:]]*//p' | head -1)
CC_DIR=""
for dir in "$HIP_CLANG" "$ROCM_HOME/lib64/rocm/llvm/bin" "$ROCM_HOME/lib/rocm/llvm/bin" "$ROCM_HOME/llvm/bin"; do
    if [ -n "$dir" ] && [ -x "$dir/clang++" ]; then CC_DIR=$dir; break; fi
done
if [ -z "$CC_DIR" ]; then
    echo "build_fedora: no ROCm clang++ found next to any of \$ROCM_HOME/{lib64/rocm/,lib/rocm/,}llvm/bin" >&2
    echo "              install it with: sudo dnf install rocm-devel" >&2
    exit 1
fi
export CC=$CC_DIR/clang CXX=$CC_DIR/clang++

# 2. Fedora keeps 64-bit libraries in /usr/lib64 while torch.utils.cpp_extension adds only
#    $ROCM_HOME/lib to -L. GNU ld's default search dirs cover /usr/lib64 so the link works
#    either way; this is here so a toolchain that does not (lld, --no-default-libs) fails on the
#    Fedora tree instead of silently picking up the torch wheel's bundled copies.
LIBDIRS=""
for lib in "$ROCM_HOME/lib64" "$ROCM_HOME/lib"; do
    if [ -d "$lib" ]; then LIBDIRS="$LIBDIRS:$lib"; fi
done
export LIBRARY_PATH="${LIBDIRS#:}${LIBRARY_PATH:+:$LIBRARY_PATH}"

# 3. libstdc++ 15 and newer spell GNU attributes as [[__gnu__::__noinline__]] inside <format>,
#    and hip/amd_detail/host_defines.h does #define __noinline__ __attribute__((noinline)). The
#    expansion is a parse error for ROCm's clang, and <chrono> pulls <format> in (through
#    torch's c10/util/ApproximateClock.h), so every host TU that sees a hip header before torch
#    headers dies on it. Clang carries cuda_wrappers/bits/*.h for exactly this on the CUDA side
#    and nothing equivalent for HIP. Shadow <format> with a copy whose attribute spellings are
#    rewritten to __attribute__ form: semantics are unchanged and nothing here uses std::format.
FORMAT=""
INCLUDES=$( { "$CXX" -std=c++20 -xc++ /dev/null -E -Wp,-v 2>&1 || true; } | sed -n 's/^ //p')
for dir in $INCLUDES; do
    if [ -f "$dir/format" ]; then FORMAT=$dir/format; break; fi
done
HOST_DEFINES=$ROCM_HOME/include/hip/amd_detail/host_defines.h
if [ -n "$FORMAT" ] && grep -qF '[[__gnu__::' "$FORMAT" \
   && [ -f "$HOST_DEFINES" ] && grep -qE 'define[[:space:]]+__noinline__' "$HOST_DEFINES"; then
    SHIM=${XDG_CACHE_HOME:-$HOME/.cache}/exllamav3-rocm-fedora/include
    mkdir -p "$SHIM"
    sed -E 's/\[\[__gnu__::__([A-Za-z_]+)__\]\]/__attribute__((__\1__))/g' "$FORMAT" > "$SHIM/format"
    export CPLUS_INCLUDE_PATH="$SHIM${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    echo "build_fedora: hip's __noinline__ vs libstdc++ <format>: shimmed $FORMAT -> $SHIM/format"
fi

# 4. python3.XX-devel supplies the headers the extension is compiled against; the venv only
#    symlinks the interpreter, so a bare python3.XX package is not enough.
if ! python -c 'import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' 2>/dev/null; then
    echo "build_fedora: no Python.h for $(python -V 2>&1)" >&2
    echo "              install it with: sudo dnf install python$(python -c 'import sys;print("%d.%d" % sys.version_info[:2])')-devel" >&2
    exit 1
fi

if [ "${1:-}" = --clean ]; then
    rm -rf "$REPO/build"
    find "$REPO/exllamav3" \( -name '*.hip' -o -name '*_hip.cuh' -o -name '*_hip.cpp' -o -name '*_hip.h' \
         -o -name 'hip_drv.*' -o -name 'hip_host.*' \) -delete
    find "$REPO" -maxdepth 1 -name '*.so' -delete
    echo "build_fedora: removed build/ and the hipified sources"
fi

echo "build_fedora: CC=$CC"
echo "build_fedora: CXX=$CXX"
echo "build_fedora: ROCM_HOME=$ROCM_HOME  PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH  MAX_JOBS=$MAX_JOBS"
echo "build_fedora: LIBRARY_PATH=$LIBRARY_PATH"
echo "build_fedora: CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH:-<unchanged>}"

export ROCM_HOME PYTORCH_ROCM_ARCH MAX_JOBS
exec "$REPO/rocm/scripts/build.sh"
