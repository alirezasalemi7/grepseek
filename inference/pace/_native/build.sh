#!/usr/bin/env bash
# Build the native extension (materialize.c + vocab_search.c + vendored
# libsais -> materialize.so, one shared library for all of it).
#
# Loaded at runtime via ctypes from core.py, with an automatic fallback to
# pure-Python implementations if this .so is missing or fails to load --
# so building it is an optional performance step, never a correctness
# requirement (except that the vocabulary suffix-array lookup has no
# pure-Python fallback path; see core.py for details). Run this once per
# target architecture/environment (the .so is a compiled binary, not
# portable across incompatible glibc/CPU architectures) after cloning or
# updating any of the .c files:
#
#   bash inference/pace/_native/build.sh
#
# libsais (libsais.c/.h, vendored under libsais/, Apache-2.0, see
# libsais/LICENSE) provides linear-time generalized suffix array
# construction; vocab_search.c implements the substring-range binary search
# on top of it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CC="${CC:-cc}"

"${CC}" -O3 -fPIC -shared -Wall -Wextra -D_GNU_SOURCE \
    -I "${SCRIPT_DIR}/libsais" \
    -o "${SCRIPT_DIR}/materialize.so" \
    "${SCRIPT_DIR}/materialize.c" \
    "${SCRIPT_DIR}/vocab_search.c" \
    "${SCRIPT_DIR}/postings_codec.c" \
    "${SCRIPT_DIR}/verify_lines.c" \
    "${SCRIPT_DIR}/elias_fano.c" \
    "${SCRIPT_DIR}/libsais/libsais.c"

echo "Built ${SCRIPT_DIR}/materialize.so"
