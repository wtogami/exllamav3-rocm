# Building on Fedora

Fedora packages ROCm under `/usr` instead of the upstream `/opt/rocm` tree, which breaks a few
assumptions in `rocm/scripts/build.sh`, and its GCC is newer than any distro AMD validates
against. This page lists the build packages and the three distribution-specific problems, and
points at `rocm/scripts/build_fedora.sh`, which handles all of them.

Verified on Fedora 44 (kernel `7.2.8-200.fc44.x86_64`), Radeon RX 7900 XTX (`gfx1100`),
Fedora's ROCm 7.1.1 packages (`rocm-7.1.0-3.fc44`, `hipcc-20-13.rocm7.1.1.fc44`), and the
Python 3.12 venv from `rocm/scripts/setup_env.sh` with `torch 2.13.0+rocm7.2` and
`triton-rocm 3.7.1`.

Everything past the build step — models, TabbyAPI, the serving configs — is unchanged from the
main README.

## Build packages

```bash
sudo dnf install rocm rocm-devel                  # ROCm runtime plus every -devel package
sudo dnf install python3.12 python3.12-devel      # interpreter AND its headers
```

| What the build needs | Fedora package | Notes |
| --- | --- | --- |
| `hipcc`, ROCm clang 20, rocBLAS / hipBLAS / hipBLASLt / rocFFT / RCCL headers, hipify | `rocm-devel` | Development metapackage: pulls `hipcc`, `rocm-clang-20`, `rocm-hip-devel`, `rocblas-devel`, `hipblas-devel`, `hipblaslt-devel`, `rocm-cmake` (which pulls `cmake`). `rocm` alone is the runtime only. |
| HIP device nodes | none | `/dev/kfd` and `/dev/dri/renderD128` come from the in-kernel `amdgpu` driver; the AMDGPU KFD is not a Fedora kernel module you have to install. Put the user in the `video` group. |
| `Python.h` for the interpreter the venv uses | `python3.12-devel` | The single genuinely missing dependency. `python3.12` ships only the interpreter, and `python3.12-libs` only `/usr/include/python3.12/pyconfig-64.h`, so a venv built from it has an empty `include/python3.12` directory. |
| `ninja` | none (pip) | `setup_env.sh` installs it into the venv; make sure the venv is active so its `bin` is on `PATH`. |
| Host C++ compiler | `rocm-clang-20` | Used as `CC`/`CXX`. Fedora's standalone `clang` package is not involved (installing or removing it changes nothing), and g++ cannot parse the HIP bf16 headers. |

SELinux stays `enforcing`; no policy tweaks were needed for an ordinary user shell.

## Known-good build command

```bash
cd ~/exllamav3-rocm && source .venv-rocm/bin/activate
rocm/scripts/build_fedora.sh --clean          # 192 targets, ~14.5 min on 32 cores
```

`--clean` first deletes `build/` and the generated hipified sources (`*.hip`, `*_hip.cpp`,
`*_hip.cuh`, `*_hip.h`, `hip_drv.*`, `hip_host.*`), so a run after a header edit is not served
stale objects. The script prints the resolved environment before building:

```
build_fedora: CC=/usr/lib64/rocm/llvm/bin/clang
build_fedora: CXX=/usr/lib64/rocm/llvm/bin/clang++
build_fedora: ROCM_HOME=/usr  PYTORCH_ROCM_ARCH=gfx1100  MAX_JOBS=32
build_fedora: LIBRARY_PATH=/usr/lib64:/usr/lib
build_fedora: CPLUS_INCLUDE_PATH=~/.cache/exllamav3-rocm-fedora/include
```

Equivalent by hand, if you would rather not use the script:

```bash
export CC=/usr/lib64/rocm/llvm/bin/clang CXX=/usr/lib64/rocm/llvm/bin/clang++
export ROCM_HOME=/usr PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16
export LIBRARY_PATH=/usr/lib64
SHIM=$HOME/.cache/exllamav3-rocm-fedora/include; mkdir -p "$SHIM"
sed -E 's/\[\[__gnu__::__([A-Za-z_]+)__\]\]/__attribute__((__\1__))/g' \
    /usr/include/c++/16/format > "$SHIM/format"
export CPLUS_INCLUDE_PATH="$SHIM"
rocm/scripts/build.sh
```

## What breaks with the plain `build.sh` on Fedora

### 1. `/usr/llvm/bin/clang++: No such file or directory` (exit 127)

`build.sh` defaults the host compiler to `$ROCM_HOME/llvm/bin/clang`, the upstream layout. With
`ROCM_HOME=/usr` that path does not exist: Fedora installs ROCm's clang in
`/usr/lib64/rocm/llvm/bin`, which is what `hipconfig` reports as `HIP_CLANG_PATH`. Nothing is
missing from the system; only the path assumption is wrong. `build.sh` honours `CC`/`CXX`, so
exporting them is enough.

### 2. `fatal error: 'Python.h' file not found`

Nine TUs, including every `libtorch/*_hip.cpp` and `bindings_hip.cpp`, are host C++ that
includes `torch/extension.h`. Cause and package are in the table above.

### 3. `include/c++/16/format:4550:30: error: expected identifier`

Seven TUs (`libtorch/{dsv4_attn,dsv4_compressor,blocksparse_mlp,gated_delta_net,linear,mlp}_hip.cpp`
and `bindings_hip.cpp`). `/usr/include/hip/amd_detail/host_defines.h` does

```c
#define __noinline__ __attribute__((noinline))
```

and libstdc++ 15+ spells GNU attributes with the reserved form `[[__gnu__::__noinline__]]`, so
once a hip header has been seen, the macro expansion inside `<format>` is a syntax error for
clang. It reaches `<format>` because `torch/c10/util/ApproximateClock.h` includes `<chrono>`,
which on libstdc++ 15+ includes `bits/chrono_io.h`, which includes `<format>` unconditionally in
C++20 mode. TUs that pull `<string>` and `<memory>` before the hip headers are unaffected — clang
ships `cuda_wrappers/bits/{basic_string,basic_string_tcc,shared_ptr_base}.h` to handle exactly
this on the CUDA side, and has no equivalent for HIP.

Nothing is missing: it is a libstdc++ 15/16 vs ROCm clang 20 incompatibility that AMD's own
tested distros (Ubuntu 24.04 with GCC 13, RHEL 9 with GCC 11) do not have. The fix is to shadow
`<format>` with a copy whose `[[__gnu__::__name__]]` spellings are rewritten to
`__attribute__((__name__))`; the rewrite is semantics-preserving, the copy is found first because
`CPLUS_INCLUDE_PATH` precedes the implicit system directories, and nothing in this extension
calls `std::format`. `build_fedora.sh` regenerates the copy on every run, so a libstdc++ update
is picked up automatically; delete `~/.cache/exllamav3-rocm-fedora/` to undo it.

### Link flags: not a break, but read this before debugging one

`torch.utils.cpp_extension` adds `$ROCM_HOME/lib` to the linker search path, and Fedora keeps
64-bit libraries in `/usr/lib64`, so that `-L` points at an empty directory. It still links:
GNU `ld` searches its built-in `SEARCH_DIR` entries, which include `/usr/lib64`, so `-lamdhip64`
/ `-lhipblas` / `-lrocblas` resolve anyway (verified on this box with `LIBRARY_PATH` unset).
`build_fedora.sh` exports `LIBRARY_PATH` for both directories regardless, so that a toolchain
using `lld` or `--no-default-libs` fails loudly instead of picking up the wheel's copies.
Runtime loading is unaffected either way: `/usr/lib64` is in the default loader cache, and the
torch wheel carries its own copies in `site-packages/torch/lib`.

## Caveats

- **ROCm version skew.** Fedora 44 ships ROCm 7.1.1; the numbers in the README were measured on
  7.2.4, and the `torch 2.13.0+rocm7.2` wheel bundles its own 7.2 runtime
  (`site-packages/torch/lib/libamdhip64.so` and friends). Device code therefore comes from
  clang 20 / ROCm 7.1 headers and is loaded by the wheel's 7.2 runtime. There is no 7.2 in
  Fedora's repositories, so if a kernel misbehaves, suspect this skew before the port itself: the
  alternatives are AMD's RHEL 9/10 repositories or a Rocky 9 / Ubuntu 24.04 container with
  `--device /dev/kfd --device /dev/dri`.
- **Warnings are expected.** `-Ofast` is deprecated for clang 20 (it suggests `-O3 -ffast-math`)
  and torch's headers emit `-Wnan-infinity-disabled` on the bf16 fast-math paths. Roughly 30
  warnings per TU is normal; the build is clean otherwise.
- **`MAX_JOBS`.** 32 parallel clang jobs on 32 cores peaked well under the 125 GB of RAM here;
  `MAX_JOBS=16` is the README default and is a safe floor on smaller machines.
- **Compile time.** 14m23s wall / 258m CPU with `MAX_JOBS=32`; the compiler peaked at about 11 GB
  RSS across 64 clang processes, so memory is not the constraint. The heavy TUs are the EXL3
  matmul instantiations (`quant/comp_units/exl3_*_inst_*`, 1-8 bit times 1-16 rows) and
  `lm_head`; a single `exl3_moe_inst_*` file occupies the last few minutes on its own. The
  result is a 114 MB `exllamav3_ext.cpython-312-x86_64-linux-gnu.so`.
