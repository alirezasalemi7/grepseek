# Training environment (SFT + RL)

The SFT and RL stages run on **verl** and need a heavier environment than the
data-gen stage (CUDA, PyTorch, flash-attn, vLLM, …). Use **Option B** for normal
setup; Option A is an archival conda snapshot of the original machine.

**Versions:** Python **3.12.9**, CUDA **12.8**, PyTorch **2.10.0+cu128**,
vLLM **0.17.0**, flash-attn **2.8.3**, flash-linear-attention **0.4.2**,
transformers pinned to a git commit with Qwen3.5 support.

## Option A — conda snapshot (archival, machine-specific)

[`environment-train.yml`](environment-train.yml) is a direct
`conda env export` of the env (conda + pip packages, exact build strings). It
may fail on machines where those exact conda builds are unavailable in the
configured channels:

```bash
conda env create -f environment-train.yml      # creates env "grepseek"
conda activate grepseek
```

> Build strings target the original Linux/x86-64 + CUDA 12.8 setup; for a
> portable install, use Option B instead.

## Option B — pip (portable, recommended)

This is the **exact ordered recipe we re-ran from scratch and verified** (a fresh
conda env built this way imports torch+cuda, transformers-git, flash-attn,
causal-conv1d, fla, vllm, ray, tensordict, and verl, and trains a 5-step SFT
smoke). **Order matters and the steps are not interchangeable:**

- flash-attn compiles from source against torch (no wheel for torch 2.10) and
  needs a CUDA toolkit + `g++≤12` (recent system gcc is rejected by nvcc). Cap
  `MAX_JOBS` — the parallel `nvcc` passes are memory-heavy and `$(nproc)` OOMs the
  compile on high-core nodes (we use `MAX_JOBS=4`). `NVCC_THREADS` also matters:
  a `MAX_JOBS=4`, `NVCC_THREADS=4` build launches 4 ninja jobs, each with 4 nvcc
  threads.
- Install **vLLM first** so it pulls a self-consistent tree (ray, outlines,
  msgspec, xformers, …); installing the pinned list in one pass hits an
  `outlines`/`vllm` resolver conflict.
- vLLM downgrades `huggingface_hub`/`tokenizers`/`safetensors`; **force them back**
  with `--no-deps` before transformers (the git transformers needs
  `huggingface_hub>=1.x`'s `is_offline_mode`).
- Install the **Qwen3.5 git transformers last, `--no-deps`** so it doesn't refight
  the `vllm`/`peft` pins.

```bash
conda create -n grepseek python=3.12 && conda activate grepseek

# 1) CUDA 12.8 toolkit + a compatible host compiler (so flash-attn can compile)
conda install -c conda-forge cuda-toolkit=12.8 gxx_linux-64=12 gcc_linux-64=12 ripgrep -y
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

# 2) PyTorch (CUDA 12.8)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 3) build prerequisites, then the compiled kernels (~45 min; MAX_JOBS capped)
pip install psutil packaging ninja wheel setuptools
MAX_JOBS=4 pip install flash-attn==2.8.3 causal-conv1d --no-build-isolation
pip install flash-linear-attention==0.4.2

# 4) vLLM FIRST — it pulls a self-consistent tree (ray, outlines, msgspec, ...)
pip install vllm==0.17.0

# 5) the remaining verl deps — pin only tensordict; let the rest resolve vs vLLM
pip install tensordict==0.12.2 ray accelerate deepspeed peft datasets hydra-core \
    omegaconf codetiming dill einops pyarrow pandas wandb sentencepiece tensorboard \
    pylatexenc latex2sympy2_extended math_verify torchdata flashinfer-python

# 6) force the transformers-git companions back to our versions (vLLM downgraded them)
pip install --no-deps "huggingface_hub==1.8.0" "tokenizers==0.22.2" "safetensors==0.7.0"

# 7) transformers (Qwen3.5 git build) LAST, --no-deps
pip install --no-deps "transformers @ git+https://github.com/huggingface/transformers.git@f048e845684894fe60440bb8506f26ffaf7b69ac"

# 8) verl is vendored in ./verl and used via PYTHONPATH — no install needed.
#    The SFT/RL launchers add it automatically; to use it outside them:
#       export PYTHONPATH=$PWD/verl:${PYTHONPATH:-}
#    (optional editable install instead: `pip install -e verl --no-deps`)
```

### flash-attn build resources

The flash-attn build uses **host RAM and filesystem I/O**, not GPU memory. If
the build log ends with plain `Killed`, the OS likely terminated `nvcc` for
memory pressure. If jobs sit in `D` state with low read/write throughput, move
`TMPDIR`/`PIP_CACHE_DIR` to local node scratch or build the wheel in a CPU job
with fast local storage.

For the paper's A100 setup, compile only the needed architecture:

```bash
export FLASH_ATTN_CUDA_ARCHS=80     # A100 / sm80
```

Rule-of-thumb memory requests for an A100-only build:

| build knobs | suggested host RAM |
|---|---:|
| `MAX_JOBS=1 NVCC_THREADS=1` | 64 GB |
| `MAX_JOBS=4 NVCC_THREADS=4` | 100 GB |
| `MAX_JOBS=8 NVCC_THREADS=4` | 200 GB |

A GPU is not required during compilation as long as the conda CUDA toolkit
provides `nvcc` and `FLASH_ATTN_CUDA_ARCHS` is set. One useful cluster pattern is
to build a reusable wheel in a large CPU job, then install that wheel in the GPU
runtime environment:

```bash
export CUDA_HOME=$CONDA_PREFIX
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=4
export NVCC_THREADS=4
export TMPDIR=${SLURM_TMPDIR:-/tmp}/flashattn-build
export PIP_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" wheelhouse

pip wheel -v flash-attn==2.8.3 --no-build-isolation --wheel-dir wheelhouse
pip install wheelhouse/flash_attn-*.whl
```

[`requirements-train.txt`](requirements-train.txt) lists the exact versions this
recipe lands on — a reference manifest, **not** a one-shot `pip install -r` file
(the vLLM-first ordering and the `--no-deps` force-pins above can't be expressed
in a flat requirements file).

> `environment-train-freeze.txt` is also provided as the *full* `pip freeze`
> snapshot, but it is **not** directly `pip install`-able (it carries env extras
> like `gpt-oss`/`unsloth` and would overwrite the cu128 torch). Prefer the
> ordered recipe above.

## Relation to our conda env

Both files are taken directly from the `grepseek` conda env used for the
paper. The pip freeze omits exactly **3** `@ file://` local-build entries that
can't install on another machine:

| dropped | what to do |
|---|---|
| `causal-conv1d` | needed (Qwen3.5 linear-attention kernels) → built in step 3 above |
| `conda-pack`, `nvitop` | tooling only, not required |

verl and the GrepSeek RL integration code are **not** pip-installed in our env
(they run via `PYTHONPATH`), so neither file references them — verl is vendored in
[`./verl`](verl) and the GrepSeek package in [`./rl`](rl); the training launchers
add both to `PYTHONPATH`.
