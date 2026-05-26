# Training environment (SFT + RL)

The SFT and RL stages run on **verl** and need a heavier environment than the
data-gen stage (CUDA, PyTorch, flash-attn, vLLM, …). Two exact snapshots of the
environment used in the paper are provided — pick one.

**Versions:** Python **3.12.9**, CUDA **12.8**, PyTorch **2.10.0+cu128**,
vLLM **0.17.0**, flash-attn **2.8.3**, flash-linear-attention **0.4.2**,
transformers pinned to a git commit with Qwen3.5 support.

## Option A — conda (most exact)

[`environment-train.yml`](environment-train.yml) is a direct
`conda env export` of the env (conda + pip packages, exact build strings):

```bash
conda env create -f environment-train.yml      # creates env "grepseek"
conda activate grepseek
```

> Build strings target Linux/x86-64 + CUDA 12.8; on a different platform use
> Option B instead.

## Option B — pip (portable, recommended)

This is the **exact ordered recipe we re-ran from scratch and verified** (a fresh
conda env built this way imports torch+cuda, transformers-git, flash-attn,
causal-conv1d, fla, vllm, ray, tensordict, and verl, and trains a 5-step SFT
smoke). **Order matters and the steps are not interchangeable:**

- flash-attn compiles from source against torch (no wheel for torch 2.10) and
  needs a CUDA toolkit + `g++≤12` (recent system gcc is rejected by nvcc). Cap
  `MAX_JOBS` — the parallel `nvcc` passes are memory-heavy and `$(nproc)` OOMs the
  compile on high-core nodes (we use `MAX_JOBS=4`).
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
conda install -c conda-forge cuda-toolkit=12.8 gxx_linux-64=12 gcc_linux-64=12
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
#       export PYTHONPATH=$PWD/verl:$PYTHONPATH
#    (optional editable install instead: `pip install -e verl --no-deps`)
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
