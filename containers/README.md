# GrepSeek Containers

Prebuilt GrepSeek runtime images are available from GitHub Container Registry:

```text
ghcr.io/alirezasalemi7/grepseek:v1-slim
ghcr.io/alirezasalemi7/grepseek-retriever:v1-slim
ghcr.io/alirezasalemi7/grepseek-all:v1-slim
```

The split images are recommended. The combined image is only needed when one
runtime must contain both environments.

Run commands from the GrepSeek repository root:

```bash
cd <project root>
```

## Pull Images

Docker:

```bash
docker pull ghcr.io/alirezasalemi7/grepseek:v1-slim
```

Pull the retriever or combined image only when needed:

```bash
docker pull ghcr.io/alirezasalemi7/grepseek-retriever:v1-slim
docker pull ghcr.io/alirezasalemi7/grepseek-all:v1-slim
```

Runtime-aware helper:

```bash
bash containers/pull_images.sh
```

The helper defaults to `CONTAINER_RUNTIME=auto`: it uses Docker when the Docker
engine is usable, otherwise Apptainer.

By default this pulls only `grepseek:v1-slim`. Pull a different image explicitly:

```bash
bash containers/pull_images.sh --image grepseek-retriever
bash containers/pull_images.sh --image grepseek-all
bash containers/pull_images.sh --all
```

When Apptainer is selected, these commands write SIF files under
`containers/images/`. To force SIF creation on a machine that also has Docker,
run with `CONTAINER_RUNTIME=apptainer`.

If the packages are private, authenticate first.

Docker:

```bash
echo "$CR_PAT" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin
```

Apptainer:

```bash
export APPTAINER_DOCKER_USERNAME="$GITHUB_USER"
export APPTAINER_DOCKER_PASSWORD="$CR_PAT"
CONTAINER_RUNTIME=apptainer bash containers/pull_images.sh
```

## Runtime Selection

The workflow wrappers support both Docker and Apptainer. Runtime selection is
automatic by default:

```bash
CONTAINER_RUNTIME=auto       # default: usable Docker engine, else Apptainer
CONTAINER_RUNTIME=docker     # require a usable Docker engine
CONTAINER_RUNTIME=apptainer  # require Apptainer and use a local SIF file
```

Apptainer does not pull images during workflow execution; run
`bash containers/pull_images.sh` first on clusters or force
`CONTAINER_RUNTIME=apptainer bash containers/pull_images.sh` when preparing SIF
files on a Docker-capable machine.

All wrappers mount:

```text
<project root>        -> /workspace
containers/cache      -> /cache
```

Additional data paths are mounted by each wrapper as needed. Keep model caches,
corpus files, and outputs on a roomy filesystem.

## Serve a Model

Serve any HuggingFace model ID or local HF-format checkpoint with vLLM. For the
released GrepSeek RL model:

```bash
MODEL_PATH=alireza7/GrepSeek-Qwen3.5-9B-GRPO \
PORT=10730 \
TP_SIZE=1 \
bash containers/serve_vllm.sh
```

The server exposes an OpenAI-compatible endpoint at:

```text
http://<host>:10730/v1
```

For Docker, the wrapper publishes the selected port. For Apptainer on a cluster,
run inside an allocated GPU job and connect to the compute node and port allowed
by your cluster.

The same wrapper can serve the teacher model used for SFT data collection:

```bash
MODEL_PATH=<teacher model id or HF checkpoint> \
SERVED_MODEL_NAME=<teacher served model name> \
PORT=10731 \
TP_SIZE=<number of GPUs> \
bash containers/serve_vllm.sh
```

Use the selected `SERVED_MODEL_NAME`, host, and port as `LLM_MODEL`,
`LLM_HOST`, and `LLM_PORT` in the SFT data-collection step.

## Run Sample Inference

This uses a tiny generated corpus under `containers/cache` and does not require
the full Wikipedia corpus:

```bash
BASE_URL=http://<host>:10730/v1 \
MODEL=grepseek \
bash containers/run_sample_inference.sh
```

It writes outputs under:

```text
containers/cache/sample_inference/out/
```

## Run Full Inference

Use the full corpus directory containing `wiki_corpus.jsonl`:

```bash
BASE_URL=http://<host>:10730/v1 \
MODEL=grepseek \
GREPSEEK_CORPUS_ROOT=<corpus dir> \
bash containers/run_full_inference.sh \
  --input examples/questions.jsonl \
  --out_dir output/gen
```

Benchmark evaluation:

```bash
BASE_URL=http://<host>:10730/v1 \
MODEL=grepseek \
GREPSEEK_CORPUS_ROOT=<corpus dir> \
bash containers/run_full_inference.sh \
  --datasets all \
  --parallel 16 \
  --out_dir output/eval
```

All extra arguments are passed to `inference/run_inference.sh`.

## Collect SFT Data

Start a teacher model first, then run mixed HotpotQA + NQ cold-start data
collection:

```bash
LLM_HOST=<teacher host> \
LLM_PORT=<teacher port> \
LLM_MODEL=<teacher served model name> \
CORPUS_DIR=<corpus dir> \
bash containers/run_sft_data_collection.sh
```

Default collection settings:

```text
DATASETS=hotpotqa,nq
HOTPOT_N=100
NQ_N=100
SPLIT=train
PARALLEL_EXAMPLES=8
BUILD_PARQUET=1
```

Small smoke run:

```bash
LLM_HOST=<teacher host> \
LLM_PORT=<teacher port> \
LLM_MODEL=<teacher served model name> \
CORPUS_DIR=<corpus dir> \
HOTPOT_N=1 \
NQ_N=1 \
PARALLEL_EXAMPLES=1 \
bash containers/run_sft_data_collection.sh
```

Outputs are written under:

```text
sft/data_generation/output/container_<timestamp>/
```

## Run SFT Training

Train from a parquet file produced by the data-collection stage:

```bash
TRAIN_PARQUET=<path to train.parquet> \
MODEL_PATH=Qwen/Qwen3.5-9B \
NPROC=4 \
bash containers/run_sft_training.sh
```

Useful smoke-test overrides:

```bash
TRAIN_PARQUET=<path to train.parquet> \
MODEL_PATH=Qwen/Qwen3.5-9B \
NPROC=4 \
TOTAL_TRAINING_STEPS=5 \
SAVE_FREQ=-1 \
SAVE_DIR=<output dir> \
bash containers/run_sft_training.sh
```

## Merge SFT Checkpoint

SFT training writes sharded FSDP checkpoints. Merge one checkpoint to
HuggingFace format before using it for RL initialization or serving. The merge
runs without GPU access and does not require a CUDA toolkit:

```bash
SFT_CKPT_DIR=<sft output dir>/global_step_<N> \
bash containers/merge_sft_checkpoint.sh
```

By default this writes:

```text
<sft output dir>/global_step_<N>/huggingface
```

Choose a different output directory with `HF_DIR`:

```bash
SFT_CKPT_DIR=<sft output dir>/global_step_<N> \
HF_DIR=<sft hf output dir> \
bash containers/merge_sft_checkpoint.sh
```

## Prepare RL Data

Build the mixed NQ + HotpotQA RL train/dev JSONL files:

```bash
bash containers/prepare_rl_data.sh
```

By default this writes:

```text
data/rl/nq_hotpot/train.jsonl
data/rl/nq_hotpot/dev.jsonl
```

Choose a different output directory or pass through `rl/prepare_rl_data.py`
options:

```bash
OUT_DIR=<rl data dir> \
bash containers/prepare_rl_data.sh \
  --nq_train_count 5000 \
  --hotpot_train_count 5000 \
  --val_count_per_source 250
```

## Run RL Training

Run GRPO with train/dev JSONL files and the full corpus:

```bash
GREPSEEK_MODEL_PATH=<sft hf checkpoint or model id> \
GREPSEEK_TRAIN_FILES=<train jsonl> \
GREPSEEK_VAL_FILES=<dev jsonl> \
GREPSEEK_CORPUS_ROOT=<corpus dir> \
NPROC=4 \
bash containers/run_rl_training.sh
```

Outputs go to `GREPSEEK_OUTPUT_DIR` if set, otherwise the default from
`rl/run_rl.sh`.

## Merge and Serve RL Checkpoint

RL training writes sharded actor checkpoints. Merge an RL checkpoint to
HuggingFace format. The merge runs without GPU access and does not require a
CUDA toolkit:

```bash
CKPT_DIR=<rl output dir>/global_step_<N> \
bash containers/merge_rl_checkpoint.sh
```

By default this writes:

```text
<rl output dir>/global_step_<N>/actor/huggingface
```

Serve the merged post-RL model:

```bash
MODEL_PATH=<rl output dir>/global_step_<N>/actor/huggingface \
PORT=10730 \
TP_SIZE=<number of GPUs> \
bash containers/serve_vllm.sh
```

## End-to-End Container Pipeline

The full from-scratch flow can be run entirely through the container wrappers:

```bash
# 1. Serve a teacher model.
MODEL_PATH=<teacher model id or HF checkpoint> \
SERVED_MODEL_NAME=<teacher served model name> \
PORT=<teacher port> \
TP_SIZE=<number of GPUs> \
bash containers/serve_vllm.sh

# 2. Collect mixed HotpotQA + NQ cold-start SFT data from that teacher.
LLM_HOST=<teacher host> \
LLM_PORT=<teacher port> \
LLM_MODEL=<teacher served model name> \
CORPUS_DIR=<corpus dir> \
bash containers/run_sft_data_collection.sh

# 3. Train SFT.
TRAIN_PARQUET=<sft parquet dir>/train.parquet \
MODEL_PATH=Qwen/Qwen3.5-9B \
NPROC=4 \
SAVE_DIR=<sft output dir> \
bash containers/run_sft_training.sh

# 4. Merge the SFT checkpoint to HF format.
SFT_CKPT_DIR=<sft output dir>/global_step_<N> \
bash containers/merge_sft_checkpoint.sh

# 5. Prepare RL data, then train RL from the merged SFT checkpoint.
bash containers/prepare_rl_data.sh
GREPSEEK_MODEL_PATH=<sft output dir>/global_step_<N>/huggingface \
GREPSEEK_TRAIN_FILES=data/rl/nq_hotpot/train.jsonl \
GREPSEEK_VAL_FILES=data/rl/nq_hotpot/dev.jsonl \
GREPSEEK_CORPUS_ROOT=<corpus dir> \
NPROC=4 \
GREPSEEK_OUTPUT_DIR=<rl output dir> \
bash containers/run_rl_training.sh

# 6. Merge and serve the post-RL checkpoint.
CKPT_DIR=<rl output dir>/global_step_<N> \
bash containers/merge_rl_checkpoint.sh
MODEL_PATH=<rl output dir>/global_step_<N>/actor/huggingface \
PORT=10730 \
TP_SIZE=<number of GPUs> \
bash containers/serve_vllm.sh
```

Set `CONTAINER_RUNTIME=apptainer` before any wrapper command to run the same
workflow from local SIF files instead of Docker images.

## Notes

The images contain runtime environments only. They do not include model
checkpoints, the Wikipedia corpus, or this repository. The wrappers mount the
repository and the selected data paths at runtime.
