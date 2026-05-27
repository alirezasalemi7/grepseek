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
docker pull ghcr.io/alirezasalemi7/grepseek-retriever:v1-slim
docker pull ghcr.io/alirezasalemi7/grepseek-all:v1-slim
```

Apptainer:

```bash
bash containers/pull_images.sh
```

By default this pulls only `grepseek:v1-slim`. Pull a different image explicitly:

```bash
bash containers/pull_images.sh --image grepseek-retriever
bash containers/pull_images.sh --image grepseek-all
bash containers/pull_images.sh --all
```

These commands write SIF files under `containers/images/`.

If the packages are private, authenticate first.

Docker:

```bash
echo "$CR_PAT" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin
```

Apptainer:

```bash
export APPTAINER_DOCKER_USERNAME="$GITHUB_USER"
export APPTAINER_DOCKER_PASSWORD="$CR_PAT"
bash containers/pull_images.sh
```

## Runtime Selection

The workflow wrappers support both Apptainer and Docker:

```bash
CONTAINER_RUNTIME=auto      # default
CONTAINER_RUNTIME=apptainer
CONTAINER_RUNTIME=docker
```

`auto` uses a local Apptainer SIF when available, otherwise Docker. Apptainer
does not pull images during workflow execution; run `bash containers/pull_images.sh`
first on clusters.

All wrappers mount:

```text
<project root>        -> /workspace
containers/cache      -> /cache
```

Additional data paths are mounted by each wrapper as needed. Keep model caches,
corpus files, and outputs on a roomy filesystem.

## Serve GrepSeek

Serve the released model with vLLM:

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

## Notes

The images contain runtime environments only. They do not include model
checkpoints, the Wikipedia corpus, or this repository. The wrappers mount the
repository and the selected data paths at runtime.
