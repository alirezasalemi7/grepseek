# GrepSeek Container Images

Prebuilt GrepSeek runtime images are available from GitHub Container Registry:

```text
ghcr.io/alirezasalemi7/grepseek:v1            main GrepSeek/vLLM runtime
ghcr.io/alirezasalemi7/grepseek-retriever:v1  retriever runtime
ghcr.io/alirezasalemi7/grepseek-all:v1        optional combined runtime
```

The two split images are recommended. The combined image is larger and is only
needed when one container must contain both environments.

Run helper commands from the GrepSeek repository root.

## Docker

Pull the split images:

```bash
docker pull ghcr.io/alirezasalemi7/grepseek:v1
docker pull ghcr.io/alirezasalemi7/grepseek-retriever:v1
```

Pull the optional combined image:

```bash
docker pull ghcr.io/alirezasalemi7/grepseek-all:v1
```

If the package is private, login first:

```bash
echo "$CR_PAT" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin
```

## Apptainer

On systems that use Apptainer, pull the split images with:

```bash
bash containers/pull_images.sh
```

This creates:

```text
containers/images/grepseek_v1.sif
containers/images/grepseek-retriever_v1.sif
```

Pull the optional combined image too:

```bash
bash containers/pull_images.sh --all
```

Use another tag:

```bash
TAG=v1-slim bash containers/pull_images.sh
```

If the package is private, set Apptainer's Docker auth variables first:

```bash
export APPTAINER_DOCKER_USERNAME="$GITHUB_USER"
export APPTAINER_DOCKER_PASSWORD="$CR_PAT"

bash containers/pull_images.sh
```

## Check

Docker:

```bash
docker run --rm ghcr.io/alirezasalemi7/grepseek:v1 \
  python -c "import torch, vllm, causal_conv1d; print('grepseek ok')"

docker run --rm ghcr.io/alirezasalemi7/grepseek-retriever:v1 \
  python -c "import torch, faiss, sentence_transformers; print('retriever ok')"
```

Apptainer:

```bash
apptainer exec --nv containers/images/grepseek_v1.sif \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Notes

The images contain the runtime environments only. They do not include model
checkpoints, the Wikipedia corpus, or a local clone of this repository.
