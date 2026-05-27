# GrepSeek Notebooks

The notebooks are for interactive demos and inspection. They are not the primary
way to run long jobs.

Hosted Google Colab does not run the GrepSeek Docker or Apptainer image
directly. The recommended container workflow is:

1. Run the container on a machine or cluster with Docker or Apptainer.
2. Serve GrepSeek with `bash containers/serve_vllm.sh`.
3. Point a notebook at the OpenAI-compatible endpoint.

## Notebooks

```text
GrepSeek_container_endpoint_demo.ipynb
```

Lightweight endpoint-client notebook. It does not launch a container. Use it
when a vLLM server is already running and reachable through `BASE_URL`.

```text
GrepSeek_demo.ipynb
```

Standalone Colab-style demo that installs dependencies in the notebook runtime.
Use it for a quick interactive demo when you are not using the container image.

## Recommended Use

For container-backed inference:

```bash
cd <project root>
MODEL_PATH=alireza7/GrepSeek-Qwen3.5-9B-GRPO PORT=10730 TP_SIZE=1 \
  bash containers/serve_vllm.sh
```

Then open `GrepSeek_container_endpoint_demo.ipynb` and set:

```python
BASE_URL = "http://<host>:10730/v1"
MODEL = "grepseek"
API_KEY = "EMPTY"
```
