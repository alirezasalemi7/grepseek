import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import tqdm
from openai import OpenAI


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1024
    n: int = 1
    stop: Optional[list] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    extra_body: dict = field(default_factory=dict)


def batchify(items, batch_size: int):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def load_url_from_log_file(log_addr: str):
    with open(log_addr, "r") as f:
        lines = f.readlines()
    host = lines[0].strip()
    port = lines[1].strip()
    return f"http://{host}:{port}/v1"


def build_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


class ChatChoice:
    def __init__(self, text: str, reasoning: Optional[str] = None, tool_calls=None, finish_reason: Optional[str] = None):
        self.text = text
        self.reasoning = reasoning
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason

    def __repr__(self):
        return f"ChatChoice(text={self.text!r}, finish_reason={self.finish_reason!r})"


class ChatResponse:
    def __init__(self, choices: list):
        self.outputs = choices

    def __getitem__(self, index):
        return self.outputs[index]

    def __iter__(self):
        return iter(self.outputs)

    def __len__(self):
        return len(self.outputs)

    @property
    def text(self) -> str:
        return self.outputs[0].text if self.outputs else ""


def _request(client: OpenAI, messages: list, model: str, sampling_params: SamplingParams, max_retries: int):
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                n=sampling_params.n,
                stop=sampling_params.stop,
                presence_penalty=sampling_params.presence_penalty,
                frequency_penalty=sampling_params.frequency_penalty,
                extra_body=sampling_params.extra_body or None,
            )
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))
    print(f"[ServerLLM] giving up after {max_retries} retries: {last_err}")
    return None


def _post_process_response(response) -> "ChatResponse":
    if response is None:
        return ChatResponse([ChatChoice(text="cannot generate a response", finish_reason="error")])
    choices = []
    for choice in response.choices:
        msg = choice.message
        reasoning = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
        )
        choices.append(
            ChatChoice(
                text=msg.content or "",
                reasoning=reasoning,
                tool_calls=getattr(msg, "tool_calls", None),
                finish_reason=choice.finish_reason,
            )
        )
    return ChatResponse(choices)


class ServerLLM:
    """Chat with a vLLM OpenAI-compatible server using ChatML-style message lists.

    Each item in the input batch is a list of `{"role": ..., "content": ...}` dicts
    (the OpenAI chat format, which vLLM serializes into the model's ChatML template).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        base_url: Optional[str] = None,
        model: str = "qwen3.5-27b",
        api_key: str = "EMPTY",
        max_retries: int = 10,
        num_workers: int = 8,
        pool=None,
    ):
        if pool is not None:
            self._pool = pool
            self._mode = "pool"
            self.base_url = None
            self.model = pool.primary_model_name() or model
            self.max_retries = max_retries
            self.num_workers = max(1, num_workers)
            self.client = None
            return
        if base_url is None:
            if host is None or port is None:
                raise ValueError("Provide either base_url or both host and port.")
            base_url = build_url(host, port)
        self._pool = None
        self._mode = "single"
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.num_workers = max(1, num_workers)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        messages_batch: list,
        sampling_params: Optional[SamplingParams] = None,
        show_progress: bool = True,
    ) -> list:
        if sampling_params is None:
            sampling_params = SamplingParams()

        results = [None] * len(messages_batch)
        chunks = batchify(list(range(len(messages_batch))), self.num_workers)
        iterator = tqdm.tqdm(chunks, desc="ServerLLM") if show_progress else chunks

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for idx_chunk in iterator:
                futures = {
                    executor.submit(
                        _request,
                        self.client,
                        messages_batch[i],
                        self.model,
                        sampling_params,
                        self.max_retries,
                    ): i
                    for i in idx_chunk
                }
                for future in futures:
                    i = futures[future]
                    results[i] = self._post_process(future.result())
        return results

    def chat(self, messages: list, sampling_params: Optional[SamplingParams] = None) -> ChatResponse:
        if self._mode == "pool":
            sp = sampling_params or SamplingParams()
            return self._pool.chat(messages, sp)
        return self.generate([messages], sampling_params=sampling_params, show_progress=False)[0]

    @staticmethod
    def _post_process(response) -> ChatResponse:
        return _post_process_response(response)


if __name__ == "__main__":
    import os
    llm = ServerLLM(
        host=os.environ.get("LLM_HOST", "127.0.0.1"),
        port=int(os.environ.get("LLM_PORT", "8000")),
        model=os.environ.get("LLM_MODEL", "served-model"),
        num_workers=8,
    )
    batch = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hi in one word."},
        ],
        [
            {"role": "user", "content": "What is 2 + 2?"},
        ],
    ]
    outs = llm.generate(batch, SamplingParams(temperature=0.0, max_tokens=64))
    for r in outs:
        print(r.text)
