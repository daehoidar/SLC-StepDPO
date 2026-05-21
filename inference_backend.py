"""inference_backend.py — vLLM이 없는 환경(Mac M-series 등)을 위한 transformers fallback.

3_build_pairs.py·5_evaluate.py가 vLLM의 `LLM` + `SamplingParams` 인터페이스를
사용한다. 이 모듈은 동일 인터페이스를 흉내내는 transformers 기반 wrapper를
제공해 Mac 로컬 테스트를 가능하게 한다.

사용:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        from inference_backend import TransformersLLM as LLM, \
                                     TransformersSamplingParams as SamplingParams

생성 비용은 모델 로드 1회. 이후 .generate(prompts, sp) 호출은 in-place 추론.
"""
from __future__ import annotations
import warnings
from types import SimpleNamespace


class TransformersSamplingParams:
    """vLLM.SamplingParams 부분 호환. temperature/max_tokens/n/top_p/stop만 지원."""

    def __init__(self, temperature: float = 1.0, max_tokens: int = 512,
                 n: int = 1, top_p: float = 0.95, stop=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.top_p = top_p
        self.stop = stop or []


class TransformersLLM:
    """vLLM.LLM 부분 호환. .generate(prompts, sampling_params)만 노출.

    반환 형식도 vLLM과 같게 흉내: list[RequestOutput], 각 RequestOutput은
    .outputs (list[CompletionOutput])를 가지며 CompletionOutput은 .text를 가짐.
    """

    def __init__(self, model_path: str, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16  # MPS는 bf16 부분 미지원
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        warnings.warn(f"[TransformersLLM] loading {model_path} on {self.device} ({self.dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self.torch = torch

    def generate(self, prompts: list[str], sampling_params: TransformersSamplingParams):
        """프롬프트 리스트 → RequestOutput 리스트.

        - 각 프롬프트에 대해 sampling_params.n개 샘플 생성
        - stop sequences는 생성 후 후처리로 자름 (vLLM의 native stop 토큰과 약간 다름)
        """
        torch = self.torch
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[1]
            sample_outs = []
            for _ in range(sampling_params.n):
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=sampling_params.max_tokens,
                        do_sample=sampling_params.temperature > 0,
                        temperature=max(sampling_params.temperature, 1e-3),
                        top_p=sampling_params.top_p,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                text = self.tokenizer.decode(
                    out[0][input_len:], skip_special_tokens=True
                )
                # stop sequence 후처리
                for s in sampling_params.stop:
                    if s in text:
                        text = text[: text.index(s)]
                sample_outs.append(SimpleNamespace(text=text))
            results.append(SimpleNamespace(outputs=sample_outs))
        return results


def make_llm(model_path: str, backend: str = "auto") -> tuple[object, object]:
    """편의 함수: (LLM 인스턴스, SamplingParams 클래스) 반환.

    backend:
        "auto":     vLLM 우선, ImportError 시 transformers fallback
        "vllm":     vLLM 강제 (없으면 raise)
        "transformers": transformers 강제
    """
    if backend == "vllm":
        from vllm import LLM, SamplingParams  # type: ignore
        return LLM(model_path), SamplingParams
    if backend == "transformers":
        return TransformersLLM(model_path), TransformersSamplingParams
    # auto
    try:
        from vllm import LLM, SamplingParams  # type: ignore
        return LLM(model_path), SamplingParams
    except ImportError:
        return TransformersLLM(model_path), TransformersSamplingParams
