import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run Nano-vLLM with a local or Hugging Face model.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HUB_CACHE", str(Path(__file__).resolve().parents[2] / "hf_cache")),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--rms-norm-backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--qk-rope-backend", choices=["torch", "triton"], default="torch")
    args = parser.parse_args()

    path = os.path.expanduser(args.model)
    if not os.path.isdir(path):
        path = snapshot_download(
            args.model,
            cache_dir=args.cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        )
    print(f"Model directory: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        rms_norm_backend=args.rms_norm_backend,
        qk_rope_backend=args.qk_rope_backend,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
