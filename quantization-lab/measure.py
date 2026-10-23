"""Measure one quantization format in a fresh GPU process."""

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter

import bitsandbytes as bnb
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MIB = 1024 ** 2
PROMPT = "Explain why a language model uses a KV cache during text generation."
QUESTIONS = [
    ("Reply with only the number: What is 7 times 8?", "56"),
    ("Reply with only the city name: What is the capital of France?", "Paris"),
    ("Reply with only the next integer: 2, 4, 6, 8, ?", "10"),
]
SAMPLE_TEXT = (
    "The GPU stores a model's weights in device memory. During autoregressive "
    "generation, each newly generated token extends the sequence. A key-value "
    "cache keeps attention keys and values from earlier tokens, avoiding "
    "repeated computation.\n\nQuantization represents weights with fewer bits. "
    "A scale maps compressed values back to approximate floating-point values. "
    "The smaller representation saves memory, while rounding introduces numerical error."
)


def tokenize(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")


@torch.inference_mode()
def generate_fixed(model, inputs, count):
    # Fixed-length generation ignores EOS so all formats process equal work.
    torch.cuda.synchronize()
    started = perf_counter()
    output = model(**inputs, use_cache=True, logits_to_keep=1)
    past = output.past_key_values
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    first_token = perf_counter()
    key = past.layers[0].keys
    cache_info = {"dtype": str(key.dtype), "first_layer_key_shape_after_prefill": list(key.shape)}
    tokens = [token]
    mask = inputs["attention_mask"]
    # Each pass consumes the previously predicted token, extending the KV cache.
    for _ in range(count - 1):
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=1)
        output = model(
            input_ids=token, attention_mask=mask, past_key_values=past,
            use_cache=True, logits_to_keep=1,
        )
        past = output.past_key_values
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(token)
    torch.cuda.synchronize()
    finished = perf_counter()
    ids = torch.cat(tokens, dim=1)[0].cpu().tolist()
    return {
        "ttft_ms": (first_token - started) * 1000,
        "decode_tokens_per_second": (count - 1) / (finished - first_token),
        "generation_seconds": finished - started,
        "token_ids": ids,
        "kv_cache": cache_info,
    }


@torch.inference_mode()
def quality_checks(model, tokenizer):
    results = []
    for prompt, expected in QUESTIONS:
        batch = tokenize(tokenizer, prompt)
        output = model.generate(
            **batch, max_new_tokens=64, do_sample=False, use_cache=True,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(output[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
        normalized = " ".join(answer.split()).rstrip(".!").casefold()
        results.append({"prompt": prompt, "expected": expected, "answer": answer, "correct": normalized == expected.casefold()})
    return results


@torch.inference_mode()
def sample_text_loss(model, tokenizer):
    inputs = tokenizer(SAMPLE_TEXT, return_tensors="pt").to("cuda")
    loss = model(**inputs, labels=inputs.input_ids, use_cache=False).loss.float()
    return {
        "text": SAMPLE_TEXT, "tokens": inputs.input_ids.shape[1],
        "mean_negative_log_likelihood": loss.item(), "perplexity": loss.exp().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fp16", "int8", "nf4"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.tokens < 2 or args.repeats < 1:
        parser.error("--tokens must be at least 2 and --repeats must be positive")
    torch.manual_seed(0)
    torch.cuda.init()
    torch.cuda.synchronize()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    quantization = None
    if args.mode == "int8":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    elif args.mode == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False,
        )
    started = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map={"": 0},
        quantization_config=quantization, attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    torch.cuda.synchronize()
    load_seconds = perf_counter() - started
    load_allocated_mib = torch.cuda.memory_allocated() / MIB
    projection = model.model.layers[0].self_attn.q_proj
    converted = sum(isinstance(module, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)) for module in model.modules())
    if args.mode != "fp16" and converted == 0:
        raise RuntimeError("No linear modules were quantized")
    print(f"Loaded {args.mode}: {type(projection).__name__}, weight dtype={projection.weight.dtype}, allocated={load_allocated_mib:.1f} MiB", flush=True)
    inputs = tokenize(tokenizer, PROMPT)
    generate_fixed(model, inputs, 8)  # Warm up prefill and decode before timing.
    measurements = []
    for repeat in range(args.repeats):
        torch.cuda.reset_peak_memory_stats()
        result = generate_fixed(model, inputs, args.tokens)
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / MIB
        measurements.append(result)
        print(f"Repeat {repeat + 1}: TTFT={result['ttft_ms']:.1f} ms, decode={result['decode_tokens_per_second']:.1f} tokens/s", flush=True)
    checks = quality_checks(model, tokenizer)
    text_loss = sample_text_loss(model, tokenizer)
    report = {
        "mode": args.mode, "model": args.model, "gpu": torch.cuda.get_device_name(0),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "bitsandbytes": bnb.__version__},
        "batch_size": 1, "prompt": PROMPT, "prompt_tokens": inputs.input_ids.shape[1],
        "output_tokens_per_repeat": args.tokens, "repeats": args.repeats,
        "attention_backend": "sdpa", "non_quantized_dtype": "float16",
        "linear_compute": {"fp16": "FP16", "int8": "LLM.int8 mixed INT8/FP16", "nf4": "NF4 storage with FP16 compute"}[args.mode],
        "load_seconds": load_seconds, "load_allocated_mib": load_allocated_mib,
        "peak_inference_allocated_mib": max(row["peak_allocated_mib"] for row in measurements),
        "ttft_ms": statistics.median(row["ttft_ms"] for row in measurements),
        "decode_tokens_per_second": statistics.median(row["decode_tokens_per_second"] for row in measurements),
        "projection": {"class": type(projection).__name__, "weight_dtype": str(projection.weight.dtype),
                       "stored_shape": list(projection.weight.shape), "weight_tensor_bytes": projection.weight.untyped_storage().nbytes()},
        "quantized_linear_modules": converted, "kv_cache": measurements[0]["kv_cache"],
        "measurements": measurements, "example_text": tokenizer.decode(measurements[0]["token_ids"], skip_special_tokens=True),
        "quality_checks": checks, "exact_match_correct": sum(row["correct"] for row in checks), "exact_match_total": len(checks),
        "sample_text_loss": text_loss,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Strict answer matches: {report['exact_match_correct']}/{len(checks)}; sample-text perplexity: {text_loss['perplexity']:.3f}", flush=True)


if __name__ == "__main__":
    main()
