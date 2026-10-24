from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be nonnegative (0 means greedy)"
        assert self.max_tokens > 0, "max_tokens must be positive"
