"""
src/scheduler/pid_scheduler.py
─────────────────────────────────────────────────────────────────────────────
Novel Idea #2 — PID-Controller Adaptive Batch Scheduler for vLLM on Ascend.

Replaces static `max_num_seqs` with a closed-loop PID controller that tracks
real-time NPU utilization and KV-cache usage, dynamically adjusting batch
capacity to minimize p99 TPOT latency spikes while keeping NPU utilization >85%.

Integration:
    1. Import and initialize PIDScheduler in your serving application.
    2. Poll vLLM Prometheus metrics every `interval_s` seconds.
    3. Call compute_max_seqs() to get the adjusted limit.
    4. Apply via vLLM AsyncLLMEngine reconfiguration API.

Usage (standalone monitor):
    python src/scheduler/pid_scheduler.py \
        --vllm-url http://localhost:8000 \
        --metrics-url http://localhost:9090 \
        --interval 0.5

References:
    - vLLM Prometheus metrics: /metrics endpoint, port 9090
    - Ascend npu-smi for NPU utilization
    - PID control theory: Åström & Hägglund, "PID Controllers" (1995)
"""

import argparse
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [PID] %(message)s")


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class SchedulerState:
    """Snapshot of system metrics at a point in time."""
    ts: float = 0.0
    npu_util: float = 0.0          # 0.0 – 1.0, from npu-smi or Prometheus
    kv_cache_usage: float = 0.0    # 0.0 – 1.0, vllm:gpu_cache_usage_perc
    num_running: int = 0           # vllm:num_requests_running
    num_waiting: int = 0           # vllm:num_requests_waiting
    p99_tpot_ms: float = 0.0       # vllm:time_per_output_token_seconds p99


@dataclass
class PIDConfig:
    # Setpoints
    target_npu_util: float = 0.88    # 88% NPU utilization target
    target_kv_usage: float = 0.80    # 80% KV-cache usage target

    # PID gains (tune via sweep or Ziegler-Nichols)
    kp: float = 0.35    # Proportional gain
    ki: float = 0.06    # Integral gain (anti-windup capped)
    kd: float = 0.02    # Derivative gain (noise-filtered)

    # Limits
    min_max_seqs: int = 32
    max_max_seqs: int = 1024
    integral_cap: float = 50.0   # Anti-windup: cap integral term

    # Anti-spike: don't increase too fast if KV nearly full
    kv_brake_threshold: float = 0.92
    kv_brake_factor: float = 0.5

    # Derivative filter (EMA coefficient)
    deriv_alpha: float = 0.3


# ── Controller ───────────────────────────────────────────────────────────────

class PIDScheduler:
    """
    Closed-loop PID controller for adaptive vLLM batch sizing.

    Error signal: npu_utilization error = target - actual
    Output:       delta to max_num_seqs (positive = more concurrency)

    Secondary constraint: KV-cache usage. If kv_usage > brake_threshold,
    the upward adjustment is dampened to prevent OOM thrashing.
    """

    def __init__(self, config: Optional[PIDConfig] = None,
                 initial_max_seqs: int = 128):
        self.cfg = config or PIDConfig()
        self.current_max_seqs = initial_max_seqs

        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_deriv = 0.0
        self._history: deque = deque(maxlen=60)   # last 30s at 0.5Hz

    def compute_max_seqs(self, state: SchedulerState) -> int:
        """
        Given current system state, compute the new max_num_seqs target.

        Returns:
            int — new value to apply as vLLM max_num_seqs
        """
        error = self.cfg.target_npu_util - state.npu_util

        # Integral with anti-windup
        self._integral += error
        self._integral = max(-self.cfg.integral_cap,
                             min(self.cfg.integral_cap, self._integral))

        # Derivative with EMA smoothing (reduces noise sensitivity)
        raw_deriv = error - self._prev_error
        self._filtered_deriv = (
            self.cfg.deriv_alpha * raw_deriv
            + (1 - self.cfg.deriv_alpha) * self._filtered_deriv
        )
        self._prev_error = error

        pid_output = (
            self.cfg.kp * error
            + self.cfg.ki * self._integral
            + self.cfg.kd * self._filtered_deriv
        )

        # KV-cache brake — prevent batch growth when nearly full
        if state.kv_cache_usage > self.cfg.kv_brake_threshold:
            if pid_output > 0:
                pid_output *= self.cfg.kv_brake_factor
            logger.debug(
                f"KV brake active ({state.kv_cache_usage:.1%}): "
                f"dampening upward adjustment"
            )

        # Apply as relative change
        new_max = self.current_max_seqs * (1.0 + pid_output)
        new_max = int(
            max(self.cfg.min_max_seqs,
                min(self.cfg.max_max_seqs, math.ceil(new_max)))
        )

        self._history.append({
            "ts": state.ts,
            "npu_util": state.npu_util,
            "kv_usage": state.kv_cache_usage,
            "error": error,
            "pid_output": pid_output,
            "max_seqs": new_max,
        })
        self.current_max_seqs = new_max
        return new_max

    def status(self) -> dict:
        """Return current controller state for monitoring."""
        return {
            "current_max_seqs": self.current_max_seqs,
            "integral": self._integral,
            "filtered_deriv": self._filtered_deriv,
        }


# ── Metrics collector ────────────────────────────────────────────────────────

class MetricsCollector:
    """Pulls vLLM Prometheus metrics and NPU utilization."""

    def __init__(self, metrics_url: str):
        self.metrics_url = metrics_url.rstrip("/") + "/metrics"

    def _parse_prometheus(self, text: str) -> dict[str, float]:
        values = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    values[parts[0].strip()] = float(parts[1])
                except ValueError:
                    pass
        return values

    def collect(self) -> SchedulerState:
        state = SchedulerState(ts=time.time())
        try:
            resp = requests.get(self.metrics_url, timeout=3)
            metrics = self._parse_prometheus(resp.text)

            state.kv_cache_usage = metrics.get(
                'vllm:gpu_cache_usage_perc{model_name="default"}', 0.0)
            state.num_running = int(metrics.get(
                'vllm:num_requests_running{model_name="default"}', 0))
            state.num_waiting = int(metrics.get(
                'vllm:num_requests_waiting{model_name="default"}', 0))

            # NPU utilization — try npu-smi, fall back to proxy via num_running
            state.npu_util = self._get_npu_util(state.num_running)
        except Exception as e:
            logger.warning(f"Metrics collection failed: {e}")
        return state

    def _get_npu_util(self, num_running: int) -> float:
        """Get NPU utilization from npu-smi if available, else estimate."""
        try:
            import subprocess
            result = subprocess.run(
                ["npu-smi", "info", "-t", "common", "-i", "0"],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.splitlines():
                if "AI Core" in line or "Chip" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            return float(p.strip("%")) / 100.0
                        except ValueError:
                            continue
        except Exception:
            pass
        # Proxy estimate: saturated at 256 seqs → ~95% util
        return min(0.98, num_running / 256.0)


# ── vLLM reconfiguration ─────────────────────────────────────────────────────

def apply_new_max_seqs(vllm_url: str, new_max: int):
    """
    Apply updated max_num_seqs to running vLLM server.
    NOTE: vLLM 0.10+ supports partial reconfiguration via admin endpoint.
    """
    try:
        resp = requests.post(
            f"{vllm_url}/admin/set_config",
            json={"max_num_seqs": new_max},
            timeout=3
        )
        if resp.ok:
            logger.debug(f"Applied max_num_seqs={new_max}")
        else:
            logger.warning(f"Config apply failed: {resp.status_code} — {resp.text}")
    except Exception as e:
        logger.warning(f"Could not apply config: {e}")


# ── Main loop ────────────────────────────────────────────────────────────────

def run_controller(vllm_url: str, metrics_url: str, interval_s: float,
                   initial_max_seqs: int):
    controller = PIDScheduler(initial_max_seqs=initial_max_seqs)
    collector  = MetricsCollector(metrics_url)

    logger.info(f"PID scheduler started (interval={interval_s}s, "
                f"target_npu_util={controller.cfg.target_npu_util:.0%}, "
                f"target_kv={controller.cfg.target_kv_usage:.0%})")

    while True:
        state = collector.collect()
        new_max = controller.compute_max_seqs(state)
        apply_new_max_seqs(vllm_url, new_max)
        logger.info(
            f"npu={state.npu_util:.1%} kv={state.kv_cache_usage:.1%} "
            f"running={state.num_running} → max_seqs={new_max}"
        )
        time.sleep(interval_s)


def main():
    p = argparse.ArgumentParser(description="PID adaptive batch scheduler")
    p.add_argument("--vllm-url",      default="http://localhost:8000")
    p.add_argument("--metrics-url",   default="http://localhost:9090")
    p.add_argument("--interval",      type=float, default=0.5)
    p.add_argument("--initial-seqs",  type=int, default=128)
    args = p.parse_args()

    run_controller(
        vllm_url=args.vllm_url,
        metrics_url=args.metrics_url,
        interval_s=args.interval,
        initial_max_seqs=args.initial_seqs
    )


if __name__ == "__main__":
    main()
