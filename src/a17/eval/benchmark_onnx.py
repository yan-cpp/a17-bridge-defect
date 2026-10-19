"""Benchmark an exported ONNX model with ONNX Runtime providers."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ONNX Runtime latency benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--output", default="outputs/benchmark_onnx")
    args = parser.parse_args(argv)

    requested = tuple(value.strip() for value in args.providers.split(",") if value.strip())
    providers = [provider for provider in requested if provider in ort.get_available_providers()]
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(args.model, providers=providers)
    input_meta = session.get_inputs()[0]
    input_type = np.float16 if input_meta.type == "tensor(float16)" else np.float32
    input_name = input_meta.name
    output_names = [output.name for output in session.get_outputs()]
    input_array = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=input_type)

    for _ in range(args.warmup):
        session.run(output_names, {input_name: input_array})

    latencies = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        session.run(output_names, {input_name: input_array})
        latencies.append((time.perf_counter() - start) * 1000.0)

    payload = {
        "model": args.model,
        "input": {"name": input_name, "type": str(input_type.__name__)},
        "outputs": [output.shape for output in session.get_outputs()],
        "providers_requested": list(requested),
        "providers_active": session.get_providers(),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "mean_ms": round(float(np.mean(latencies)), 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "fps": round(1000.0 / float(np.mean(latencies)), 3),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
