"""Read completed AIPerf runs; emit a compact JSON dataset for the report."""

import json
import sys
from pathlib import Path


def collect(root):
    records = []
    for configuration in ("baseline", "eagle3-k2"):
        for rpm in (120, 140, 180, 200, 250, 300, 350, 400, 450, 500):
            name = configuration if rpm == 120 else (
                f"sweep640-{'nospec' if configuration == 'baseline' else configuration}-{rpm}"
            )
            profile = root / name / "profile"
            data = json.loads((profile / "profile_export_aiperf.json").read_text())
            manifest = json.loads((profile / "phase_manifest.json").read_text())
            phase = next(p for p in manifest["phases"] if p["phase_kind"] == "profiling")
            settings = next(p for p in data["input_config"]["phases"] if p["kind"] == "profiling")
            assert settings["requests"] == 640, (name, settings)
            assert abs(settings["rate"] * 60 - rpm) < 0.01, (name, settings)
            assert data["is_complete"] and not data["was_cancelled"], name
            assert not phase["was_cancelled"], name
            assert data["request_latency"]["unit"] == "ms", name
            latency = data["request_latency"]
            records.append({
                "configuration": configuration,
                "rpm": rpm,
                "mean_latency_s": latency["avg"] / 1000,
                "p90_latency_s": latency["p90"] / 1000,
                "p99_latency_s": latency["p99"] / 1000,
                "mean_ttft_ms": data["time_to_first_token"]["avg"],
                "mean_output_tokens": data["usage_completion_tokens"]["avg"],
                "benchmark_duration_s": data["benchmark_duration"]["avg"],
                "successful_requests": phase["successful_request_count"],
                "error_requests": phase["error_request_count"],
                "total_requests": phase["total_request_count"],
                "error_summary": phase["error_summary"],
                "error_rate_pct": data["request_error_rate"]["avg"],
                "source": str(profile / "profile_export_aiperf.json"),
            })
    return {"request_count_per_run": 640, "runs": records}


if __name__ == "__main__":
    print(json.dumps(collect(Path(sys.argv[1])), ensure_ascii=False, indent=2))
