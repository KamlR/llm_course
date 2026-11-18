"""Collect completed 300-second AIPerf load tests for a configuration."""

import json
import sys
from pathlib import Path


def collect(root, configuration, rpms):
    runs = []
    for rpm in rpms:
        profile = root / f"sla-{configuration}-{rpm}" / "profile"
        data = json.loads((profile / "profile_export_aiperf.json").read_text())
        manifest = json.loads((profile / "phase_manifest.json").read_text())
        phase = next(p for p in manifest["phases"] if p["phase_kind"] == "profiling")
        settings = next(p for p in data["input_config"]["phases"] if p["kind"] == "profiling")
        assert settings["duration"] == 300, (rpm, settings)
        assert abs(settings["rate"] * 60 - rpm) < 0.01, (rpm, settings)
        assert data["is_complete"] and not data["was_cancelled"], rpm
        assert not phase["was_cancelled"], rpm
        assert data["request_latency"]["unit"] == "ms", rpm
        latency = data["request_latency"]
        runs.append({
            "rpm": rpm,
            "requested_rps": settings["rate"],
            "load_duration_s": settings["duration"],
            "benchmark_duration_s": data["benchmark_duration"]["avg"],
            "mean_latency_s": latency["avg"] / 1000,
            "p90_latency_s": latency["p90"] / 1000,
            "p99_latency_s": latency["p99"] / 1000,
            "successful_requests": phase["successful_request_count"],
            "error_requests": phase["error_request_count"],
            "total_requests": phase["total_request_count"],
            "sla_met": latency["p90"] / 1000 < 15,
            "source": str(profile / "profile_export_aiperf.json"),
        })
    return {"configuration": configuration, "sla_p90_limit_s": 15, "runs": runs}


if __name__ == "__main__":
    print(json.dumps(collect(Path(sys.argv[1]), sys.argv[2], list(map(int, sys.argv[3:]))),
                     ensure_ascii=False, indent=2))
