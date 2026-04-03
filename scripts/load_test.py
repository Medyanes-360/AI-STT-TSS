import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class Result:
    ok: bool
    status_code: int
    latency_ms: float
    error: str | None = None


DEFAULT_TEXTS = [
    "The cat is sleeping.",
    "Can you say hello?",
    "I like red apples.",
    "The sun is warm today.",
    "Let us count to three.",
    "Please open the book.",
    "The bird can fly high.",
    "Time to brush your teeth.",
]


async def one_request(client: httpx.AsyncClient, url: str, text: str, voice: str, speed: float) -> Result:
    started = time.perf_counter()
    try:
        response = await client.post(
            url,
            json={
                "text": text,
                "voice": voice,
                "speed": speed,
                "format": "wav",
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return Result(
            ok=response.status_code == 200,
            status_code=response.status_code,
            latency_ms=latency_ms,
            error=None if response.status_code == 200 else response.text[:200],
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return Result(ok=False, status_code=0, latency_ms=latency_ms, error=str(exc))


async def bounded_gather(
    client: httpx.AsyncClient,
    url: str,
    concurrency: int,
    total_requests: int,
    voice: str,
    speed: float,
) -> list[Result]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Result] = []

    async def run(index: int) -> None:
        text = DEFAULT_TEXTS[index % len(DEFAULT_TEXTS)]
        async with semaphore:
            results.append(await one_request(client, url, text, voice, speed))

    await asyncio.gather(*(run(i) for i in range(total_requests)))
    return results


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round((pct / 100) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def build_report(concurrency: int, total_requests: int, results: list[Result]) -> dict[str, object]:
    latencies = sorted(r.latency_ms for r in results)
    oks = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]
    report: dict[str, object] = {
        "concurrency": concurrency,
        "total": total_requests,
        "ok": len(oks),
        "fail": len(failures),
    }
    if latencies:
        report.update(
            {
                "avg_ms": round(statistics.mean(latencies), 1),
                "p50_ms": round(percentile(latencies, 50), 1),
                "p95_ms": round(percentile(latencies, 95), 1),
                "p99_ms": round(percentile(latencies, 99), 1),
                "max_ms": round(max(latencies), 1),
            }
        )
    if failures:
        report["sample_failure"] = failures[0].error or failures[0].status_code
    return report


def print_report(report: dict[str, object]) -> None:
    print(
        f"concurrency={report['concurrency']} total={report['total']} "
        f"ok={report['ok']} fail={report['fail']}"
    )
    if "avg_ms" in report:
        print(
            f"avg_ms={report['avg_ms']:.1f} p50_ms={report['p50_ms']:.1f} "
            f"p95_ms={report['p95_ms']:.1f} p99_ms={report['p99_ms']:.1f} "
            f"max_ms={report['max_ms']:.1f}"
        )
    if "sample_failure" in report:
        print(f"sample_failure={report['sample_failure']}")
    print()


def write_reports(output_file: str, reports: list[dict[str, object]]) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": reports}
    path.write_text(json.dumps(payload, indent=2) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--requests-per-level", type=int, default=200)
    parser.add_argument("--output-file")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument(
        "--burst-only",
        action="store_true",
        help="send exactly N requests for concurrency level N",
    )
    parser.add_argument(
        "--levels",
        default="1,10,50,100",
        help="comma-separated concurrency levels, e.g. 1,10,50,100,500,1000",
    )
    args = parser.parse_args()

    levels = [int(part.strip()) for part in args.levels.split(",") if part.strip()]
    reports: list[dict[str, object]] = []

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        if args.warmup_requests > 0:
            await bounded_gather(
                client=client,
                url=args.url,
                concurrency=min(args.warmup_requests, 10),
                total_requests=args.warmup_requests,
                voice=args.voice,
                speed=args.speed,
            )
        for concurrency in levels:
            total_requests = concurrency if args.burst_only else max(args.requests_per_level, concurrency)
            results = await bounded_gather(
                client=client,
                url=args.url,
                concurrency=concurrency,
                total_requests=total_requests,
                voice=args.voice,
                speed=args.speed,
            )
            report = build_report(concurrency, total_requests, results)
            reports.append(report)
            print_report(report)

    if args.output_file:
        write_reports(args.output_file, reports)


if __name__ == "__main__":
    asyncio.run(main())
