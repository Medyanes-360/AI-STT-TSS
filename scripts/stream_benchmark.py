import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


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


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round((pct / 100) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


async def one_stream_request(client: httpx.AsyncClient, url: str, text: str, voice: str, speed: float) -> dict[str, object]:
    started = time.perf_counter()
    first_byte_ms = None
    total_bytes = 0
    try:
        async with client.stream(
            "POST",
            url,
            json={
                "text": text,
                "voice": voice,
                "speed": speed,
                "format": "wav",
            },
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                total_ms = (time.perf_counter() - started) * 1000
                return {
                    "ok": False,
                    "status_code": response.status_code,
                    "first_byte_ms": round(first_byte_ms or total_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "bytes": len(body),
                    "error": body.decode("utf-8", errors="replace") or f"status_{response.status_code}",
                }
            async for chunk in response.aiter_bytes():
                if first_byte_ms is None:
                    first_byte_ms = (time.perf_counter() - started) * 1000
                total_bytes += len(chunk)
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "first_byte_ms": round(first_byte_ms or total_ms, 1),
            "total_ms": round(total_ms, 1),
            "bytes": total_bytes,
            "error": None if response.status_code == 200 else f"status_{response.status_code}",
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "ok": False,
            "status_code": 0,
            "first_byte_ms": round(first_byte_ms or total_ms, 1),
            "total_ms": round(total_ms, 1),
            "bytes": total_bytes,
            "error": str(exc),
        }


async def run_level(client: httpx.AsyncClient, url: str, concurrency: int, total_requests: int, voice: str, speed: float):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def run(index: int):
        async with sem:
            text = DEFAULT_TEXTS[index % len(DEFAULT_TEXTS)]
            results.append(await one_stream_request(client, url, text, voice, speed))

    await asyncio.gather(*(run(i) for i in range(total_requests)))
    return results


def summarize(concurrency: int, total_requests: int, results: list[dict[str, object]]) -> dict[str, object]:
    firsts = sorted(float(r["first_byte_ms"]) for r in results)
    totals = sorted(float(r["total_ms"]) for r in results)
    oks = sum(1 for r in results if r["ok"])
    fails = len(results) - oks
    report = {
        "concurrency": concurrency,
        "total": total_requests,
        "ok": oks,
        "fail": fails,
        "avg_first_byte_ms": round(statistics.mean(firsts), 1),
        "p50_first_byte_ms": round(percentile(firsts, 50), 1),
        "p95_first_byte_ms": round(percentile(firsts, 95), 1),
        "p99_first_byte_ms": round(percentile(firsts, 99), 1),
        "avg_total_ms": round(statistics.mean(totals), 1),
        "p50_total_ms": round(percentile(totals, 50), 1),
        "p95_total_ms": round(percentile(totals, 95), 1),
        "p99_total_ms": round(percentile(totals, 99), 1),
    }
    if fails:
        report["sample_failure"] = next(r["error"] for r in results if not r["ok"])
    return report


def print_report(report: dict[str, object]) -> None:
    print(f"concurrency={report['concurrency']} total={report['total']} ok={report['ok']} fail={report['fail']}")
    print(
        f"first_byte avg_ms={report['avg_first_byte_ms']:.1f} "
        f"p50_ms={report['p50_first_byte_ms']:.1f} "
        f"p95_ms={report['p95_first_byte_ms']:.1f} "
        f"p99_ms={report['p99_first_byte_ms']:.1f}"
    )
    print(
        f"total avg_ms={report['avg_total_ms']:.1f} "
        f"p50_ms={report['p50_total_ms']:.1f} "
        f"p95_ms={report['p95_total_ms']:.1f} "
        f"p99_ms={report['p99_total_ms']:.1f}"
    )
    if "sample_failure" in report:
        print(f"sample_failure={report['sample_failure']}")
    print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--burst-only", action="store_true")
    parser.add_argument("--requests-per-level", type=int, default=200)
    parser.add_argument("--levels", default="1,10,50,100")
    parser.add_argument("--output-file")
    args = parser.parse_args()

    levels = [int(part.strip()) for part in args.levels.split(",") if part.strip()]
    reports = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        if args.warmup_requests > 0:
            await run_level(
                client=client,
                url=args.url,
                concurrency=min(args.warmup_requests, 10),
                total_requests=args.warmup_requests,
                voice=args.voice,
                speed=args.speed,
            )
        for concurrency in levels:
            total_requests = concurrency if args.burst_only else max(args.requests_per_level, concurrency)
            results = await run_level(
                client=client,
                url=args.url,
                concurrency=concurrency,
                total_requests=total_requests,
                voice=args.voice,
                speed=args.speed,
            )
            report = summarize(concurrency, total_requests, results)
            reports.append(report)
            print_report(report)

    if args.output_file:
        path = Path(args.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"results": reports}, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
