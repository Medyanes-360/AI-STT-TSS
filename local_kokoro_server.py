import asyncio
import io
import os
import time
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("DEFAULT_SPEED", "1.0"))
LANG_CODE = os.getenv("LANG_CODE", "a")
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "128"))
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "8"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "10"))


@dataclass
class SynthesisRequest:
    text: str
    voice: str
    speed: float
    audio_format: str
    future: asyncio.Future[bytes]


class BatchingEngine:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[SynthesisRequest] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.worker_task: asyncio.Task | None = None
        self.pipeline = None

    async def start(self) -> None:
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code=LANG_CODE)
        self.worker_task = asyncio.create_task(self._batch_loop())
        await self.enqueue("Warm up request.", DEFAULT_VOICE, DEFAULT_SPEED, "wav")

    async def stop(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, text: str, voice: str, speed: float, audio_format: str) -> bytes:
        if self.queue.full():
            raise RuntimeError("queue_full")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()
        await self.queue.put(SynthesisRequest(text=text, voice=voice, speed=speed, audio_format=audio_format, future=future))
        return await future

    async def _batch_loop(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            started = time.perf_counter()
            while len(batch) < BATCH_MAX_SIZE:
                remaining = (BATCH_WAIT_MS / 1000) - (time.perf_counter() - started)
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            await self._run_batch(batch)

    async def _run_batch(self, batch: list[SynthesisRequest]) -> None:
        for item in batch:
            try:
                audio = await asyncio.to_thread(
                    self._synthesize_one,
                    item.text,
                    item.voice,
                    item.speed,
                    item.audio_format,
                )
                item.future.set_result(audio)
            except Exception as exc:
                item.future.set_exception(exc)

    def _synthesize_one(self, text: str, voice: str, speed: float, audio_format: str) -> bytes:
        import numpy as np
        import soundfile as sf

        if self.pipeline is None:
            raise RuntimeError("pipeline_not_ready")

        chunks = []
        generator = self.pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        for _, _, audio in generator:
            chunks.append(audio)

        if not chunks:
            raise RuntimeError("kokoro_returned_empty_audio")

        merged = np.concatenate(chunks)
        buffer = io.BytesIO()
        if audio_format != "wav":
            raise ValueError("only_wav_is_supported")
        sf.write(buffer, merged, 24000, format="WAV")
        return buffer.getvalue()

    def stream_synthesize(self, text: str, voice: str, speed: float):
        import numpy as np

        if self.pipeline is None:
            raise RuntimeError("pipeline_not_ready")

        generator = self.pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        for _, _, audio in generator:
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            pcm16 = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16)
            yield pcm16.tobytes()


app = FastAPI(title="Local Kokoro TTS")
engine = BatchingEngine()


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=180)
    voice: str = DEFAULT_VOICE
    speed: float = Field(default=DEFAULT_SPEED, ge=0.8, le=1.2)
    format: str = "wav"


@app.on_event("startup")
async def startup() -> None:
    await engine.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await engine.stop()


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {
        "ok": True,
        "batch_max_size": BATCH_MAX_SIZE,
        "batch_wait_ms": BATCH_WAIT_MS,
        "max_queue_size": MAX_QUEUE_SIZE,
    }


@app.post("/tts")
async def tts(request: TTSRequest) -> Response:
    try:
        audio = await engine.enqueue(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            audio_format=request.format,
        )
    except RuntimeError as exc:
        if str(exc) == "queue_full":
            raise HTTPException(status_code=429, detail="server_queue_full")
        raise
    return Response(content=audio, media_type="audio/wav")


@app.post("/tts/stream")
async def tts_stream(request: TTSRequest) -> StreamingResponse:
    if request.format != "wav":
        raise HTTPException(status_code=400, detail="stream_endpoint_outputs_pcm16")

    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def produce() -> None:
        def run_stream() -> None:
            try:
                for chunk in engine.stream_synthesize(
                    text=request.text,
                    voice=request.voice,
                    speed=request.speed,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put(f"__error__:{exc}".encode("utf-8")),
                    loop,
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        await asyncio.to_thread(run_stream)

    async def body():
        producer = asyncio.create_task(produce())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if chunk.startswith(b"__error__:"):
                    raise RuntimeError(chunk.decode("utf-8"))
                yield chunk
        finally:
            await producer

    return StreamingResponse(
        body(),
        media_type="audio/L16; rate=24000; channels=1",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
        },
    )


if __name__ == "__main__":
    uvicorn.run("local_kokoro_server:app", host="127.0.0.1", port=8000, reload=False)
