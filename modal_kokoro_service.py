import asyncio
import io
import os
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import modal


APP_NAME = os.getenv("APP_NAME", "kokoro-tts-short-requests")
ENDPOINT_LABEL = os.getenv("ENDPOINT_LABEL", "kokoro-tts")
GPU_TYPE = os.getenv("GPU_TYPE", "L4")
MAX_CONTAINERS = int(os.getenv("MAX_CONTAINERS", "10"))
MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "8"))
BUFFER_CONTAINERS = int(os.getenv("BUFFER_CONTAINERS", "2"))
SCALEDOWN_WINDOW = int(os.getenv("SCALEDOWN_WINDOW", "300"))
MAX_INPUTS = int(os.getenv("MAX_INPUTS", "16"))
TARGET_INPUTS = int(os.getenv("TARGET_INPUTS", "8"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "64"))
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "8"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "10"))
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("DEFAULT_SPEED", "1.0"))
LANG_CODE = os.getenv("LANG_CODE", "a")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "120"))
LOCK_TO_DEFAULT_VOICE = os.getenv("LOCK_TO_DEFAULT_VOICE", "1") == "1"
LOCK_TO_DEFAULT_SPEED = os.getenv("LOCK_TO_DEFAULT_SPEED", "1") == "1"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App(APP_NAME)


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
        self.stream_pipeline = None
        self.model = None
        self.voice_packs: dict[str, object] = {}
        self.phoneme_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.max_phoneme_cache = 10000

    async def start(self) -> None:
        import torch
        from kokoro import KModel, KPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = KModel().to(device).eval()
        self.pipeline = KPipeline(lang_code=LANG_CODE, model=False)
        self.stream_pipeline = KPipeline(lang_code=LANG_CODE, model=self.model)
        self.voice_packs[DEFAULT_VOICE] = self.pipeline.load_voice(DEFAULT_VOICE).to(device)
        self.worker_task = asyncio.create_task(self._batch_loop())

        # Warm the model before receiving benchmark traffic.
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
        await self.queue.put(
            SynthesisRequest(
                text=text,
                voice=voice,
                speed=speed,
                audio_format=audio_format,
                future=future,
            )
        )
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
        grouped: dict[tuple[str, float, str], list[SynthesisRequest]] = defaultdict(list)
        for item in batch:
            grouped[(item.voice, item.speed, item.audio_format)].append(item)

        for (voice, speed, audio_format), items in grouped.items():
            try:
                phonemes = [self._phonemize_one(item.text) for item in items]
                if any(not ps for ps in phonemes):
                    raise RuntimeError("phonemization_returned_empty")

                audios = self._synthesize_batch(
                    phonemes=phonemes,
                    voice=voice,
                    speed=speed,
                    audio_format=audio_format,
                )
                for item, audio in zip(items, audios, strict=True):
                    item.future.set_result(audio)
            except Exception as exc:
                for item in items:
                    item.future.set_exception(exc)

    def _phonemize_one(self, text: str) -> str:
        if self.pipeline is None:
            raise RuntimeError("pipeline_not_ready")

        cache_key = (LANG_CODE, text)
        cached = self.phoneme_cache.get(cache_key)
        if cached is not None:
            self.phoneme_cache.move_to_end(cache_key)
            return cached

        results = list(self.pipeline(text))
        phonemes = " ".join(result.phonemes for result in results if result.phonemes).strip()
        if not phonemes:
            raise RuntimeError("phonemization_returned_empty")

        self.phoneme_cache[cache_key] = phonemes
        self.phoneme_cache.move_to_end(cache_key)
        if len(self.phoneme_cache) > self.max_phoneme_cache:
            self.phoneme_cache.popitem(last=False)
        return phonemes

    def stream_synthesize(self, text: str, voice: str, speed: float):
        import numpy as np

        if self.stream_pipeline is None:
            raise RuntimeError("stream_pipeline_not_ready")

        generator = self.stream_pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        for result in generator:
            audio = result.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            pcm16 = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16)
            yield pcm16.tobytes()

    def _get_voice_pack(self, voice: str):
        if self.pipeline is None or self.model is None:
            raise RuntimeError("pipeline_not_ready")
        pack = self.voice_packs.get(voice)
        if pack is None:
            pack = self.pipeline.load_voice(voice).to(self.model.device)
            self.voice_packs[voice] = pack
        return pack

    def _synthesize_batch(self, phonemes: list[str], voice: str, speed: float, audio_format: str) -> list[bytes]:
        import numpy as np
        import soundfile as sf
        import torch
        from torch.nn.utils.rnn import pad_sequence

        if self.model is None:
            raise RuntimeError("model_not_ready")

        ids = []
        lengths = []
        for ps in phonemes:
            token_ids = [0, *[self.model.vocab[p] for p in ps if p in self.model.vocab], 0]
            if len(token_ids) > self.model.context_length:
                raise ValueError(f"phoneme_sequence_too_long:{len(token_ids)}")
            ids.append(torch.tensor(token_ids, dtype=torch.long))
            lengths.append(len(token_ids))

        input_ids = pad_sequence(ids, batch_first=True, padding_value=0).to(self.model.device)
        input_lengths = torch.tensor(lengths, dtype=torch.long, device=self.model.device)
        voice_pack = self._get_voice_pack(voice)
        ref_s = torch.cat([voice_pack[min(len(ps), 510) - 1] for ps in phonemes], dim=0).to(self.model.device)

        with torch.inference_mode():
            audios = self._forward_batch(
                input_ids=input_ids,
                input_lengths=input_lengths,
                ref_s=ref_s,
                speed=speed,
            )

        encoded: list[bytes] = []
        for audio in audios:
            merged = audio.cpu().numpy()
            if merged.ndim > 1:
                merged = np.squeeze(merged)
            buffer = io.BytesIO()
            if audio_format != "wav":
                raise ValueError("only_wav_is_supported_in_this_template")
            sf.write(buffer, merged, 24000, format="WAV")
            encoded.append(buffer.getvalue())
        return encoded

    def _forward_batch(self, input_ids, input_lengths, ref_s, speed: float):
        import torch

        assert self.model is not None

        batch_size, max_tokens = input_ids.shape
        text_mask = torch.arange(max_tokens, device=self.model.device).unsqueeze(0) >= input_lengths.unsqueeze(1)
        bert_dur = self.model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()

        frame_lengths = []
        alignments = []
        for batch_index in range(batch_size):
            token_count = int(input_lengths[batch_index].item())
            sample_dur = pred_dur[batch_index, :token_count]
            frame_count = int(sample_dur.sum().item())
            frame_lengths.append(frame_count)

            alignment = torch.zeros((token_count, frame_count), device=self.model.device)
            indices = torch.repeat_interleave(torch.arange(token_count, device=self.model.device), sample_dur)
            alignment[indices, torch.arange(frame_count, device=self.model.device)] = 1
            alignments.append(alignment)

        max_frames = max(frame_lengths)
        pred_aln_trg = torch.zeros((batch_size, max_tokens, max_frames), device=self.model.device)
        for batch_index, alignment in enumerate(alignments):
            token_count, frame_count = alignment.shape
            pred_aln_trg[batch_index, :token_count, :frame_count] = alignment

        en = torch.bmm(d.transpose(-1, -2), pred_aln_trg)
        f0_pred, n_pred = self.model.predictor.F0Ntrain(en, s)
        t_en = self.model.text_encoder(input_ids, input_lengths, text_mask)
        asr = torch.bmm(t_en, pred_aln_trg)
        decoded = self.model.decoder(asr, f0_pred, n_pred, ref_s[:, :128])

        outputs = []
        for batch_index, frame_count in enumerate(frame_lengths):
            audio = decoded[batch_index].squeeze()
            outputs.append(audio.cpu())
        return outputs


@app.function(
    image=image,
    gpu=GPU_TYPE,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    buffer_containers=BUFFER_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=60,
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
@modal.asgi_app(label=ENDPOINT_LABEL)
def serve():
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    web_app = FastAPI(title="Kokoro TTS")
    engine = BatchingEngine()

    class TTSRequest(BaseModel):
        text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
        voice: str = DEFAULT_VOICE
        speed: float = Field(default=DEFAULT_SPEED, ge=0.8, le=1.2)
        format: str = "wav"

    @web_app.on_event("startup")
    async def startup() -> None:
        await engine.start()

    @web_app.on_event("shutdown")
    async def shutdown() -> None:
        await engine.stop()

    @web_app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "ok": True,
            "gpu_type": GPU_TYPE,
            "max_containers": MAX_CONTAINERS,
            "min_containers": MIN_CONTAINERS,
            "batch_max_size": BATCH_MAX_SIZE,
            "batch_wait_ms": BATCH_WAIT_MS,
            "target_inputs": TARGET_INPUTS,
            "max_inputs": MAX_INPUTS,
            "max_text_length": MAX_TEXT_LENGTH,
            "lock_to_default_voice": LOCK_TO_DEFAULT_VOICE,
            "lock_to_default_speed": LOCK_TO_DEFAULT_SPEED,
        }

    @web_app.post("/warmup")
    async def warmup() -> dict[str, object]:
        await engine.enqueue(
            text="Warm up request.",
            voice=DEFAULT_VOICE,
            speed=DEFAULT_SPEED,
            audio_format="wav",
        )
        return {"ok": True}

    @web_app.post("/tts")
    async def tts(request: TTSRequest) -> Response:
        if LOCK_TO_DEFAULT_VOICE and request.voice != DEFAULT_VOICE:
            raise HTTPException(status_code=400, detail="voice_locked_for_benchmark")
        if LOCK_TO_DEFAULT_SPEED and abs(request.speed - DEFAULT_SPEED) > 1e-9:
            raise HTTPException(status_code=400, detail="speed_locked_for_benchmark")
        try:
            audio = await engine.enqueue(
                text=request.text,
                voice=request.voice,
                speed=request.speed,
                audio_format=request.format,
            )
        except RuntimeError as exc:
            if str(exc) == "queue_full":
                raise HTTPException(status_code=429, detail="container_queue_full")
            raise

        return Response(content=audio, media_type="audio/wav")

    @web_app.post("/tts/stream")
    async def tts_stream(request: TTSRequest) -> StreamingResponse:
        if LOCK_TO_DEFAULT_VOICE and request.voice != DEFAULT_VOICE:
            raise HTTPException(status_code=400, detail="voice_locked_for_benchmark")
        if LOCK_TO_DEFAULT_SPEED and abs(request.speed - DEFAULT_SPEED) > 1e-9:
            raise HTTPException(status_code=400, detail="speed_locked_for_benchmark")
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

            try:
                await asyncio.to_thread(run_stream)
            except Exception:
                if queue.empty():
                    await queue.put(None)

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

    return web_app
