import asyncio
import io
import os
import site
import time
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("DEFAULT_SPEED", "1.0"))
LANG_CODE = os.getenv("LANG_CODE", "a")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "120"))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "64"))
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "8"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "10"))
LOCK_TO_DEFAULT_VOICE = os.getenv("LOCK_TO_DEFAULT_VOICE", "1") == "1"
LOCK_TO_DEFAULT_SPEED = os.getenv("LOCK_TO_DEFAULT_SPEED", "1") == "1"
GPU_INFERENCE_PARALLELISM = int(os.getenv("GPU_INFERENCE_PARALLELISM", "1"))

STT_MODEL = os.getenv("STT_MODEL", "distil-large-v3")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "")
STT_TASK = os.getenv("STT_TASK", "transcribe")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "").strip() or None
STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "5"))
STT_ENABLE_VAD = os.getenv("STT_ENABLE_VAD", "1") == "1"
STT_WORD_TIMESTAMPS = os.getenv("STT_WORD_TIMESTAMPS", "0") == "1"
STT_MAX_QUEUE_SIZE = int(os.getenv("STT_MAX_QUEUE_SIZE", "32"))
STT_BATCH_MAX_SIZE = int(os.getenv("STT_BATCH_MAX_SIZE", "4"))
STT_BATCH_WAIT_MS = int(os.getenv("STT_BATCH_WAIT_MS", "10"))
STT_NUM_WORKERS = int(os.getenv("STT_NUM_WORKERS", "2"))
STT_USE_BATCHED_PIPELINE = os.getenv("STT_USE_BATCHED_PIPELINE", "0") == "1"
STT_BATCH_SIZE = int(os.getenv("STT_BATCH_SIZE", "8"))


def _configure_cuda_library_path() -> None:
    lib_dirs: list[str] = []

    def add_lib_dir(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved not in lib_dirs:
            lib_dirs.append(resolved)

    for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib", "nvidia.cuda_nvrtc.lib"):
        try:
            module = __import__(module_name, fromlist=["__file__"])
        except ImportError:
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        add_lib_dir(Path(module_file).resolve().parent)

    # Fallback: discover NVIDIA lib folders directly from site-packages.
    patterns = ("libcublas.so*", "libcudnn.so*", "libnvrtc.so*", "libnvrtc-builtins.so*")
    for site_root in site.getsitepackages():
        nvidia_root = Path(site_root) / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib_dir in nvidia_root.rglob("lib"):
            if not lib_dir.is_dir():
                continue
            if any(lib_dir.glob(pattern) for pattern in patterns):
                add_lib_dir(lib_dir)

    if not lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [part for part in current.split(":") if part]
    merged = lib_dirs + [part for part in current_parts if part not in lib_dirs]
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _default_stt_compute_type(device: str) -> str:
    if STT_COMPUTE_TYPE:
        return STT_COMPUTE_TYPE
    return "float16" if device == "cuda" else "int8"


def _build_silence_wav_bytes(duration_ms: int = 250, sample_rate: int = 16000) -> bytes:
    import numpy as np
    import soundfile as sf

    frames = max(1, int(sample_rate * duration_ms / 1000))
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(frames, dtype=np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


class SharedGPUExecutor:
    def __init__(self, max_parallel: int) -> None:
        self.semaphore = asyncio.Semaphore(max(1, max_parallel))

    @asynccontextmanager
    async def slot(self):
        async with self.semaphore:
            yield


@dataclass
class SynthesisRequest:
    text: str
    voice: str
    speed: float
    audio_format: str
    future: asyncio.Future[bytes]


@dataclass
class TranscriptionRequest:
    audio_bytes: bytes
    filename: str
    language: str | None
    task: str
    beam_size: int
    vad_filter: bool
    word_timestamps: bool
    future: asyncio.Future[dict[str, Any]]


class TTSBatchingEngine:
    def __init__(self, gpu_executor: SharedGPUExecutor) -> None:
        self.gpu_executor = gpu_executor
        self.queue: asyncio.Queue[SynthesisRequest] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.worker_task: asyncio.Task | None = None
        self.pipeline = None
        self.stream_pipeline = None
        self.model = None
        self.voice_packs: dict[str, object] = {}
        self.phoneme_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.max_phoneme_cache = 10000

    async def start(self) -> None:
        _configure_cuda_library_path()
        import torch
        from kokoro import KModel, KPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = KModel().to(device).eval()
        self.pipeline = KPipeline(lang_code=LANG_CODE, model=False)
        self.stream_pipeline = KPipeline(lang_code=LANG_CODE, model=self.model)
        self.voice_packs[DEFAULT_VOICE] = self.pipeline.load_voice(DEFAULT_VOICE).to(device)
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

                async with self.gpu_executor.slot():
                    audios = await asyncio.to_thread(
                        self._synthesize_batch,
                        phonemes,
                        voice,
                        speed,
                        audio_format,
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
        for batch_index, _frame_count in enumerate(frame_lengths):
            audio = decoded[batch_index].squeeze()
            outputs.append(audio.cpu())
        return outputs


class STTEngine:
    def __init__(self, gpu_executor: SharedGPUExecutor) -> None:
        self.gpu_executor = gpu_executor
        self.queue: asyncio.Queue[TranscriptionRequest] = asyncio.Queue(maxsize=STT_MAX_QUEUE_SIZE)
        self.worker_task: asyncio.Task | None = None
        self.model = None
        self.batched_model = None
        self.device = "cpu"
        self.compute_type = "int8"

    async def start(self) -> None:
        await asyncio.to_thread(self._load_model)
        self.worker_task = asyncio.create_task(self._batch_loop())
        await self.warmup()

    async def stop(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def warmup(self) -> None:
        silence = _build_silence_wav_bytes()
        await self.enqueue(
            audio_bytes=silence,
            filename="warmup.wav",
            language=STT_LANGUAGE,
            task=STT_TASK,
            beam_size=STT_BEAM_SIZE,
            vad_filter=STT_ENABLE_VAD,
            word_timestamps=False,
        )

    async def enqueue(
        self,
        audio_bytes: bytes,
        filename: str,
        language: str | None,
        task: str,
        beam_size: int,
        vad_filter: bool,
        word_timestamps: bool,
    ) -> dict[str, Any]:
        if self.queue.full():
            raise RuntimeError("queue_full")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        await self.queue.put(
            TranscriptionRequest(
                audio_bytes=audio_bytes,
                filename=filename,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                future=future,
            )
        )
        return await future

    def _load_model(self) -> None:
        import torch
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        _configure_cuda_library_path()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = _default_stt_compute_type(self.device)
        self.model = WhisperModel(
            STT_MODEL,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=STT_NUM_WORKERS,
        )
        if STT_USE_BATCHED_PIPELINE:
            self.batched_model = BatchedInferencePipeline(model=self.model)

    async def _batch_loop(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            started = time.perf_counter()

            while len(batch) < STT_BATCH_MAX_SIZE:
                remaining = (STT_BATCH_WAIT_MS / 1000) - (time.perf_counter() - started)
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            await self._run_batch(batch)

    async def _run_batch(self, batch: list[TranscriptionRequest]) -> None:
        async def handle(item: TranscriptionRequest) -> None:
            try:
                async with self.gpu_executor.slot():
                    result = await asyncio.to_thread(self._transcribe_one, item)
                item.future.set_result(result)
            except Exception as exc:
                item.future.set_exception(exc)

        await asyncio.gather(*(handle(item) for item in batch))

    def _transcribe_one(self, item: TranscriptionRequest) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("stt_model_not_ready")

        transcriber = self.batched_model or self.model
        kwargs: dict[str, Any] = {
            "audio": io.BytesIO(item.audio_bytes),
            "language": item.language,
            "task": item.task,
            "beam_size": item.beam_size,
            "vad_filter": item.vad_filter,
            "word_timestamps": item.word_timestamps,
        }
        if self.batched_model is not None:
            kwargs["batch_size"] = STT_BATCH_SIZE

        segments, info = transcriber.transcribe(**kwargs)
        text_parts: list[str] = []
        serialized_segments: list[dict[str, Any]] = []

        for segment in segments:
            text_parts.append(segment.text)
            words = None
            if item.word_timestamps and getattr(segment, "words", None):
                words = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]

            serialized_segments.append(
                {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob,
                    "words": words,
                }
            )

        return {
            "text": "".join(text_parts).strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": getattr(info, "duration", None),
            "duration_after_vad": getattr(info, "duration_after_vad", None),
            "all_language_probs": getattr(info, "all_language_probs", None),
            "filename": item.filename,
            "model": STT_MODEL,
            "device": self.device,
            "compute_type": self.compute_type,
            "segments": serialized_segments,
        }


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    voice: str = DEFAULT_VOICE
    speed: float = Field(default=DEFAULT_SPEED, ge=0.8, le=1.2)
    format: str = "wav"


def create_web_app(title: str, runtime_label: str) -> FastAPI:
    web_app = FastAPI(title=title)
    gpu_executor = SharedGPUExecutor(GPU_INFERENCE_PARALLELISM)
    tts_engine = TTSBatchingEngine(gpu_executor)
    stt_engine = STTEngine(gpu_executor)

    @web_app.on_event("startup")
    async def startup() -> None:
        await tts_engine.start()
        await stt_engine.start()

    @web_app.on_event("shutdown")
    async def shutdown() -> None:
        await asyncio.gather(tts_engine.stop(), stt_engine.stop())

    @web_app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "runtime": runtime_label,
            "batch_max_size": BATCH_MAX_SIZE,
            "batch_wait_ms": BATCH_WAIT_MS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "gpu_inference_parallelism": GPU_INFERENCE_PARALLELISM,
            "default_voice": DEFAULT_VOICE,
            "default_speed": DEFAULT_SPEED,
            "max_text_length": MAX_TEXT_LENGTH,
            "stt_model": STT_MODEL,
            "stt_compute_type": stt_engine.compute_type,
            "stt_device": stt_engine.device,
            "stt_queue_size": STT_MAX_QUEUE_SIZE,
            "stt_num_workers": STT_NUM_WORKERS,
            "stt_use_batched_pipeline": STT_USE_BATCHED_PIPELINE,
            "max_audio_bytes": MAX_AUDIO_BYTES,
        }

    @web_app.post("/warmup")
    async def warmup() -> dict[str, bool]:
        await asyncio.gather(
            tts_engine.enqueue(
                text="Warm up request.",
                voice=DEFAULT_VOICE,
                speed=DEFAULT_SPEED,
                audio_format="wav",
            ),
            stt_engine.warmup(),
        )
        return {"ok": True}

    @web_app.post("/tts")
    async def tts(request: TTSRequest) -> Response:
        if LOCK_TO_DEFAULT_VOICE and request.voice != DEFAULT_VOICE:
            raise HTTPException(status_code=400, detail="voice_locked_for_benchmark")
        if LOCK_TO_DEFAULT_SPEED and abs(request.speed - DEFAULT_SPEED) > 1e-9:
            raise HTTPException(status_code=400, detail="speed_locked_for_benchmark")
        try:
            audio = await tts_engine.enqueue(
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
                    for chunk in tts_engine.stream_synthesize(
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
                async with gpu_executor.slot():
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

    @web_app.post("/stt")
    async def stt(
        file: Annotated[UploadFile, File(...)],
        language: Annotated[str | None, Form()] = STT_LANGUAGE,
        task: Annotated[str, Form()] = STT_TASK,
        beam_size: Annotated[int, Form(ge=1, le=20)] = STT_BEAM_SIZE,
        vad_filter: Annotated[bool, Form()] = STT_ENABLE_VAD,
        word_timestamps: Annotated[bool, Form()] = STT_WORD_TIMESTAMPS,
    ) -> dict[str, Any]:
        if task not in {"transcribe", "translate"}:
            raise HTTPException(status_code=400, detail="invalid_stt_task")

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="empty_audio_upload")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="audio_too_large")

        try:
            return await stt_engine.enqueue(
                audio_bytes=audio_bytes,
                filename=file.filename or "upload",
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
            )
        except RuntimeError as exc:
            if str(exc) == "queue_full":
                raise HTTPException(status_code=429, detail="container_queue_full")
            raise

    return web_app
