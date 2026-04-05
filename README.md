# Kokoro TTS/STT Servisi: Lokal ve Modal Kurulumu

Bu repo artık tek bir servis içinde hem Kokoro tabanlı TTS hem de `faster-whisper` tabanlı STT sunar.

Ana yapı:

- [combined_kokoro_service.py](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/combined_kokoro_service.py): ortak FastAPI uygulaması, TTS queue, STT queue, shared GPU gate
- [local_kokoro_server.py](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/local_kokoro_server.py): lokal sunucu wrapper'ı
- [modal_kokoro_service.py](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/modal_kokoro_service.py): Modal deployment wrapper'ı

Servis şu endpoint'leri verir:

- `/healthz`
- `/warmup`
- `/tts`
- `/tts/stream`
- `/stt`

## 1. Mimari Özet

Mevcut mimari iki inference engine'i aynı servis içinde çalıştırır:

1. TTS tarafı `TTSBatchingEngine` ile request'leri konteyner içi kuyruğa alır.
2. Kısa bir pencere boyunca request'ler toplanır ve micro-batch olarak GPU'ya gider.
3. STT tarafı `STTEngine` ile ses upload'larını ayrı kuyruğa alır.
4. `faster-whisper` modeli startup sırasında yüklenir.
5. TTS ve STT aynı GPU'yu paylaşır.
6. Çakışmayı sınırlamak için shared `asyncio.Semaphore` kullanılır.

Bu yüzden servis aynı anda hem TTS hem STT kabul eder ama GPU tarafındaki gerçek paralellik `GPU_INFERENCE_PARALLELISM` ile kontrollü tutulur.

## 2. Gereksinimler

Python bağımlılıkları [requirements.txt](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/requirements.txt) içinde tanımlıdır:

- `modal`
- `fastapi`
- `uvicorn`
- `httpx`
- `python-multipart`
- `numpy`
- `soundfile`
- `kokoro`
- `misaki[en]`
- `torch`
- `faster-whisper`

Notlar:

- `/stt` endpoint'i multipart upload kullandığı için `python-multipart` gereklidir.
- Modal tarafında ayrıca CUDA runtime için `nvidia-cublas-cu12` ve `nvidia-cudnn-cu12` image içine eklenir.

## 3. Lokal Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Lokalde ilk deneme için küçük Whisper modeli kullanmak daha pratiktir:

```bash
STT_MODEL=tiny GPU_INFERENCE_PARALLELISM=1 python local_kokoro_server.py
```

Varsayılan lokal adres:

```text
http://127.0.0.1:8000
```

## 4. Modal Üzerinde Çalıştırma

Geliştirme modunda:

```bash
modal serve modal_kokoro_service.py
```

Kalıcı deploy:

```bash
modal deploy modal_kokoro_service.py
```

Varsayılan endpoint label:

```text
kokoro-tts
```

Tipik URL:

```text
https://<workspace>--kokoro-tts.modal.run
```

Kesin URL'yi `modal deploy` çıktısından alın.

## 5. Ortam Değişkenleri

### Genel servis ayarları

| Değişken | Varsayılan | Açıklama |
|---|---:|---|
| `APP_NAME` | `kokoro-tts-short-requests` | Modal app adı |
| `ENDPOINT_LABEL` | `kokoro-tts` | Modal ASGI label |
| `GPU_TYPE` | `L4` | Modal GPU tipi |
| `MAX_CONTAINERS` | `10` | En fazla konteyner |
| `MIN_CONTAINERS` | `8` | Sıcak tutulacak minimum konteyner |
| `BUFFER_CONTAINERS` | `2` | Ek buffer konteyner hedefi |
| `SCALEDOWN_WINDOW` | `300` | Boş konteyner kapanma süresi, saniye |
| `MAX_INPUTS` | `16` | Konteyner başına eşzamanlı input limiti |
| `TARGET_INPUTS` | `8` | Modal hedef concurrency değeri |
| `GPU_INFERENCE_PARALLELISM` | `1` | TTS ve STT'nin aynı anda GPU inference başlatabilme limiti |
| `MAX_AUDIO_BYTES` | `26214400` | `/stt` için maksimum upload boyutu |

### TTS ayarları

| Değişken | Varsayılan | Açıklama |
|---|---:|---|
| `MAX_QUEUE_SIZE` | `64` | TTS kuyruk limiti |
| `BATCH_MAX_SIZE` | `8` | TTS batch boyutu |
| `BATCH_WAIT_MS` | `10` | TTS batch toplama penceresi |
| `DEFAULT_VOICE` | `af_heart` | Varsayılan voice |
| `DEFAULT_SPEED` | `1.0` | Varsayılan hız |
| `LANG_CODE` | `a` | Kokoro dil kodu |
| `MAX_TEXT_LENGTH` | `120` | `/tts` text alanı üst sınırı |
| `LOCK_TO_DEFAULT_VOICE` | `1` | `1` ise voice sabitlenir |
| `LOCK_TO_DEFAULT_SPEED` | `1` | `1` ise speed sabitlenir |

### STT ayarları

| Değişken | Varsayılan | Açıklama |
|---|---:|---|
| `STT_MODEL` | `distil-large-v3` | Whisper modeli |
| `STT_COMPUTE_TYPE` | boş | Boşsa GPU'da `float16`, CPU'da `int8` seçilir |
| `STT_TASK` | `transcribe` | Varsayılan STT task'i |
| `STT_LANGUAGE` | boş | Boş ise otomatik dil tespiti |
| `STT_BEAM_SIZE` | `5` | Beam size |
| `STT_ENABLE_VAD` | `1` | VAD filter açık mı |
| `STT_WORD_TIMESTAMPS` | `0` | Kelime timestamp dönsün mü |
| `STT_MAX_QUEUE_SIZE` | `32` | STT kuyruk limiti |
| `STT_BATCH_MAX_SIZE` | `4` | STT batch toplama limiti |
| `STT_BATCH_WAIT_MS` | `10` | STT batch bekleme süresi |
| `STT_NUM_WORKERS` | `2` | `faster-whisper` worker sayısı |
| `STT_USE_BATCHED_PIPELINE` | `0` | `BatchedInferencePipeline` kullanımı |
| `STT_BATCH_SIZE` | `8` | Batched pipeline aktifse batch size |

### Örnek Modal deploy

```bash
APP_NAME=kokoro-tts-stt-prod \
ENDPOINT_LABEL=kokoro-tts \
GPU_TYPE=L4 \
MIN_CONTAINERS=4 \
MAX_CONTAINERS=10 \
BUFFER_CONTAINERS=2 \
MAX_INPUTS=16 \
TARGET_INPUTS=8 \
GPU_INFERENCE_PARALLELISM=1 \
STT_MODEL=distil-large-v3 \
STT_NUM_WORKERS=2 \
STT_ENABLE_VAD=1 \
modal deploy modal_kokoro_service.py
```

## 6. Endpoint'ler

### `GET /healthz`

Servisin ayakta olduğunu ve aktif TTS/STT config'ini gösterir.

Örnek:

```bash
curl http://127.0.0.1:8000/healthz
```

Dönen alanlardan bazıları:

- `runtime`
- `batch_max_size`
- `gpu_inference_parallelism`
- `stt_model`
- `stt_compute_type`
- `stt_device`
- `max_audio_bytes`

### `POST /warmup`

Hem TTS hem STT tarafını ısıtır.

```bash
curl -X POST http://127.0.0.1:8000/warmup
```

### `POST /tts`

Standart TTS endpoint'i. Çıktı `audio/wav`.

İstek:

```json
{
  "text": "Hello from Kokoro",
  "voice": "af_heart",
  "speed": 1.0,
  "format": "wav"
}
```

Örnek:

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The cat is sleeping.",
    "voice": "af_heart",
    "speed": 1.0,
    "format": "wav"
  }' \
  --output out.wav
```

Olası hatalar:

- `400 voice_locked_for_benchmark`
- `400 speed_locked_for_benchmark`
- `429 container_queue_full`

### `POST /tts/stream`

Bu endpoint `wav` container dönmez. `24000 Hz`, mono, `pcm_s16le` stream üretir.

Örnek:

```bash
curl -N -X POST http://127.0.0.1:8000/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Streaming test",
    "voice": "af_heart",
    "speed": 1.0,
    "format": "wav"
  }'
```

Başlıklar:

- `Content-Type: audio/L16; rate=24000; channels=1`
- `X-Audio-Format: pcm_s16le`
- `X-Sample-Rate: 24000`
- `X-Channels: 1`

### `POST /stt`

Multipart audio upload kabul eder. Yanıt JSON transcription sonucudur.

Örnek:

```bash
curl -X POST http://127.0.0.1:8000/stt \
  -F "file=@out.wav" \
  -F "task=transcribe"
```

Dil ve word timestamps ile:

```bash
curl -X POST http://127.0.0.1:8000/stt \
  -F "file=@sample.wav" \
  -F "language=en" \
  -F "task=transcribe" \
  -F "beam_size=5" \
  -F "vad_filter=true" \
  -F "word_timestamps=true"
```

Örnek response alanları:

- `text`
- `language`
- `language_probability`
- `duration`
- `duration_after_vad`
- `model`
- `device`
- `compute_type`
- `segments`

Olası hatalar:

- `400 invalid_stt_task`
- `400 empty_audio_upload`
- `413 audio_too_large`
- `429 container_queue_full`

## 7. Lokal Test Akışı

Sunucuyu başlatın:

```bash
STT_MODEL=tiny GPU_INFERENCE_PARALLELISM=1 python3 local_kokoro_server.py
```

Sağlık kontrolü:

```bash
curl http://127.0.0.1:8000/healthz
```

TTS üretin:

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Kokoro","voice":"af_heart","speed":1.0,"format":"wav"}' \
  --output out.wav
```

Üretilen dosyayı STT'ye geri verin:

```bash
curl -X POST http://127.0.0.1:8000/stt \
  -F "file=@out.wav" \
  -F "task=transcribe"
```

## 8. Concurrency Notları

Bu servis iki farklı seviyede concurrency kullanır:

- Modal seviyesi: `@modal.concurrent(max_inputs=..., target_inputs=...)`
- Konteyner içi GPU seviyesi: `GPU_INFERENCE_PARALLELISM`

Pratik öneri:

- İlk kurulum için `GPU_INFERENCE_PARALLELISM=1` ile başlayın.
- STT için önce `STT_MODEL=tiny` veya `small` ile test edin.
- VRAM yeterliyse `STT_NUM_WORKERS` ve gerekirse `STT_USE_BATCHED_PIPELINE=1` deneyin.
- Streaming TTS de aynı shared GPU gate'i kullandığı için yoğun trafik altında uzun stream'ler diğer işleri bekletebilir.

## 9. Dosya Rolleri

- [combined_kokoro_service.py](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/combined_kokoro_service.py): ortak servis mantığı
- [modal_kokoro_service.py](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/modal_kokoro_service.py): Modal image ve deploy ayarları
- [local_kokoro_server.py](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/local_kokoro_server.py): lokal uvicorn entrypoint

## 10. Referanslar

- Faster Whisper repo: https://github.com/SYSTRAN/faster-whisper
- Modal GPU guide: https://modal.com/docs/guide/gpu
- Modal GPU comparison: https://modal.com/blog/gpu-types
