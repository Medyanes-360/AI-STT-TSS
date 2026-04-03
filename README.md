# Kokoro TTS Servisi: Modal Üzerinde Kurulum ve Kullanım

Bu repo, `modal_kokoro_service.py` dosyası üzerinden Kokoro tabanlı bir TTS servisini Modal'a deploy etmek için hazırlanmış bir baseline sunar. Servis, kısa metinleri GPU üzerinde toplu işleyerek `wav` ses dosyası üretir; ek olarak streaming endpoint ile ham `PCM16` ses akışı da verebilir.

Kodun temel hedefi yüksek eşzamanlılıkta stabil davranış elde etmektir:

- Modal konteynerlerini sıcak tutar
- Konteyner içinde kuyruk ve micro-batching uygular
- Tek tek request yerine batch forward pass çalıştırır
- Aşırı yüklenmede bekleme süresini sonsuza uzatmak yerine `429` dönebilir

## 1. Mimari Özeti

`modal_kokoro_service.py` şu yapıda çalışır:

1. Modal bir image oluşturur.
2. GPU'lu bir ASGI uygulaması ayağa kalkar.
3. FastAPI endpoint'leri `BatchingEngine` üzerinden request alır.
4. Request'ler konteyner içindeki asyncio kuyruğuna girer.
5. Batch loop, kısa bir pencere boyunca request toplar.
6. Aynı `voice`, `speed` ve `format` kombinasyonları gruplanır.
7. Metinler önce phoneme dizisine çevrilir.
8. GPU üzerinde tekil değil toplu inference çalışır.
9. Sonuç `audio/wav` ya da stream için `pcm_s16le` olarak dönülür.

## 2. Dosya Ne Yapıyor?

Ana servis dosyası: [modal_kokoro_service.py](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/modal_kokoro_service.py)

Bu dosya şu ana parçalardan oluşur:

- Modal uygulama ve image tanımı
- Ortam değişkenlerinden config yükleme
- `SynthesisRequest` veri sınıfı
- `BatchingEngine` ile kuyruk, cache ve batch inference mantığı
- FastAPI endpoint'leri: `/healthz`, `/warmup`, `/tts`, `/tts/stream`

## 3. Gereksinimler

Python bağımlılıkları [requirements.txt](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/requirements.txt) içinde tanımlı:

- `modal`
- `fastapi`
- `uvicorn`
- `httpx`
- `numpy`
- `soundfile`
- `kokoro`
- `misaki[en]`
- `torch`

Not: Mevcut bağımlılıklar English odaklı bir kurulum yapıyor. `LANG_CODE="a"` da buna uygun varsayılan değerdir.

## 4. Lokal Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Lokalde aşağıdaki dosya ile servisi çalıştırın:

```bash
local_kokoro_server.py
```

Modal CLI girişi gerekli ise:

```bash
modal setup
```

## 5. Modal Üzerinde Çalıştırma

Gelişim modunda servis:

```bash
modal serve modal_kokoro_service.py
```

Kalıcı deploy:

```bash
modal deploy modal_kokoro_service.py
```

Deploy sonrasında endpoint genel olarak şu formda olur:

```text
https://<workspace>--<endpoint-label>.modal.run
```

Bu projede varsayılan endpoint label `kokoro-tts` olduğu için tam endpoint genelde şunlardan biri olur:

```text
https://<workspace>--kokoro-tts.modal.run
https://<workspace>--kokoro-tts-dev.modal.run
```

Kesin URL'yi `modal deploy` çıktısından alın.

## 6. Ortam Değişkenleri

Servis davranışı tamamen ortam değişkenleri ile ayarlanabilir:

| Değişken | Varsayılan | Açıklama |
|---|---:|---|
| `APP_NAME` | `kokoro-tts-short-requests` | Modal uygulama adı |
| `ENDPOINT_LABEL` | `kokoro-tts` | `modal.asgi_app` label değeri |
| `GPU_TYPE` | `L4` | Kullanılacak GPU tipi |
| `MAX_CONTAINERS` | `10` | En fazla kaç GPU konteyner açılabileceği |
| `MIN_CONTAINERS` | `8` | Sıcak tutulacak minimum konteyner sayısı |
| `BUFFER_CONTAINERS` | `2` | Modal'in ekstra tampon konteyner hedefi |
| `SCALEDOWN_WINDOW` | `300` | Boş kalan konteynerlerin ne kadar süre sonra kapanacağı, saniye |
| `MAX_INPUTS` | `16` | Bir konteynerin aynı anda kabul edeceği maksimum input |
| `TARGET_INPUTS` | `8` | Modal'in hedeflediği eşzamanlı input sayısı |
| `MAX_QUEUE_SIZE` | `64` | Konteyner içi kuyruk limiti |
| `BATCH_MAX_SIZE` | `8` | Bir batch içindeki maksimum request sayısı |
| `BATCH_WAIT_MS` | `10` | Yeni request toplamak için batch bekleme süresi, milisaniye |
| `DEFAULT_VOICE` | `af_heart` | Varsayılan voice |
| `DEFAULT_SPEED` | `1.0` | Varsayılan konuşma hızı |
| `LANG_CODE` | `a` | Kokoro pipeline dil kodu |
| `MAX_TEXT_LENGTH` | `120` | `text` alanının izin verilen maksimum uzunluğu |
| `LOCK_TO_DEFAULT_VOICE` | `1` | `1` ise voice değiştirilemez |
| `LOCK_TO_DEFAULT_SPEED` | `1` | `1` ise speed değiştirilemez |

### Örnek deploy komutu

Tek satırlık değişken geçişiyle deploy:

```bash
APP_NAME=kokoro-tts-prod \
ENDPOINT_LABEL=kokoro-tts \
GPU_TYPE=L4 \
MIN_CONTAINERS=8 \
MAX_CONTAINERS=10 \
BUFFER_CONTAINERS=2 \
MAX_INPUTS=16 \
TARGET_INPUTS=8 \
BATCH_MAX_SIZE=8 \
BATCH_WAIT_MS=10 \
MAX_QUEUE_SIZE=64 \
DEFAULT_VOICE=af_heart \
DEFAULT_SPEED=1.0 \
MAX_TEXT_LENGTH=120 \
LOCK_TO_DEFAULT_VOICE=1 \
LOCK_TO_DEFAULT_SPEED=1 \
modal deploy modal_kokoro_service.py
```

## 7. GPU Opsiyonları

Kodda `gpu=GPU_TYPE` kullanılıyor. Bu nedenle `GPU_TYPE` değişkenine Modal'in desteklediği GPU adlarından biri verilebilir.

Bu servis için pratikte en anlamlı seçenekler:

- `T4`: en ucuz baseline, throughput daha düşük
- `L4`: fiyat/performans açısından iyi başlangıç noktası
- `A10`: L4'e göre bazı iş yüklerinde daha güçlü alternatif
- `A100`: daha yüksek throughput ve daha büyük batch denemeleri için
- `H100`: bu servis tipi için genelde pahalı, ancak en güçlü seçenek

Kısa TTS istekleri için genelde şu şekilde düşünebilirsiniz:

- Maliyet öncelikliyse: `T4` veya `L4`
- Dengeli başlangıç istiyorsanız: `L4`
- Daha agresif benchmark hedefi varsa: `A10` veya `A100`

Notlar:

- Bu repodaki varsayılan seçim `L4`
- GPU mevcutluğu workspace ve bölgeye göre değişebilir
- Modal birden fazla fallback GPU da destekler, ancak bu dosyada tek string kullanılıyor

Referanslar:

- Modal GPU guide: https://modal.com/docs/guide/gpu
- Modal GPU karşılaştırması: https://modal.com/blog/gpu-types
- Fallback GPU örneği: https://frontend.modal.com/docs/examples/gpu_fallbacks

## 8. Endpoint'ler

### `GET /healthz`

Servisin ayakta olduğunu ve aktif config değerlerini gösterir.

Örnek:

```bash
curl https://YOUR-ENDPOINT.modal.run/healthz
```

Dönen alanlar arasında şunlar vardır:

- `ok`
- `gpu_type`
- `max_containers`
- `min_containers`
- `batch_max_size`
- `batch_wait_ms`
- `target_inputs`
- `max_inputs`
- `max_text_length`
- `lock_to_default_voice`
- `lock_to_default_speed`

### `POST /warmup`

Modeli ve pipeline'i ısındırmak için bir deneme request'i gönderir.

```bash
curl -X POST https://YOUR-ENDPOINT.modal.run/warmup
```

### `POST /tts`

Standart TTS endpoint'i. Çıktı `audio/wav`.

İstek gövdesi:

```json
{
  "text": "Hello from Kokoro",
  "voice": "af_heart",
  "speed": 1.0,
  "format": "wav"
}
```

`curl` örneği:

```bash
curl -X POST https://YOUR-ENDPOINT.modal.run/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The cat is sleeping.",
    "voice": "af_heart",
    "speed": 1.0,
    "format": "wav"
  }' \
  --output out.wav
```

Python örneği:

```python
import requests

resp = requests.post(
    "https://YOUR-ENDPOINT.modal.run/tts",
    json={
        "text": "The cat is sleeping.",
        "voice": "af_heart",
        "speed": 1.0,
        "format": "wav",
    },
    timeout=30,
)
resp.raise_for_status()

with open("out.wav", "wb") as f:
    f.write(resp.content)
```

Olası hatalar:

- `400 voice_locked_for_benchmark`
- `400 speed_locked_for_benchmark`
- `429 container_queue_full`

### `POST /tts/stream`

Streaming endpoint'i. Bu endpoint `wav` dosyası dönmez; `24000 Hz`, mono, `pcm_s16le` parçaları stream eder.

İstek:

```json
{
  "text": "Hello from Kokoro",
  "voice": "af_heart",
  "speed": 1.0,
  "format": "wav"
}
```

Dönüş başlıkları:

- `Content-Type: audio/L16; rate=24000; channels=1`
- `X-Audio-Format: pcm_s16le`
- `X-Sample-Rate: 24000`
- `X-Channels: 1`

Not:

- Request içinde `format` alanı yine `wav` olmalı
- Buna rağmen dönen veri `wav` container değil, ham PCM stream'dir
- Canlı oynatma ya da düşük gecikmeli istemci tarafı tüketim için uygundur

## 9. Fonksiyonların Mantığı

### `BatchingEngine.start()`

- `torch` ve Kokoro sınıflarını yükler
- Modeli GPU varsa `cuda`, yoksa `cpu` üzerine alır
- Phoneme pipeline ve stream pipeline'i oluşturur
- Varsayılan voice pack'i bellekte hazırlar
- Batch worker task'ini başlatır
- İlk warmup request'ini çalıştırır

### `BatchingEngine.enqueue()`

- Kuyruk doluysa `queue_full` hatası fırlatır
- Her request için bir `Future` oluşturur
- Request'i kuyruğa ekler
- Sonucu asenkron şekilde bekler

### `BatchingEngine._batch_loop()`

- Kuyruktan ilk request'i alır
- `BATCH_WAIT_MS` süresi boyunca ek request toplar
- Batch boyutu `BATCH_MAX_SIZE` sınırına ulaşınca ya da süre dolunca işleme geçer

### `BatchingEngine._run_batch()`

- Batch içindeki request'leri `voice`, `speed`, `audio_format` kombinasyonuna göre gruplar
- Her grup için phoneme üretir
- Tek seferde `_synthesize_batch()` çağırır
- Sonucu ilgili request future'larına yazar

### `BatchingEngine._phonemize_one()`

- Metni phoneme dizisine çevirir
- Sonuçları LRU benzeri bir cache içinde tutar
- Aynı metin tekrar geldiğinde phonemization maliyetini azaltır

### `BatchingEngine._get_voice_pack()`

- Voice pack daha önce yüklenmemişse pipeline üzerinden yükler
- GPU belleğinde saklar
- Tekrar kullanır

### `BatchingEngine._synthesize_batch()`

- Phoneme'leri model token'larına çevirir
- Tensor'ları pad ederek ortak batch yapısına getirir
- Voice referans embedding'lerini hazırlar
- `_forward_batch()` ile GPU üzerinde toplu inference yapar
- Sonucu `soundfile` ile `wav` byte dizisine çevirir

### `BatchingEngine._forward_batch()`

- Kokoro modelinin alt katmanlarını doğrudan batch olarak çağırır
- Duration tahmini yapar
- Alignment tensor'larını oluşturur
- Decoder ile ses çıkışını üretir

Bu bölüm, servisin yüksek throughput almasını sağlayan asıl kritik noktadır. Kod, tek tek `KPipeline(...)(text)` çağrısı yapmak yerine daha alt seviyede gerçek batch forward pass uygular.

### `BatchingEngine.stream_synthesize()`

- Streaming pipeline üzerinden parçalı audio üretir
- Float audio'yu `int16 PCM` formatına çevirir
- Chunk chunk istemciye yollar

### `serve()`

- FastAPI uygulamasını kurar
- `startup` olayında engine'i başlatır
- `shutdown` olayında engine'i durdurur
- Tüm HTTP endpoint'lerini kaydeder

## 10. Benchmark ve Örnek Yük Testi

Repo içinde hazır bir yük testi script'i vardır: [scripts/load_test.py](/Users/sabrierendagdelen/Desktop/MEDYANES/Kokoro%20Modal/scripts/load_test.py)

Örnek kullanım:

```bash
python scripts/load_test.py \
  --url https://YOUR-ENDPOINT.modal.run/tts \
  --levels 1,10,50,100 \
  --requests-per-level 200
```

Warmup ile test:

```bash
python scripts/load_test.py \
  --url https://YOUR-ENDPOINT.modal.run/tts \
  --levels 1,10,50,100 \
  --warmup-requests 20 \
  --requests-per-level 200 \
  --output-file results/run.json
```

Burst testi:

```bash
python scripts/load_test.py \
  --url https://YOUR-ENDPOINT.modal.run/tts \
  --levels 1,10,50,100 \
  --burst-only
```

Script şu metrikleri raporlar:

- `avg_ms`
- `p50_ms`
- `p95_ms`
- `p99_ms`
- `max_ms`
- `ok`
- `fail`

## 11. Önerilen Başlangıç Ayarları

Kısa metin ve benchmark odaklı kullanım için makul başlangıç:

- `GPU_TYPE=L4`
- `MIN_CONTAINERS=8`
- `MAX_CONTAINERS=10`
- `BUFFER_CONTAINERS=2`
- `MAX_INPUTS=16`
- `TARGET_INPUTS=8`
- `MAX_QUEUE_SIZE=64`
- `BATCH_MAX_SIZE=8`
- `BATCH_WAIT_MS=10`
- `DEFAULT_VOICE=af_heart`
- `DEFAULT_SPEED=1.0`
- `MAX_TEXT_LENGTH=120`

## 12. Sınırlar ve Dikkat Edilecek Noktalar

- Batch endpoint şu anda sadece `wav` çıkışı destekliyor
- Stream endpoint `wav` değil ham PCM16 döndürüyor
- Varsayılan konfigürasyon tek voice ve tek speed benchmark'ı kolaylaştırmak için kilitli
- `MAX_TEXT_LENGTH` aşıldığında request Pydantic seviyesinde reddedilir
- Konteyner içi kuyruk dolarsa `429` döner
- `LANG_CODE` değiştirmek tek başına yeterli olmayabilir; ilgili phonemizer ve model uyumu gerekir

## 13. Hangi Durumda Hangi Ayarı Değiştirmeli?

- `429` artıyorsa: `MAX_QUEUE_SIZE`, `MIN_CONTAINERS`, `MAX_CONTAINERS` ve `TARGET_INPUTS` değerlerine bakın
- GPU boş görünüyor ama throughput düşükse: `BATCH_MAX_SIZE` artırılabilir
- Gecikme artıyorsa: `BATCH_WAIT_MS` ve `BATCH_MAX_SIZE` düşürülebilir
- Çok farklı voice kullanılacaksa: voice pack önbellek davranışını izlemeniz gerekir
- Uzun metin desteği gerekiyorsa: `MAX_TEXT_LENGTH` ve model context sınırları birlikte ele alınmalıdır

## 14. Hızlı Başlangıç

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal serve modal_kokoro_service.py
```

Yeni bir terminalde:

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"The cat is sleeping.","voice":"af_heart","speed":1.0,"format":"wav"}' \
  --output out.wav
```

Kalıcı deploy için:

```bash
modal deploy modal_kokoro_service.py
```
