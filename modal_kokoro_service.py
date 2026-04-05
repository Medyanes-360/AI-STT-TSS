import os

import modal

from combined_kokoro_service import create_web_app


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
    .pip_install("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")
)

app = modal.App(APP_NAME)


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
    return create_web_app(title="Kokoro TTS/STT", runtime_label="modal")
