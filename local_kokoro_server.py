import uvicorn

from combined_kokoro_service import create_web_app

app = create_web_app(title="Local Kokoro TTS/STT", runtime_label="local")


if __name__ == "__main__":
    uvicorn.run("local_kokoro_server:app", host="127.0.0.1", port=8000, reload=False)
