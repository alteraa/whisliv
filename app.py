"""Browser microphone -> Silero VAD -> power-aware faster-whisper GPU STT."""
from __future__ import annotations

import asyncio
import os
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

_CUDA_DLL_HANDLES = []


def configure_cuda_dlls() -> None:
    if os.name != "nt":
        return
    base = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = [Path(os.environ["CUDA_PATH"]) / "bin"] if os.environ.get("CUDA_PATH") else []
    candidates.extend(base.glob("NVIDIA GPU Computing Toolkit/CUDA/v12*/bin"))
    for path in candidates:
        if path.is_dir():
            os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(path)))


configure_cuda_dlls()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

ROOT = Path(__file__).parent
RATE = 16_000
VAD_FRAME = 512
PARTIAL_WINDOW = RATE * 4
PRE_ROLL_FRAMES = 8
MIN_UTTERANCE = RATE // 4
END_SILENCE_FRAMES = int(0.8 * RATE / VAD_FRAME)
PARTIAL_BEAM_SIZE = int(os.getenv("WHISPER_PARTIAL_BEAM_SIZE", "1"))
FINAL_BEAM_SIZE = int(os.getenv("WHISPER_FINAL_BEAM_SIZE", "3"))

MODE_PRESETS: dict[str, float | None] = {
    "final_only": None,  # Lowest GPU activity: one inference at end of speech.
    "balanced": 1.0,     # Good responsiveness with reduced GPU wakeups.
    "realtime": 0.5,     # Fastest progressive feedback, highest GPU activity.
}
TRANSCRIPTION_MODE = os.getenv("WHISPER_TRANSCRIPTION_MODE", "balanced").lower()
if TRANSCRIPTION_MODE not in MODE_PRESETS:
    raise ValueError(f"WHISPER_TRANSCRIPTION_MODE must be one of: {', '.join(MODE_PRESETS)}")
interval_override = os.getenv("WHISPER_PARTIAL_INTERVAL_SECONDS")
PARTIAL_INTERVAL_S = MODE_PRESETS[TRANSCRIPTION_MODE] if interval_override is None else float(interval_override)
if TRANSCRIPTION_MODE == "final_only":
    PARTIAL_INTERVAL_S = None
elif PARTIAL_INTERVAL_S is None or PARTIAL_INTERVAL_S <= 0:
    raise ValueError("WHISPER_PARTIAL_INTERVAL_SECONDS must be greater than zero")
PARTIAL_STEP = int(RATE * PARTIAL_INTERVAL_S) if PARTIAL_INTERVAL_S else None

app = FastAPI(title="Live Whisper with VAD")
whisper_model = vad_model = None
model_lock = asyncio.Lock()


def models():
    global whisper_model, vad_model
    if vad_model is None:
        print("Loading Silero VAD...", flush=True)
        vad_model = load_silero_vad()
    if whisper_model is None:
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")
        print(f"Loading Whisper small on {device} ({compute})...", flush=True)
        whisper_model = WhisperModel("small", device=device, compute_type=compute)
    return vad_model, whisper_model


def decode(audio: np.ndarray, beam_size: int) -> tuple[str, float]:
    began = perf_counter()
    _, model = models()
    segments, _ = model.transcribe(
        audio.astype(np.float32) / 32768.0,
        beam_size=beam_size,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip(), perf_counter() - began


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "mode": TRANSCRIPTION_MODE, "partial_interval_seconds": PARTIAL_INTERVAL_S}


async def vad_worker(ws: WebSocket, queue: asyncio.Queue[np.ndarray | None]):
    vad, _ = models()
    vad.reset_states()
    pending = np.empty(0, np.int16)
    utterance = np.empty(0, np.int16)
    pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES)
    silence = partial_at = count = partial_revision = 0
    total_time = total_audio = 0.0
    speaking = False
    partial_latest: tuple[int, np.ndarray] | None = None
    partial_task: asyncio.Task | None = None

    async def metrics(elapsed: float, audio_s: float) -> None:
        nonlocal count, total_time, total_audio
        count += 1
        total_time += elapsed
        total_audio += audio_s
        await ws.send_json({
            "type": "metrics", "inference_count": count,
            "last_inference_ms": round(elapsed * 1000, 1),
            "total_inference_ms": round(total_time * 1000, 1),
            "audio_seconds": round(total_audio, 2),
            "rtf": round(total_time / total_audio, 3) if total_audio else 0,
        })

    async def infer(audio: np.ndarray, beam_size: int, final: bool, revision: int | None = None) -> None:
        try:
            async with model_lock:
                text, elapsed = await asyncio.to_thread(decode, audio, beam_size)
            await metrics(elapsed, len(audio) / RATE)
            if text and (final or revision == partial_revision):
                await ws.send_json({"type": "transcript", "text": text, "final": final})
        except Exception as exc:
            await ws.send_json({"type": "error", "message": f"Whisper inference error: {exc}"})

    async def progressive_worker() -> None:
        nonlocal partial_latest
        while partial_latest is not None:
            revision, audio = partial_latest
            partial_latest = None
            await infer(audio, PARTIAL_BEAM_SIZE, False, revision)

    try:
        while True:
            incoming = await queue.get()
            if incoming is None:
                return
            pending = np.concatenate((pending, incoming))
            while len(pending) >= VAD_FRAME:
                frame, pending = pending[:VAD_FRAME], pending[VAD_FRAME:]
                with torch.no_grad():
                    probability = float(vad(torch.from_numpy(frame.astype(np.float32) / 32768.0), RATE))

                if probability >= 0.5:
                    if not speaking:
                        speaking = True
                        utterance = np.concatenate(tuple(pre_roll)) if pre_roll else np.empty(0, np.int16)
                        silence = count = 0
                        partial_at = len(utterance)
                        partial_revision += 1
                        partial_latest = None
                        total_time = total_audio = 0.0
                        await ws.send_json({"type": "speech_start"})
                        await ws.send_json({"type": "metrics_reset"})
                    silence = 0
                    utterance = np.concatenate((utterance, frame))
                    if PARTIAL_STEP is not None and len(utterance) - partial_at >= PARTIAL_STEP:
                        partial_at = len(utterance)
                        partial_revision += 1
                        partial_latest = (partial_revision, utterance[-PARTIAL_WINDOW:].copy())
                        if partial_task is None or partial_task.done():
                            partial_task = asyncio.create_task(progressive_worker())

                elif speaking:
                    utterance = np.concatenate((utterance, frame))
                    silence += 1
                    if silence >= END_SILENCE_FRAMES:
                        speech_end = max(0, len(utterance) - silence * VAD_FRAME)
                        final_audio = utterance[:speech_end]
                        partial_revision += 1
                        partial_latest = None
                        if partial_task is not None and not partial_task.done():
                            await partial_task
                        if len(final_audio) >= MIN_UTTERANCE:
                            await infer(final_audio, FINAL_BEAM_SIZE, True)
                        await ws.send_json({"type": "speech_end"})
                        pre_roll.clear()
                        for start in range(max(speech_end, len(utterance) - PRE_ROLL_FRAMES * VAD_FRAME), len(utterance), VAD_FRAME):
                            pre_roll.append(utterance[start:start + VAD_FRAME].copy())
                        speaking = False
                        utterance = np.empty(0, np.int16)

                else:
                    pre_roll.append(frame.copy())
    finally:
        vad.reset_states()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=64)
    worker = asyncio.create_task(vad_worker(ws, queue))
    await ws.send_json({
        "type": "ready", "sample_rate": RATE, "vad": "silero",
        "mode": TRANSCRIPTION_MODE, "partial_interval_seconds": PARTIAL_INTERVAL_S,
    })
    try:
        while True:
            await queue.put(np.frombuffer(await ws.receive_bytes(), np.int16).copy())
    except WebSocketDisconnect:
        worker.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
