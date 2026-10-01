# Live Whisper web app

Browser microphone audio is sent over WebSocket to Silero VAD and faster-whisper `small` on the remote GPU. The app emits progressive and final transcripts.

```powershell
uv sync
uv run python app.py
```

Forward port 8000 over SSH, then open `http://localhost:8000` locally:

```powershell
ssh -L 8000:localhost:8000 user@remote-host
```

## Power and latency modes

Set these before starting the server:

```powershell
# Lowest GPU activity: one final inference after speech ends.
$env:WHISPER_TRANSCRIPTION_MODE = "final_only"

# Default: one progressive inference per second, then a final result.
$env:WHISPER_TRANSCRIPTION_MODE = "balanced"

# Fastest UI updates: one progressive inference every 0.5 seconds.
$env:WHISPER_TRANSCRIPTION_MODE = "realtime"

# Optional override for balanced or realtime modes.
$env:WHISPER_PARTIAL_INTERVAL_SECONDS = "1.5"

uv run python app.py
```

The active mode and partial interval appear in the browser UI. VAD uses 16 kHz frames and ends an utterance after roughly 0.8 seconds of silence. CUDA `float16` is the default; set `WHISPER_COMPUTE_TYPE=int8_float16` to benchmark a lower-VRAM alternative.
