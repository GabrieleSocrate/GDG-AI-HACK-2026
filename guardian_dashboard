from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


app = FastAPI(title="Desk Guardian Dashboard")

_state_lock = threading.Lock()
_latest_jpeg: Optional[bytes] = None
_status = {
    "phase": "BOOTING",
    "owner_present": False,
    "protected_object": "LAPTOP",
    "alarm_active": False,
    "alarm_reason": "",
    "distance_text": "--",
    "last_update": time.time(),
}


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Desk Guardian | Workspace Security Camera</title>
  <style>
    :root {
      --bg: #070b14;
      --panel: rgba(255, 255, 255, 0.08);
      --text: #f8fafc;
      --muted: #94a3b8;
      --line: rgba(255, 255, 255, 0.14);
      --green: #22c55e;
      --yellow: #f59e0b;
      --red: #ef4444;
      --blue: #38bdf8;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.22), transparent 32%),
        radial-gradient(circle at 70% 20%, rgba(34, 197, 94, 0.15), transparent 30%),
        linear-gradient(135deg, #050816 0%, #0f172a 55%, #020617 100%);
      overflow-x: hidden;
    }

    .shell {
      width: min(1220px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 24px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .logo {
      width: 46px;
      height: 46px;
      border-radius: 15px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, #38bdf8, #22c55e);
      box-shadow: 0 18px 50px rgba(56, 189, 248, 0.25);
      font-weight: 900;
      color: #020617;
    }

    .brand h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.03em;
    }

    .brand p {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.62);
      backdrop-filter: blur(10px);
      color: #cbd5e1;
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 16px var(--green);
    }

    .hero {
      display: grid;
      grid-template-columns: 1.42fr 0.58fr;
      gap: 20px;
      align-items: stretch;
    }

    .video-card, .side-card, .feature-card {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.105), rgba(255,255,255,0.055));
      border-radius: 28px;
      box-shadow: 0 25px 80px rgba(0, 0, 0, 0.33);
      backdrop-filter: blur(14px);
    }

    .video-card {
      overflow: hidden;
      min-height: 540px;
      position: relative;
    }

    .video-top {
      height: 58px;
      padding: 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      background: rgba(2, 6, 23, 0.45);
    }

    .video-top strong {
      font-size: 14px;
    }

    .video-wrap {
      position: relative;
      height: calc(100% - 58px);
      min-height: 482px;
      background: #020617;
      display: grid;
      place-items: center;
    }

    .video-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #020617;
    }

    .watermark {
      position: absolute;
      left: 18px;
      bottom: 18px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(2, 6, 23, 0.72);
      border: 1px solid rgba(255,255,255,0.12);
      font-size: 12px;
      color: #cbd5e1;
    }

    .side-card {
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .side-card h2 {
      margin: 0;
      font-size: 28px;
      letter-spacing: -0.04em;
      line-height: 1.04;
    }

    .side-card .subtitle {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }

    .status-box {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(2, 6, 23, 0.35);
      padding: 16px;
    }

    .status-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 11px 0;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    .status-line:last-child { border-bottom: 0; }
    .status-line span:first-child { color: var(--muted); font-size: 13px; }
    .status-line span:last-child { font-weight: 700; font-size: 13px; text-align: right; }

    .badge {
      padding: 7px 10px;
      border-radius: 999px;
      font-weight: 800;
      font-size: 12px;
      letter-spacing: 0.02em;
    }

    .badge.safe { color: #052e16; background: var(--green); }
    .badge.armed { color: #451a03; background: var(--yellow); }
    .badge.alarm { color: white; background: var(--red); }

    .features {
      margin-top: 20px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }

    .feature-card {
      padding: 18px;
    }

    .feature-card .icon {
      width: 40px;
      height: 40px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      background: rgba(56, 189, 248, 0.14);
      color: var(--blue);
      margin-bottom: 14px;
      font-size: 20px;
    }

    .feature-card h3 {
      margin: 0 0 8px;
      font-size: 16px;
    }

    .feature-card p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    footer {
      margin-top: 18px;
      color: #64748b;
      font-size: 12px;
      text-align: center;
    }

    @media (max-width: 900px) {
      header { flex-direction: column; align-items: flex-start; }
      .hero { grid-template-columns: 1fr; }
      .features { grid-template-columns: 1fr; }
      .video-card { min-height: 420px; }
      .video-wrap { min-height: 360px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">
        <div class="logo">DG</div>
        <div>
          <h1>Desk Guardian</h1>
          <p>Spatial AI camera security for libraries, study rooms and coworking desks.</p>
        </div>
      </div>
      <div class="pill"><span class="dot"></span> Live OAK camera stream</div>
    </header>

    <section class="hero">
      <main class="video-card">
        <div class="video-top">
          <strong>Protected workspace live feed</strong>
          <span class="pill" id="phase-pill">BOOTING</span>
        </div>
        <div class="video-wrap">
          <img src="/video_feed" alt="Desk Guardian live stream" />
          <div class="watermark">OAK-D vision · person re-ID · protected object monitoring</div>
        </div>
      </main>

      <aside class="side-card">
        <div>
          <h2>Leave your laptop. Keep control.</h2>
          <p class="subtitle">
            Desk Guardian watches your desk when you step away. It recognizes the owner,
            tracks the protected laptop and raises an alert when an unknown person gets too close.
          </p>
        </div>

        <div class="status-box">
          <div class="status-line">
            <span>System state</span>
            <span id="system-state" class="badge armed">BOOTING</span>
          </div>
          <div class="status-line">
            <span>Owner</span>
            <span id="owner-state">Unknown</span>
          </div>
          <div class="status-line">
            <span>Protected item</span>
            <span id="protected-object">LAPTOP</span>
          </div>
          <div class="status-line">
            <span>Distance</span>
            <span id="distance">--</span>
          </div>
          <div class="status-line">
            <span>Alarm</span>
            <span id="alarm">Inactive</span>
          </div>
        </div>
      </aside>
    </section>

    <section class="features">
      <div class="feature-card">
        <div class="icon">👤</div>
        <h3>Owner recognition</h3>
        <p>The system learns the desk owner during enrollment and stays disarmed while the owner is present.</p>
      </div>
      <div class="feature-card">
        <div class="icon">💻</div>
        <h3>Protected laptop tracking</h3>
        <p>The camera tracks the protected computer and visualizes proximity in real time.</p>
      </div>
      <div class="feature-card">
        <div class="icon">🚨</div>
        <h3>Smart alarm logic</h3>
        <p>Alerts are triggered only when the owner is absent and an unknown person approaches the device.</p>
      </div>
    </section>

    <footer>Desk Guardian prototype · Designed for libraries, classrooms and coworking areas</footer>
  </div>

  <script>
    async function refreshStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        const systemState = document.getElementById('system-state');
        const phasePill = document.getElementById('phase-pill');
        const ownerState = document.getElementById('owner-state');
        const protectedObject = document.getElementById('protected-object');
        const distance = document.getElementById('distance');
        const alarm = document.getElementById('alarm');

        phasePill.textContent = data.phase || 'UNKNOWN';
        protectedObject.textContent = data.protected_object || 'LAPTOP';
        distance.textContent = data.distance_text || '--';

        if (data.alarm_active) {
          systemState.textContent = 'ALARM';
          systemState.className = 'badge alarm';
          alarm.textContent = data.alarm_reason || 'Active';
        } else if (data.owner_present) {
          systemState.textContent = 'DISARMED';
          systemState.className = 'badge safe';
          alarm.textContent = 'Blocked by owner presence';
        } else {
          systemState.textContent = 'ARMED';
          systemState.className = 'badge armed';
          alarm.textContent = 'Inactive';
        }

        ownerState.textContent = data.owner_present ? 'Present / recently seen' : 'Absent';
      } catch (err) {
        console.error(err);
      }
    }

    setInterval(refreshStatus, 500);
    refreshStatus();
  </script>
</body>
</html>
"""


def update_dashboard_frame(frame, jpeg_quality: int = 82) -> None:
    global _latest_jpeg

    if frame is None:
        return

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return

    with _state_lock:
        _latest_jpeg = encoded.tobytes()
        _status["last_update"] = time.time()


def update_dashboard_status(
    *,
    phase: Optional[str] = None,
    owner_present: Optional[bool] = None,
    protected_object: Optional[str] = None,
    alarm_active: Optional[bool] = None,
    alarm_reason: Optional[str] = None,
    distance_text: Optional[str] = None,
) -> None:
    with _state_lock:
        if phase is not None:
            _status["phase"] = phase
        if owner_present is not None:
            _status["owner_present"] = bool(owner_present)
        if protected_object is not None:
            _status["protected_object"] = protected_object
        if alarm_active is not None:
            _status["alarm_active"] = bool(alarm_active)
        if alarm_reason is not None:
            _status["alarm_reason"] = alarm_reason
        if distance_text is not None:
            _status["distance_text"] = distance_text
        _status["last_update"] = time.time()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/api/status")
def api_status() -> JSONResponse:
    with _state_lock:
        return JSONResponse(dict(_status))


def _make_placeholder_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Desk Guardian is waiting for the OAK camera stream...",
        (90, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (230, 230, 230),
        2,
    )
    cv2.putText(
        frame,
        "Run main_oak_guardian.py and keep this browser window open.",
        (90, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (150, 150, 150),
        2,
    )
    return frame


def _mjpeg_generator():
    placeholder = None

    while True:
        with _state_lock:
            jpeg = _latest_jpeg

        if jpeg is None:
            if placeholder is None:
                placeholder_frame = _make_placeholder_frame()
                ok, encoded = cv2.imencode(".jpg", placeholder_frame)
                placeholder = encoded.tobytes() if ok else b""
            jpeg = placeholder

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def start_dashboard(host: str = "0.0.0.0", port: int = 8000) -> None:
    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print(f"[WEB] Desk Guardian dashboard running at http://localhost:{port}")
