#!/usr/bin/env python3
"""
run.py
Production Application Launcher for AI Object Remover.

Automates the complete application lifecycle:
1. Detects virtual environment and Python runtime.
2. Checks port 8000 availability.
3. Detects if a healthy AI Object Remover backend is already running on port 8000:
   - If yes: Reuses the existing backend and prevents duplicate processes.
   - If port 8000 is occupied by a foreign application: Reports conflict clearly without killing it.
4. If free: Starts FastAPI backend via Uvicorn.
5. Monitors startup: If backend fails, captures and displays the REAL exception output.
6. Waits for /api/health to confirm backend is online and the LaMa ONNX model is loaded.
7. Automatically opens the photo studio editor in the user's default browser.
8. Manages graceful shutdown on Ctrl+C.
"""

import os
import sys
import time
import socket
import argparse
import threading
import subprocess
import webbrowser
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

try:
    import urllib.request
    import urllib.error
    import json
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "app" / "backend"
MODELS_DIR = PROJECT_ROOT / "models"


def find_python_interpreter() -> str:
    """Finds the most appropriate Python interpreter (virtualenv preferred)."""
    candidates = [
        PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT.parent / ".venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def check_port_listening(host: str, port: int) -> bool:
    """Checks if a TCP port is currently open and listening."""
    if not (1 <= port <= 65535):
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def probe_health_endpoint(host: str, port: int, timeout: float = 2.0) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Attempts HTTP GET on /api/health.
    Returns (is_our_backend, response_data).
    """
    url = f"http://{host}:{port}/api/health"
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Object-Remover-Launcher/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                # Verify application signature
                if data.get("app") == "ai-object-remover" or (data.get("provider") and "model" in data):
                    return True, data
    except Exception:
        pass
    return False, None


def inspect_port(host: str, port: int) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Evaluates the state of the target port.
    Returns:
      'FREE'            -> Port is not in use.
      'OUR_BACKEND'     -> Port belongs to an existing AI Object Remover backend instance.
      'FOREIGN_OCCUPIED'-> Port is in use by another application.
    """
    if not check_port_listening(host, port):
        return "FREE", None

    is_our_backend, health_data = probe_health_endpoint(host, port, timeout=2.0)
    if is_our_backend:
        return "OUR_BACKEND", health_data
    return "FOREIGN_OCCUPIED", None


def launch_application(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    detach: bool = False,
):
    print("=" * 65)
    print("   AI Photo Studio — Intelligent Application Launcher")
    print("=" * 65)

    python_bin = find_python_interpreter()
    print(f"[Launcher] Runtime Python: {python_bin}")
    print(f"[Launcher] Project Root:   {PROJECT_ROOT}")

    # Step 1: Inspect target port
    port_state, health_data = inspect_port(host, port)

    if port_state == "OUR_BACKEND":
        model_ready = health_data.get("model") == "ready" if health_data else False
        print(f"[Launcher] Existing AI Object Remover backend detected on http://{host}:{port}")
        print(f"[Launcher] Model state: {'Ready' if model_ready else 'Loading'}")
        print("[Launcher] Reusing existing healthy backend instance (no duplicate process spawned).")

        app_url = f"http://{host}:{port}/"
        if open_browser:
            print(f"[Launcher] Opening photo studio in default browser: {app_url}")
            webbrowser.open(app_url)
        print("[Launcher] Ready to use.")
        return 0

    elif port_state == "FOREIGN_OCCUPIED":
        print(f"\n[Launcher Error] Port {port} is already in use by a different process.")
        print(f"[Launcher Error] AI Object Remover requires port {port} to serve.")
        print("[Launcher Error] Safety protocol: Conflicting foreign process will NOT be terminated.")
        print(f"                 Please stop the application occupying port {port} and retry.\n")
        return 1

    # Step 2: Port is FREE -> Start Backend Process
    print(f"[Launcher] Port {port} is free. Starting FastAPI backend...")

    env = os.environ.copy()
    existing_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{str(PROJECT_ROOT)}{os.pathsep}{existing_py_path}"
        if existing_py_path
        else str(PROJECT_ROOT)
    )

    cmd = [
        python_bin,
        "-m",
        "uvicorn",
        "app.backend.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    captured_logs = []
    log_lock = threading.Lock()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        print(f"\n[Launcher Error] Failed to execute backend process: {e}\n")
        return 1

    def stream_reader():
        if proc.stdout is None:
            return
        try:
            for line in iter(proc.stdout.readline, ""):
                with log_lock:
                    captured_logs.append(line)
                # Keep output accessible in terminal
                sys.stdout.write(f"  [backend] {line}")
                sys.stdout.flush()
        except Exception:
            pass

    log_thread = threading.Thread(target=stream_reader, daemon=True)
    log_thread.start()

    # Step 3: Wait for Backend & Model readiness
    print(f"[Launcher] Waiting for backend and LaMa ONNX model to initialize...")
    max_wait_seconds = 45.0
    start_time = time.time()
    backend_ready = False
    model_ready = False
    health_payload: Optional[Dict[str, Any]] = None

    while time.time() - start_time < max_wait_seconds:
        # Check if process terminated prematurely
        ret_code = proc.poll()
        if ret_code is not None:
            time.sleep(0.2)
            with log_lock:
                err_text = "".join(captured_logs[-25:])
            print(f"\n[Launcher Error] Backend process terminated unexpectedly with exit code {ret_code}.")
            print("Captured process output:")
            print("-" * 65)
            print(err_text.strip() or "No output captured.")
            print("-" * 65)
            return 1

        is_our, data = probe_health_endpoint(host, port, timeout=1.0)
        if is_our and data:
            backend_ready = True
            health_payload = data
            if data.get("model") == "ready":
                model_ready = True
                break

        time.sleep(0.5)

    if not backend_ready:
        print(f"\n[Launcher Error] Timed out waiting for backend to respond on http://{host}:{port}/api/health.")
        proc.terminate()
        return 1

    if not model_ready:
        model_state = health_payload.get("model_state", "loading") if health_payload else "loading"
        print(f"[Launcher Warning] Backend is online, but model report is: {model_state}")
    else:
        print(f"[Launcher] Backend is online: http://{host}:{port}")
        print(f"[Launcher] LaMa ONNX Inpainting Model: READY (Loaded in memory)")

    app_url = f"http://{host}:{port}/"
    if open_browser:
        print(f"[Launcher] Opening editor in default browser: {app_url}")
        webbrowser.open(app_url)

    print("=" * 65)
    print("   Application is fully active and ready to edit photos!")
    if detach:
        print("   Backend running in background detached mode.")
        print("=" * 65)
        return 0

    print("   Press Ctrl+C in this terminal anytime to cleanly stop the server.")
    print("=" * 65)

    # Keep alive and supervise child process
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                print(f"[Launcher] Backend process ended with code {ret}.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Launcher] Shutdown requested by user. Terminating backend...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[Launcher] Clean shutdown complete.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="AI Object Remover Production Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    parser.add_argument("--detach", action="store_true", help="Run backend in background without waiting in terminal")
    args = parser.parse_args()

    sys.exit(launch_application(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        detach=args.detach,
    ))


if __name__ == "__main__":
    main()
