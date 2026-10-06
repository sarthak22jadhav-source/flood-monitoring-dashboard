"""Boot the real Streamlit server, hit health + index, then shut down."""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request

PORT = 8601
out: list[str] = []

proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py",
     f"--server.port={PORT}", "--server.headless=true"],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
)

health_ok = index_ok = False
health_body = index_len = ""
try:
    for _ in range(45):  # up to 90 s
        time.sleep(2)
        try:
            with urllib.request.urlopen(
                f"http://localhost:{PORT}/_stcore/health", timeout=3
            ) as r:
                health_body = r.read().decode()
                health_ok = (r.status == 200 and health_body.strip() == "ok")
            break
        except Exception:
            continue
    if health_ok:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}", timeout=5) as r:
                index_len = str(len(r.read()))
                index_ok = r.status == 200
        except Exception as e:
            index_len = f"ERROR {e}"
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

print(f"HEALTH: ok={health_ok} body={health_body!r}")
print(f"INDEX:  ok={index_ok} bytes={index_len}")
sys.exit(0 if (health_ok and index_ok) else 1)
