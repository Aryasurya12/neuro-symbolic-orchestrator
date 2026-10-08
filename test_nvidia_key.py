"""Quick check: is NVIDIA_API_KEY set, and does it actually work?
Run from the project root:  .\\venv\\Scripts\\python.exe test_nvidia_key.py
"""
import json
import os
import time
import urllib.error
import urllib.request

MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"  # same model the comparative script uses
URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def load_key():
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key, "environment variable"
    if os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("NVIDIA_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'"), ".env file"
    return None, None


key, source = load_key()
if not key:
    print("FAIL: NVIDIA_API_KEY not found in environment or .env")
    print('Fix: add a line  NVIDIA_API_KEY=nvapi-xxxxxxxx  to .env (no spaces, no quotes needed)')
    raise SystemExit(1)

print(f"Key found via {source}: {key[:8]}...{key[-4:]} (length {len(key)})")
if not key.startswith("nvapi-"):
    print("WARNING: NVIDIA keys normally start with 'nvapi-'. Check for a copy/paste error.")

body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "max_tokens": 10,
    "temperature": 0,
}).encode("utf-8")

req = urllib.request.Request(
    URL, data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)

t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ms = (time.perf_counter() - t0) * 1000
    text = data["choices"][0]["message"]["content"]
    print(f"SUCCESS in {ms:.0f} ms. Model replied: {text!r}")
except urllib.error.HTTPError as e:
    detail = e.read().decode("utf-8", errors="replace")[:300]
    print(f"FAIL: HTTP {e.code}")
    print(f"Server said: {detail}")
    hints = {
        401: "Key is invalid or expired. Generate a new one at build.nvidia.com.",
        403: "Key is valid but not allowed to use this model.",
        404: f"Model name '{MODEL}' not available to your account. Try a different model name.",
        429: "Rate limit hit. Wait a minute and retry.",
    }
    print("Likely cause:", hints.get(e.code, "See server message above."))
except Exception as e:
    print(f"FAIL: could not reach NVIDIA at all ({type(e).__name__}: {e})")
    print("Likely cause: no internet, firewall, or proxy blocking integrate.api.nvidia.com")
