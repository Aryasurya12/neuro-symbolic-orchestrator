"""Quick check: is NVIDIA_API_KEY set, and does it actually work?
Run from the project root:  .\\venv\\Scripts\\python.exe test_nvidia_key.py
"""
import json
import os
import time
import urllib.error
import urllib.request

def load_config():
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("NVIDIA_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = os.environ.get("NVIDIA_MODEL") or os.environ.get("OPENROUTER_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
    
    url = base_url.rstrip("/") + "/chat/completions"
    source = "environment / .env"
    return key, url, model, source


key, URL, MODEL, source = load_config()
if not key:
    print("FAIL: NVIDIA_API_KEY / OPENROUTER_API_KEY not found in environment or .env")
    print('Fix: add NVIDIA_API_KEY to .env')
    raise SystemExit(1)

print(f"Key found via {source}: {key[:8]}...{key[-4:]} (length {len(key)})")
print(f"Target URL: {URL}")
print(f"Target Model: {MODEL}")


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
