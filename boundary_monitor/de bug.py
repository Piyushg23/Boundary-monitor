
"""
debug_groq.py — Drop this in your boundary_monitor folder and run it.
It will tell you exactly what is wrong with the Groq setup.
Run from inside the boundary_monitor folder:
    python debug_groq.py
"""
 
import os
import sys
from pathlib import Path
 
print("\n" + "="*60)
print("  BOUNDARY MONITOR — GROQ DEBUG")
print("="*60)
 
# 1. Where is this script?
here = Path(__file__).resolve().parent
print(f"\n[1] Script running from: {here}")
 
# 2. Find .env
env_path = here / ".env"
print(f"\n[2] Looking for .env at: {env_path}")
if env_path.exists():
    print("    FOUND .env file")
    raw = env_path.read_text()
    print(f"    File contents ({len(raw)} bytes):")
    for i, line in enumerate(raw.splitlines(), 1):
        if "GROQ" in line.upper():
            # Mask the key value
            if "=" in line:
                k, _, v = line.partition("=")
                v = v.strip()
                if len(v) > 12:
                    masked = v[:8] + "..." + v[-4:]
                else:
                    masked = repr(v)
                print(f"      Line {i}: {k}={masked}")
            else:
                print(f"      Line {i}: {line}")
        else:
            print(f"      Line {i}: {line}")
else:
    print("    NOT FOUND — you need to create this file!")
    print(f"    Create: {env_path}")
    print("    With contents:")
    print("        GROQ_API_KEY=gsk_your_actual_key_here")
 
# 3. Try loading dotenv
print("\n[3] Testing python-dotenv...")
try:
    from dotenv import load_dotenv
    print("    python-dotenv is installed OK")
    result = load_dotenv(env_path, override=True)
    print(f"    load_dotenv returned: {result}")
except ImportError:
    print("    NOT INSTALLED — run: pip install python-dotenv")
 
# 4. Check the env var
print("\n[4] Checking os.environ for GROQ_API_KEY...")
key = os.environ.get("GROQ_API_KEY", "")
if key:
    masked = key[:8] + "..." + key[-4:] if len(key) > 12 else repr(key)
    print(f"    Value: {masked}  (length: {len(key)})")
    placeholders = {"your_groq_api_key_here", "YOUR_GROQ_API_KEY_HERE"}
    if key in placeholders:
        print("    WARNING: This is the placeholder value, not a real key!")
        print("    Edit your .env and replace it with your actual key from console.groq.com")
    else:
        print("    Looks like a real key.")
else:
    print("    EMPTY — variable not set or .env not loaded")
 
# 5. Test groq package
print("\n[5] Testing groq package...")
try:
    from groq import Groq
    print("    groq package is installed OK")
    if key and key not in {"", "your_groq_api_key_here", "YOUR_GROQ_API_KEY_HERE"}:
        print("    Attempting to create Groq client...")
        try:
            client = Groq(api_key=key)
            print("    Groq client created OK")
            print("\n[6] Sending a test message to Groq API...")
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=30,
                    messages=[{"role": "user", "content": "Reply with just: GROQ OK"}],
                )
                print(f"    API response: {resp.choices[0].message.content.strip()}")
                print("\n    SUCCESS — Groq is working correctly!")
            except Exception as e:
                print(f"    API call failed: {e}")
                if "401" in str(e) or "auth" in str(e).lower() or "invalid" in str(e).lower():
                    print("    Your API key is invalid or expired.")
                    print("    Get a new one at: https://console.groq.com")
        except Exception as e:
            print(f"    Client creation failed: {e}")
    else:
        print("    Skipping API test (no valid key found)")
except ImportError:
    print("    NOT INSTALLED — run: pip install groq")
 
print("\n" + "="*60)
print("  Copy and paste the output above and share it.")
print("="*60 + "\n")