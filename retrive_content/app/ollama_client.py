import requests
import time
import json

OLLAMA_URL = "http://localhost:11434/api/generate"  # تأكد المسار صحيح

def query_ollama(model: str, prompt: str, stream: bool = False, timeout: int = 20):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    start = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    duration = time.time() - start

    parsed = None
    try:
        parsed = resp.json()
    except json.JSONDecodeError:
        raw = resp.text.strip()
        objs = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    first = line.index("{")
                    last = line.rfind("}")
                    if first != -1 and last != -1 and last > first:
                        candidate = line[first:last+1]
                        objs.append(json.loads(candidate))
                except Exception:
                    continue
        if objs:
            parsed = objs[-1]  
        else:
            parsed = {"output": resp.text}  

    text = extract_text(parsed)
    uncertainty = detect_uncertainty(text)
    return {
        "answer": text,
        "raw": parsed,
        "duration_sec": duration,
        "uncertain": uncertainty
    }

def extract_text(resp_json):
    if isinstance(resp_json, dict):
        if "output" in resp_json:
            return resp_json["output"]
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if isinstance(choice, dict):
                return choice.get("message", {}).get("content") or choice.get("text", "") or str(choice)
            return str(choice)
        return json.dumps(resp_json)
    return str(resp_json)

def detect_uncertainty(text: str) -> bool:
    heuristics = [
        "i'm not sure",
        "i don’t know",
        "i do not know",
        "cannot confirm",
        "i might be wrong",
        "as far as i know",
        "i think",
        "maybe"
    ]
    low = text.lower()
    return any(h in low for h in heuristics)