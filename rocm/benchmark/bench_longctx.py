#!/usr/bin/env python3
"""Long-context benchmark for TabbyAPI (exllamav3 backend).

Fires streaming chat/completions with cold, unique prompts and parses the
engine's own per-request log line from the engine's stdout log file (the run
script is redirected there by the harness) for accurate prefill/decode/TTFT/
draft-acceptance metrics. stdlib only; run from the host running the engine.
"""
import argparse, json, os, random, re, time, urllib.request, urllib.error

# Connection defaults target a local TabbyAPI; override via env for other setups:
#   BENCH_MODEL    served model id          BENCH_URL     full .../chat/completions URL
#   BENCH_KEYFILE  path to TabbyAPI api_tokens.yml
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-27B-EXL3-3.5bpw")
CHAT_URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8096/v1/chat/completions")
KEYFILE = os.path.expanduser(os.environ.get("BENCH_KEYFILE", "~/tabbyAPI/api_tokens.yml"))
MODELS_URL = (CHAT_URL[: -len("/chat/completions")]
              if CHAT_URL.endswith("/chat/completions")
              else CHAT_URL.rsplit("/", 1)[0]) + "/models"

# engine per-request completion line:
# 22:06:02.875 INFO:  #23 chat/completions (stream): 1,683 tokens generated at 73.8 T/s · prompt
#   32,158 tokens, 97% cached, 926 new in 1.19 s (778 T/s) · first token 1.20 s, total 24.0 s
#   · draft 1148/3745 accepted (31%)
LINE_RE = re.compile(
    r"#(?P<req>\d+).*?:\s*(?P<gen>[\d,]+) tokens generated at (?P<dec>\d+(?:\.\d+)?) T/s"
    r".*?prompt (?P<prompt>[\d,]+) tokens, (?:none|(?P<cached>\d+)%) cached, "
    r"(?P<new>[\d,]+) new in (?P<pref>\d+(?:\.\d+)?) s"
    r".*?first token (?P<ttft>\d+(?:\.\d+)?) s, total (?P<total>\d+(?:\.\d+)?) s"
    r"(?:.*?draft (?P<dacc>[\d,]+)/(?P<dden>[\d,]+) accepted(?: \((?P<dpct>\d+)%\))?)?"
)

def _i(x):
    return int(x.replace(",", "")) if x else None

def read_key():
    for line in open(KEYFILE):
        if line.strip().startswith("api_key:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("no api_key in " + KEYFILE)

def make_prompt(target_chars, rng):
    parts = [f"<BENCH {rng.randrange(1<<62):016x} {rng.randrange(1<<62):016x}>"]
    n = len(parts[0])
    while n < target_chars:
        w = str(rng.randrange(100_000_000, 999_999_999))  # 9-digit, low-compressibility
        parts.append(w); n += len(w) + 1
    parts.append("Question: compute 17 times 23. Answer with just the number.")
    return " ".join(parts)

def post(prompt, max_tokens, key):
    """Returns (wall_s, first_s, nchunk, err). err is None on success."""
    body = json.dumps({"model": MODEL, "stream": True, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time(); first = None; nchunk = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                if first is None:
                    first = time.time() - t0
                nchunk += 1
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        return time.time() - t0, 0.0, 0, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return time.time() - t0, 0.0, 0, f"{type(e).__name__}: {e}"
    return time.time() - t0, (first or 0.0), nchunk, None

def read_engine(path, want_gt, tries=40, delay=0.5):
    """Parse newest completion line in log with request number > want_gt."""
    for _ in range(tries):
        try:
            txt = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            txt = ""
        lines = [l for l in txt.splitlines() if "tokens generated at" in l]
        if lines:
            m = LINE_RE.search(lines[-1])
            if m:
                d = m.groupdict()
                if _i(d["req"]) and _i(d["req"]) > want_gt:
                    pref = float(d["pref"]); new = _i(d["new"]) or 0
                    acc = _i(d.get("dacc")); den = _i(d.get("dden"))
                    pct = _i(d.get("dpct"))
                    if pct is None and acc is not None and den:
                        pct = round(100.0 * acc / den)
                    ratio = round(acc / den, 3) if (acc is not None and den) else None
                    return {"req": _i(d["req"]), "prompt": _i(d["prompt"]),
                            "cached_pct": _i(d["cached"]) or 0, "new_tokens": new,
                            "prefill_s": pref,
                            "prefill_tps": round(new / pref, 1) if pref > 0 else None,
                            "ttft_s": float(d["ttft"]), "total_s": float(d["total"]),
                            "gen_tokens": _i(d["gen"]), "decode_tps": float(d["dec"]),
                            "draft_acc": acc, "draft_den": den,
                            "draft_pct": pct, "draft_ratio": ratio}
        time.sleep(delay)
    return None

def wait_up(key, tries=80):
    req = urllib.request.Request(MODELS_URL,
                                 headers={"Authorization": "Bearer " + key})
    for _ in range(tries):
        try:
            urllib.request.urlopen(req, timeout=5); return True
        except Exception:
            time.sleep(3)
    return False

def max_req(path):
    """Highest request number from any completion line already in the log."""
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return 0
    reqs = []
    for l in txt.splitlines():
        if "tokens generated at" in l:
            m = LINE_RE.search(l)
            if m and _i(m.group("req")) is not None:
                reqs.append(_i(m.group("req")))
    return max(reqs) if reqs else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--log", required=True, help="engine stdout log file")
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", required=True)
    a = ap.parse_args()
    key = read_key()
    rng = random.Random((os.getpid() << 20) ^ time.time_ns())  # unique prompts each run -> always cold prefill
    if not wait_up(key):
        open(a.done, "w").write("server-not-up\n"); raise SystemExit("server not up")
    if not os.path.exists(a.log):
        open(a.log, "a").close()

    # calibrate chars-per-prompt-token; ignore any requests already in the log
    base = max_req(a.log)
    cal = make_prompt(20_000, rng)
    _, _, _, cerr = post(cal, 1, key)
    if cerr:
        open(a.done, "w").write("calibration-failed: " + cerr + "\n")
        raise SystemExit("calibration failed: " + cerr)
    eng = read_engine(a.log, base)
    if not eng or not eng["prompt"]:
        open(a.done, "w").write("calibration-unparsed\n"); raise SystemExit("calibration unparsed")
    c_ratio = len(cal) / eng["prompt"]
    print(f"# calibration: {len(cal)} chars = {eng['prompt']} prompt-tokens -> "
          f"{c_ratio:.3f} chars/token (cached={eng['cached_pct']}%)", flush=True)
    req_max = eng["req"]

    targets = [int(x) for x in a.ladder.split()]
    out = open(a.out, "a", buffering=1)
    for tgt in targets:
        for rep in range(a.reps):
            prompt = make_prompt(int(tgt * c_ratio), rng)
            wall, ttft, nchunk, err = post(prompt, a.max_tokens, key)
            row = {"tag": a.tag, "target": tgt, "rep": rep, "chars": len(prompt),
                   "client_ttft_s": round(ttft, 2), "client_wall_s": round(wall, 2)}
            if err:
                row["error"] = err
                out.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
                time.sleep(1.0); continue
            e = read_engine(a.log, req_max)
            if e:
                req_max = e["req"]; row.update(e)
            else:
                row["error"] = "log-line-not-captured"
            out.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
            time.sleep(1.5)
    out.close()
    open(a.done, "w").write("ok\n")

if __name__ == "__main__":
    main()
